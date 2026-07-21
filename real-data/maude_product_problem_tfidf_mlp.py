from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from urllib3.util.retry import Retry

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

BASE_URL = "https://api.fda.gov"
DEFAULT_SEED = 7
DEFAULT_MAX_PER_CLASS = 3000

NARRATIVE_KEYS = {
    "text",
    "mdr_text",
    "manufacturer_narrative",
    "description",
    "description_of_event_or_problem",
    "event_description",
    "event_problem",
    "problem_text",
    "narrative",
    "device_narrative",
    "additional_text",
    "additional_narrative",
}

EXCLUDE_KEYS = {
    "report_id",
    "event_type",
    "event_type_raw",
    "product_problem_flag",
    "date_report",
    "manufacturer_name",
    "brand_name",
    "device_name",
    "product_code",
    "report_number",
    "mdr_report_key",
    "accession_number",
    "patient_sequence_number",
    "patient_age",
    "patient_sex",
}

BOILERPLATE_PATTERNS = [
    r"additional manufacturer narrative",
    r"description of event or problem",
    r"manufacturer report",
    r"initial submission",
    r"user facility report",
    r"health professional",
    r"lay user/patient",
    r"please refer to",
    r"if information is provided in the future",
    r"further information",
    r"supplemental report",
    r"no further information",
    r"device history record",
    r"lot history record",
    r"complaint history",
    r"not returned for analysis",
]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(
        {
            "User-Agent": "maude-product-problem-tfidf-mlp/1.0",
            "Accept": "application/json",
        }
    )
    return s


def fda_get_json(
    session: requests.Session,
    endpoint: str,
    params: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}.json"
    params = dict(params)
    if api_key:
        params["api_key"] = api_key
    r = session.get(url, params=params, timeout=90)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def first_nonempty(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip():
                return v.strip()
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, list):
            for item in v:
                out = first_nonempty(item)
                if out:
                    return out
        elif isinstance(v, dict):
            continue
    return ""


def recursive_strings(obj: Any) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        if obj.strip():
            yield obj.strip()
        return
    if isinstance(obj, (int, float)):
        yield str(obj)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from recursive_strings(item)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in EXCLUDE_KEYS:
                continue
            if k in NARRATIVE_KEYS or k.endswith("_text") or k.endswith("_narrative"):
                yield from recursive_strings(v)
        return


def extract_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []

    for key in sorted(record.keys()):
        if key in EXCLUDE_KEYS:
            continue
        if key in NARRATIVE_KEYS or key.endswith("_text") or key.endswith("_narrative"):
            parts.extend(list(recursive_strings(record.get(key))))

    # Some MAUDE records keep narratives nested a level deeper.
    for nested_key in ["mdr_text", "text"]:
        if nested_key in record:
            parts.extend(list(recursive_strings(record.get(nested_key))))

    clean: List[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", str(p)).strip()
        if not p:
            continue
        low = p.lower()
        if any(pat in low for pat in BOILERPLATE_PATTERNS):
            continue
        clean.append(p)

    text = " || ".join(clean)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iter_openfda_records(
    session: requests.Session,
    endpoint: str,
    query: str,
    max_records: int,
    page_size: int,
    api_key: Optional[str],
) -> Iterable[Dict[str, Any]]:
    skip = 0
    fetched = 0
    page_size = max(1, min(page_size, 1000))
    pbar = tqdm(total=max_records, desc=f"Downloading {endpoint} records", unit="rec")

    try:
        while fetched < max_records:
            params = {"limit": min(page_size, max_records - fetched), "skip": skip}
            if query:
                params["search"] = query
            payload = fda_get_json(session, endpoint, params=params, api_key=api_key)
            results = payload.get("results", []) or []
            if not results:
                break
            for rec in results:
                yield rec
                fetched += 1
                pbar.update(1)
                if fetched >= max_records:
                    break
            if len(results) < params["limit"]:
                break
            skip += len(results)
            time.sleep(0.08)
    finally:
        pbar.close()


def build_dataset(
    session: requests.Session,
    max_per_class: int,
    page_size: int,
    api_key: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    query_map = {"N": "product_problem_flag:N", "Y": "product_problem_flag:Y"}

    for flag in ["N", "Y"]:
        query = query_map[flag]
        records = list(
            iter_openfda_records(
                session=session,
                endpoint="device/event",
                query=query,
                max_records=max_per_class,
                page_size=page_size,
                api_key=api_key,
            )
        )
        for i, rec in enumerate(records):
            text = extract_text(rec)
            if not text:
                continue
            rows.append(
                {
                    "report_id": first_nonempty(
                        rec.get("report_id"),
                        rec.get("mdr_report_key"),
                        rec.get("accession_number"),
                        f"{flag}_{i}",
                    ),
                    "label": int(flag == "Y"),
                    "product_problem_flag": flag,
                    "text": text,
                    "date_report": first_nonempty(rec.get("date_report"), rec.get("date_received"), rec.get("date_of_event")),
                    "manufacturer_name": first_nonempty(rec.get("manufacturer_name"), rec.get("manufacturer"), rec.get("manufacturer_d_name")),
                    "raw_json": json.dumps(rec, ensure_ascii=False),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No MAUDE records were collected.")

    df = df.drop_duplicates(subset=["report_id"] if "report_id" in df.columns else None).copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()
    df = df.reset_index(drop=True)
    return df


@dataclass
class TfIdfMlpTextModel:
    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    scaler: StandardScaler
    clf: MLPClassifier

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.tfidf.transform(list(texts))
        X = self.svd.transform(X)
        X = self.scaler.transform(X)
        return self.clf.predict_proba(X)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)


def build_feature_matrix(
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    test_texts: Sequence[str],
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    svd_components: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )

    X_train_tfidf = tfidf.fit_transform(list(train_texts))
    X_val_tfidf = tfidf.transform(list(val_texts))
    X_test_tfidf = tfidf.transform(list(test_texts))

    n_features = X_train_tfidf.shape[1]
    if n_features <= 1:
        raise RuntimeError("TF-IDF produced too few features. Try lowering min_df or increasing max_features.")
    n_comp = max(2, min(svd_components, n_features - 1, X_train_tfidf.shape[0] - 1))

    svd = TruncatedSVD(n_components=n_comp, random_state=DEFAULT_SEED)
    X_train = svd.fit_transform(X_train_tfidf)
    X_val = svd.transform(X_val_tfidf)
    X_test = svd.transform(X_test_tfidf)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, tfidf, svd, scaler


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> MLPClassifier:
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, class_weights)}
    sample_weight = np.array([weight_map[int(y)] for y in y_train], dtype=np.float32)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=seed,
        verbose=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def selective_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    pos_prob = proba[:, 1]
    y_pred = (pos_prob >= 0.5).astype(int)
    confident = np.maximum(proba[:, 0], proba[:, 1]) >= threshold
    coverage = float(confident.mean())

    if confident.sum() == 0:
        return {
            "coverage": 0.0,
            "selective_accuracy": None,
            "selective_f1": None,
            "selective_precision": None,
            "selective_recall": None,
            "selective_balanced_accuracy": None,
        }

    yt = y_true[confident]
    yp = y_pred[confident]
    precision, recall, _, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)

    return {
        "coverage": coverage,
        "selective_accuracy": float(accuracy_score(yt, yp)),
        "selective_f1": float(f1_score(yt, yp, zero_division=0)),
        "selective_precision": float(precision),
        "selective_recall": float(recall),
        "selective_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
    }


def choose_threshold_by_val(y_val: np.ndarray, proba_val: np.ndarray, target_coverage: float) -> float:
    candidates = np.linspace(0.50, 0.99, 50)
    best = 0.80
    for t in candidates:
        cov = (np.maximum(proba_val[:, 0], proba_val[:, 1]) >= t).mean()
        if cov >= target_coverage:
            best = float(t)
    return best


def make_group_splits(
    df: pd.DataFrame,
    seed: int,
    group_col: str = "report_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"{group_col} column is required for grouped splitting.")

    groups = df[group_col].fillna("").astype(str).values
    y = df["label"].astype(int).values

    for attempt in range(100):
        rs = seed + attempt
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
        trainval_idx, test_idx = next(gss1.split(df, y, groups=groups))
        trainval = df.iloc[trainval_idx].copy()
        test = df.iloc[test_idx].copy()

        if len(trainval["label"].unique()) < 2 or len(test["label"].unique()) < 2:
            continue

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs + 1000)
        train_idx, val_idx = next(gss2.split(trainval, trainval["label"].values, groups=trainval[group_col].values))
        train = trainval.iloc[train_idx].copy()
        val = trainval.iloc[val_idx].copy()

        if len(train["label"].unique()) < 2 or len(val["label"].unique()) < 2:
            continue

        return train, val, test

    raise RuntimeError("Could not find a stable grouped split. Try lowering max-per-class or changing the seed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_maude_product_problem")
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENFDA_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-coverage", type=float, default=0.65)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--save-dataset-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    df = build_dataset(
        session=session,
        max_per_class=args.max_per_class,
        page_size=args.page_size,
        api_key=args.api_key or None,
    )

    df = df.reset_index(drop=True)
    df.to_csv(outdir / "dataset_all.csv", index=False)
    df.to_json(outdir / "dataset_all.jsonl", orient="records", lines=True, force_ascii=False)

    class_counts = df["label"].value_counts().sort_index().to_dict()
    class_labels = {0: "No product problem", 1: "Product problem"}

    print("Running maude_product_problem_tfidf_mlp.py with the following parameters:")
    print(f"OUTDIR: {outdir}")
    print(f"OPENFDA_API_KEY: {'<set>' if args.api_key else '<empty>'}")
    print(f"MAX_PER_CLASS: {args.max_per_class}")
    print(f"TARGET_COVERAGE: {args.target_coverage}")
    print(f"Seed: {args.seed}")
    print(f"Class counts: {class_counts}")

    if args.save_dataset_only:
        print(f"Saved outputs to: {outdir}")
        return

    train_df, val_df, test_df = make_group_splits(df, seed=args.seed, group_col="report_id")

    X_train, X_val, X_test, tfidf, svd, scaler = build_feature_matrix(
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
    )

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    clf = train_mlp(X_train, y_train, seed=args.seed)
    proba_val = clf.predict_proba(X_val)
    proba_test = clf.predict_proba(X_test)

    chosen_threshold = choose_threshold_by_val(y_val, proba_val, target_coverage=args.target_coverage)

    yhat_test = (proba_test[:, 1] >= 0.5).astype(int)
    std_acc = float(accuracy_score(y_test, yhat_test))
    std_f1 = float(f1_score(y_test, yhat_test, zero_division=0))
    std_bal_acc = float(balanced_accuracy_score(y_test, yhat_test))

    try:
        std_auc = float(roc_auc_score(y_test, proba_test[:, 1]))
    except Exception:
        std_auc = None

    best_selective = selective_metrics(y_true=y_test, proba=proba_test, threshold=chosen_threshold)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    threshold_results: List[Dict[str, Any]] = []
    for threshold in thresholds:
        sel = selective_metrics(y_true=y_test, proba=proba_test, threshold=float(threshold))
        threshold_results.append(
            {
                "threshold": float(threshold),
                "coverage": sel["coverage"],
                "accuracy": sel["selective_accuracy"],
                "f1": sel["selective_f1"],
                "precision": sel["selective_precision"],
                "recall": sel["selective_recall"],
                "balanced_accuracy": sel["selective_balanced_accuracy"],
            }
        )
    pd.DataFrame(threshold_results).to_csv(outdir / "threshold_sweep.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test,
            "prob_no_problem": proba_test[:, 0],
            "prob_problem": proba_test[:, 1],
            "predicted_label": yhat_test,
        }
    )
    prediction_df.to_csv(outdir / "test_predictions.csv", index=False)

    metrics = {
        "standard": {
            "accuracy": std_acc,
            "f1": std_f1,
            "balanced_accuracy": std_bal_acc,
            "roc_auc": std_auc,
            "confusion_matrix": confusion_matrix(y_test, yhat_test).tolist(),
        },
        "selective": {
            "threshold": chosen_threshold,
            "coverage": best_selective["coverage"],
            "selective_accuracy": best_selective["selective_accuracy"],
            "selective_f1": best_selective["selective_f1"],
            "selective_precision": best_selective["selective_precision"],
            "selective_recall": best_selective["selective_recall"],
            "selective_balanced_accuracy": best_selective["selective_balanced_accuracy"],
        },
        "validation_coverage_target": args.target_coverage,
        "validation_chosen_threshold": chosen_threshold,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "label_balance": {
            "overall_pos_rate": float(df["label"].mean()) if len(df) else None,
            "train_pos_rate": float(train_df["label"].mean()) if len(train_df) else None,
            "val_pos_rate": float(val_df["label"].mean()) if len(val_df) else None,
            "test_pos_rate": float(test_df["label"].mean()) if len(test_df) else None,
        },
        "class_counts": {class_labels[k]: int(v) for k, v in class_counts.items()},
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "task": "MAUDE narrative product problem classification",
        "label_definition": "label=1 if product_problem_flag is Y, else 0",
        "text_source": "MAUDE narrative text from openFDA device event reports",
        "abstention": "predict only when max class probability exceeds threshold",
        "threshold": chosen_threshold,
        "model": "TF-IDF + TruncatedSVD + MLP",
        "feature_notes": [
            "text only",
            "no metadata",
            "no BERT",
            "no logistic regression",
        ],
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(
        {
            "model": clf,
            "tfidf": tfidf,
            "svd": svd,
            "scaler": scaler,
            "threshold": chosen_threshold,
            "feature_names": tfidf.get_feature_names_out().tolist(),
        },
        outdir / "model.joblib",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {outdir}")
    print(f"Saved threshold sweep to: {outdir / 'threshold_sweep.csv'}")
    print(f"Saved predictions to: {outdir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
