from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.decomposition import TruncatedSVD
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from urllib3.util.retry import Retry

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

BASE_URL = "https://api.fda.gov"
DEFAULT_SEED = 7

# Words and phrases that can make the label too easy to infer.
# Keep this list conservative but broad enough to strip the most obvious leakage.
LEAKY_PATTERNS = [
    r"\bproduct problem\b",
    r"\bproblem\b",
    r"\bmalfunction\b",
    r"\bfailure\b",
    r"\bdefect\b",
    r"\bdevice issue\b",
    r"\bdevice problem\b",
    r"\bnot related\b",
    r"\bdevice related\b",
    r"\bcaused\b",
    r"\bcontributed\b",
    r"\bassociated with\b",
    r"\bdeath\b",
    r"\bdied\b",
    r"\bdeceased\b",
    r"\bexpired\b",
    r"\binjury\b",
    r"\binjured\b",
    r"\bhar\w*\b",
    r"\badverse event\b",
    r"\badverse events\b",
    r"\bserious injury\b",
    r"\bpatient expired\b",
    r"\bpatient died\b",
    r"\bpronounced deceased\b",
    r"\bcause of death\b",
]

BOILERPLATE_PATTERNS = [
    r"additional manufacturer narrative",
    r"description of event or problem",
    r"manufacturer report",
    r"initial submission",
    r"user facility report",
    r"health professional",
    r"lay user/patient",
    r"lay user patient",
    r"please refer to",
    r"if information is provided in the future",
    r"supplemental report",
    r"this information is submitted pursuant to",
    r"the device was not returned",
    r"the product was not returned",
]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "maude-product-problem-redacted/1.0",
            "Accept": "application/json",
        }
    )
    return session


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
    resp = session.get(url, params=params, timeout=90)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    return resp.json()


def iter_openfda_records(
    session: requests.Session,
    endpoint: str,
    query: str = "",
    max_records: int = 1000,
    page_size: int = 100,
    api_key: Optional[str] = None,
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
            time.sleep(0.05)
    finally:
        pbar.close()


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


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [flatten_text(v) for v in value]
        return " || ".join([p for p in parts if p])
    if isinstance(value, dict):
        parts = []
        for _, v in value.items():
            t = flatten_text(v)
            if t:
                parts.append(t)
        return " || ".join(parts)
    return str(value)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def redact_leaky_phrases(text: str) -> str:
    out = text
    for pat in LEAKY_PATTERNS:
        out = re.sub(pat, " [REDACTED] ", out, flags=re.IGNORECASE)
    return out


def remove_boilerplate(text: str) -> str:
    out = text
    for pat in BOILERPLATE_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    return out


def clean_text(text: str) -> str:
    text = flatten_text(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = remove_boilerplate(text)
    text = redact_leaky_phrases(text)
    text = re.sub(r"\b(B|b)\(\d+\)\b", " [REDACTED] ", text)
    text = re.sub(r"\b\d{2,}\b", " [NUM] ", text)
    text = normalize_ws(text)
    return text


def parse_flag(value: Any) -> Optional[int]:
    s = first_nonempty(value).upper()
    if not s:
        return None
    if s in {"Y", "YES", "1", "TRUE"}:
        return 1
    if s in {"N", "NO", "0", "FALSE"}:
        return 0
    return None


def parse_record(record: Dict[str, Any]) -> Dict[str, Any]:
    event_type = first_nonempty(record.get("event_type"))
    text = first_nonempty(record.get("mdr_text"), record.get("text"), record.get("description"), record.get("device_problem_text"))
    if not text:
        text = flatten_text(record)

    row = {
        "report_id": first_nonempty(record.get("report_id"), record.get("mdr_report_key"), record.get("event_key")),
        "date_report": first_nonempty(record.get("date_report"), record.get("date_received"), record.get("date_of_event")),
        "manufacturer_name": first_nonempty(record.get("manufacturer_name"), record.get("manufacturer")),
        "product_problem_flag": parse_flag(record.get("product_problem_flag")),
        "event_type_raw": event_type,
        "text_raw": text,
        "text": clean_text(text),
        "raw_json": json.dumps(record, ensure_ascii=False),
    }
    return row


def build_dataset(
    session: requests.Session,
    api_key: Optional[str],
    max_per_class: int,
    page_size: int,
) -> pd.DataFrame:
    # We first collect more than enough records, then balance down to the two classes.
    # The openFDA endpoint contains many more positives than negatives.
    records = list(
        iter_openfda_records(
            session,
            "device/event",
            query="product_problem_flag:(Y OR N)",
            max_records=max_per_class * 8,
            page_size=page_size,
            api_key=api_key,
        )
    )
    if not records:
        raise RuntimeError("No MAUDE records returned by openFDA.")

    rows: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="Processing MAUDE records", unit="rec"):
        row = parse_record(rec)
        if row["product_problem_flag"] is None:
            continue
        if not row["text"]:
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable MAUDE records after parsing.")

    # Balance the two classes for a fair selective-classification benchmark.
    parts = []
    for label in (0, 1):
        sub = df[df["product_problem_flag"] == label].copy()
        if len(sub) == 0:
            raise RuntimeError(f"No examples found for class {label}.")
        if len(sub) > max_per_class:
            sub = sub.sample(n=max_per_class, random_state=DEFAULT_SEED)
        parts.append(sub)
    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=DEFAULT_SEED).reset_index(drop=True)
    balanced = balanced.rename(columns={"product_problem_flag": "label"})
    return balanced


def make_splits(
    df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval, test = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["label"],
    )
    train, val = train_test_split(
        trainval,
        test_size=0.20,
        random_state=seed + 1,
        stratify=trainval["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class TextPipeline:
    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    scaler: StandardScaler

    def fit_transform(self, train_texts: Sequence[str], val_texts: Sequence[str], test_texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_train_tfidf = self.tfidf.fit_transform(train_texts)
        X_val_tfidf = self.tfidf.transform(val_texts)
        X_test_tfidf = self.tfidf.transform(test_texts)

        X_train = self.svd.fit_transform(X_train_tfidf)
        X_val = self.svd.transform(X_val_tfidf)
        X_test = self.svd.transform(X_test_tfidf)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test


def build_feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tfidf_max_features: int,
    tfidf_ngram_min: int,
    tfidf_ngram_max: int,
    svd_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], TextPipeline]:
    tfidf = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(tfidf_ngram_min, tfidf_ngram_max),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=svd_components, random_state=DEFAULT_SEED)
    scaler = StandardScaler()
    pipe = TextPipeline(tfidf=tfidf, svd=svd, scaler=scaler)
    X_train, X_val, X_test = pipe.fit_transform(
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
    )
    feature_names = [f"svd_{i}" for i in range(X_train.shape[1])]
    return X_train, X_val, X_test, feature_names, pipe


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> MLPClassifier:
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
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    confident = np.max(proba, axis=1) >= threshold
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
    precision, recall, _, _ = precision_recall_fscore_support(
        yt,
        yp,
        average="binary",
        zero_division=0,
    )
    return {
        "coverage": coverage,
        "selective_accuracy": float(accuracy_score(yt, yp)),
        "selective_f1": float(f1_score(yt, yp, zero_division=0)),
        "selective_precision": float(precision),
        "selective_recall": float(recall),
        "selective_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
    }


def choose_threshold_by_val(y_val: np.ndarray, proba_val: np.ndarray, target_coverage: float) -> float:
    candidates = np.arange(0.50, 0.991, 0.01)
    best = float(candidates[-1])
    for t in candidates:
        cov = (np.max(proba_val, axis=1) >= t).mean()
        if cov >= target_coverage:
            best = float(t)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_maude_product_problem_redacted")
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENFDA_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-coverage", type=float, default=0.65)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--svd-components", type=int, default=300)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--save-dataset-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running maude_product_problem_redacted_tfidf_mlp.py with the following parameters:")
    print(f"OUTDIR: {outdir}")
    print(f"OPENFDA_API_KEY: {'<set>' if args.api_key else '<empty>'}")
    print(f"MAX_PER_CLASS: {args.max_per_class}")
    print(f"PAGE_SIZE: {args.page_size}")
    print(f"TARGET_COVERAGE: {args.target_coverage}")
    print(f"Seed: {args.seed}")

    session = make_session()
    df = build_dataset(
        session=session,
        api_key=args.api_key or None,
        max_per_class=args.max_per_class,
        page_size=args.page_size,
    )

    df.to_csv(outdir / "dataset_all.csv", index=False)
    df.to_json(outdir / "dataset_all.jsonl", orient="records", lines=True, force_ascii=False)

    label_map = {0: "No product problem", 1: "Product problem"}
    print(f"Class counts: {df['label'].value_counts().sort_index().to_dict()}")

    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    split_counts = {
        "train": train_df["label"].value_counts().sort_index().to_dict(),
        "val": val_df["label"].value_counts().sort_index().to_dict(),
        "test": test_df["label"].value_counts().sort_index().to_dict(),
    }

    if args.save_dataset_only:
        payload = {
            "n_rows": int(len(df)),
            "class_counts": {label_map[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "split_counts": split_counts,
        }
        with open(outdir / "dataset_summary.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(payload, indent=2))
        return

    X_train, X_val, X_test, feature_names, text_pipe = build_feature_matrix(
        train_df,
        val_df,
        test_df,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        svd_components=args.svd_components,
    )

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    clf = fit_mlp(X_train, y_train, seed=args.seed)
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

    sel = selective_metrics(y_test, proba_test, threshold=chosen_threshold)

    threshold_results: List[Dict[str, Any]] = []
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    for t in thresholds:
        res = selective_metrics(y_test, proba_test, threshold=float(t))
        threshold_results.append(
            {
                "threshold": float(t),
                "coverage": res["coverage"],
                "accuracy": res["selective_accuracy"],
                "f1": res["selective_f1"],
                "precision": res["selective_precision"],
                "recall": res["selective_recall"],
                "balanced_accuracy": res["selective_balanced_accuracy"],
            }
        )

    pd.DataFrame(threshold_results).to_csv(outdir / "threshold_sweep.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test,
            "prob_no_problem": proba_test[:, 0],
            "prob_problem": proba_test[:, 1],
            "predicted_label": yhat_test,
            "confident": np.max(proba_test, axis=1) >= chosen_threshold,
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
            "coverage": sel["coverage"],
            "selective_accuracy": sel["selective_accuracy"],
            "selective_f1": sel["selective_f1"],
            "selective_precision": sel["selective_precision"],
            "selective_recall": sel["selective_recall"],
            "selective_balanced_accuracy": sel["selective_balanced_accuracy"],
            "selective_confusion_matrix": confusion_matrix(y_test[np.max(proba_test, axis=1) >= chosen_threshold], yhat_test[np.max(proba_test, axis=1) >= chosen_threshold]).tolist() if np.max(proba_test, axis=1).any() else None,
        },
        "validation_coverage_target": args.target_coverage,
        "validation_chosen_threshold": chosen_threshold,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "label_balance": {
            "overall_pos_rate": float(df["label"].mean()),
            "train_pos_rate": float(train_df["label"].mean()),
            "val_pos_rate": float(val_df["label"].mean()),
            "test_pos_rate": float(test_df["label"].mean()),
        },
        "class_counts": {label_map[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "split_counts": {
            split: {label_map[k]: int(v) for k, v in counts.items()}
            for split, counts in split_counts.items()
        },
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "task": "MAUDE product problem detection from narrative",
        "label_definition": "product_problem_flag Y/N from openFDA MAUDE reports",
        "text_source": "MAUDE narrative text redacted to remove obvious label cues",
        "model": "TF-IDF + SVD + MLP",
        "features": feature_names,
        "redaction": {
            "leaky_patterns": LEAKY_PATTERNS,
            "boilerplate_patterns": BOILERPLATE_PATTERNS,
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(
        {
            "model": clf,
            "tfidf": text_pipe.tfidf,
            "svd": text_pipe.svd,
            "scaler": text_pipe.scaler,
            "threshold": chosen_threshold,
            "feature_names": feature_names,
        },
        outdir / "model.joblib",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {outdir}")
    print(f"Saved threshold sweep to: {outdir / 'threshold_sweep.csv'}")
    print(f"Saved predictions to: {outdir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
