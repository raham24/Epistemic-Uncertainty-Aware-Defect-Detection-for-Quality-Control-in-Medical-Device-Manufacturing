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
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Words/phrases that often leak the label directly in MAUDE narratives.
LEAKY_PATTERNS = [
    r"\bdeath\b",
    r"\bdied\b",
    r"\bdeceased\b",
    r"\bexpired\b",
    r"\binjury\b",
    r"\binjured\b",
    r"\bmalfunction\b",
    r"\bfailure\b",
    r"\bdefect(?:ive)?\b",
    r"\bproblem\b",
    r"\bcaus(?:e|ed|ing)\b",
    r"\bcontribut(?:e|ed|ing)\b",
    r"\bnot related\b",
    r"\bdevice related\b",
    r"\bproduct problem\b",
    r"\bno product problem\b",
    r"\bpatient expired\b",
    r"\bpatient died\b",
    r"\bpronounced deceased\b",
    r"\bno allegation\b",
    r"\bno evidence\b",
]

SECTION_MARKERS = [
    "Additional Manufacturer Narrative",
    "Description of Event or Problem",
    "Description of Event",
    "Manufacturer report",
    "Initial submission",
    "Additional Information",
    "Clinical Review",
    "Product Event Summary",
    "Manufacturer's Narrative",
    "Manufacturer Narrative",
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
            "User-Agent": "maude-product-problem-redacted-textonly/1.0",
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


def iter_device_event_records(
    session: requests.Session,
    query: str,
    max_records: int,
    page_size: int,
    api_key: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    skip = 0
    fetched = 0
    page_size = max(1, min(page_size, 1000))
    pbar = tqdm(total=max_records, desc="Downloading device/event records", unit="rec")
    try:
        while fetched < max_records:
            params = {"limit": min(page_size, max_records - fetched), "skip": skip, "search": query}
            payload = fda_get_json(session, "device/event", params=params, api_key=api_key)
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
            # Try common text-bearing keys before skipping.
            for k in ("text", "value", "description", "problem", "narrative"):
                if k in v:
                    out = first_nonempty(v.get(k))
                    if out:
                        return out
    return ""


def flatten_text(obj: Any) -> List[str]:
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
        return out
    if isinstance(obj, (int, float)):
        out.append(str(obj))
        return out
    if isinstance(obj, list):
        for item in obj:
            out.extend(flatten_text(item))
        return out
    if isinstance(obj, dict):
        # Prefer likely narrative fields.
        preferred = ["text", "value", "description", "narrative", "problem"]
        for key in preferred:
            if key in obj:
                out.extend(flatten_text(obj.get(key)))
        if out:
            return out
        for _, val in obj.items():
            out.extend(flatten_text(val))
        return out
    return out


def extract_raw_narrative(record: Dict[str, Any]) -> str:
    candidates: List[str] = []
    for key in (
        "mdr_text",
        "text",
        "description",
        "event_description",
        "additional_text",
        "narrative",
        "manufacturer_narrative",
    ):
        if key in record:
            candidates.extend(flatten_text(record.get(key)))
    if not candidates:
        # Fallback to the whole record, but skip obvious label fields.
        for key, val in record.items():
            if key in {"product_problem_flag", "event_type", "date_report", "report_number"}:
                continue
            if isinstance(val, (str, list, dict, int, float)):
                candidates.extend(flatten_text(val))
    text = " || ".join(candidates)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_before_markers(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    cut = len(text)
    for marker in SECTION_MARKERS:
        idx = lower.find(marker.lower())
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def redact_leaky_phrases(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    for pat in LEAKY_PATTERNS:
        cleaned = re.sub(pat, " [REDACTED] ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_text(text: str) -> str:
    text = truncate_before_markers(text)
    text = redact_leaky_phrases(text)
    return text


def parse_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip().upper()
    if s in {"Y", "YES", "1", "TRUE", "T"}:
        return 1
    if s in {"N", "NO", "0", "FALSE", "F"}:
        return 0
    return None


def parse_date(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    digits = re.sub(r"\D", "", s)
    return digits[:8] if len(digits) >= 8 else ""


def build_dataset(
    session: requests.Session,
    max_per_class: int,
    page_size: int,
    api_key: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    queries = {0: 'product_problem_flag:"N"', 1: 'product_problem_flag:"Y"'}
    for label, query in queries.items():
        count = 0
        for rec in iter_device_event_records(
            session=session,
            query=query,
            max_records=max_per_class,
            page_size=page_size,
            api_key=api_key,
        ):
            parsed_label = parse_label(rec.get("product_problem_flag"))
            if parsed_label is None or parsed_label != label:
                continue

            raw_text = extract_raw_narrative(rec)
            text = clean_text(raw_text)
            if len(text) < 20:
                continue

            rows.append(
                {
                    "label": parsed_label,
                    "text": text,
                    "text_raw": raw_text,
                    "date_report": parse_date(rec.get("date_report") or rec.get("report_date") or rec.get("date")),
                    "event_type_raw": first_nonempty(rec.get("event_type")),
                    "report_number": first_nonempty(rec.get("report_number"), rec.get("report_id")),
                    "raw_json": json.dumps(rec, ensure_ascii=False),
                }
            )
            count += 1
            if count >= max_per_class:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No records were collected. Try reducing the query strictness or check the API key.")
    df = df.drop_duplicates(subset=["report_number", "text", "label"])
    df = df[df["text"].str.len() > 0].copy()
    df = df.reset_index(drop=True)
    return df


@dataclass
class TextPipeline:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    scaler: StandardScaler

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        X = self.svd.transform(X)
        X = self.scaler.transform(X)
        return X


def make_pipeline(seed: int) -> TextPipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        max_features=50000,
    )
    svd = TruncatedSVD(n_components=300, random_state=seed)
    scaler = StandardScaler()
    return TextPipeline(vectorizer=vectorizer, svd=svd, scaler=scaler)


def fit_text_features(pipe: TextPipeline, train_texts: Sequence[str], val_texts: Sequence[str], test_texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = pipe.vectorizer.fit_transform(train_texts)
    n_components = min(300, max(2, X_train.shape[1] - 1))
    pipe.svd = TruncatedSVD(n_components=n_components, random_state=7)
    X_train = pipe.svd.fit_transform(X_train)
    X_val = pipe.svd.transform(pipe.vectorizer.transform(val_texts))
    X_test = pipe.svd.transform(pipe.vectorizer.transform(test_texts))
    pipe.scaler = StandardScaler()
    X_train = pipe.scaler.fit_transform(X_train)
    X_val = pipe.scaler.transform(X_val)
    X_test = pipe.scaler.transform(X_test)
    return X_train, X_val, X_test


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
            "selective_confusion_matrix": None,
        }

    yt = y_true[confident]
    yp = y_pred[confident]
    precision, recall, _, _ = precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0
    )
    return {
        "coverage": coverage,
        "selective_accuracy": float(accuracy_score(yt, yp)),
        "selective_f1": float(f1_score(yt, yp, zero_division=0)),
        "selective_precision": float(precision),
        "selective_recall": float(recall),
        "selective_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "selective_confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }


def choose_threshold_by_val(y_val: np.ndarray, proba_val: np.ndarray, target_coverage: float) -> float:
    candidates = np.linspace(0.50, 0.99, 50)
    best = candidates[0]
    best_gap = float("inf")
    for t in candidates:
        cov = (np.maximum(proba_val[:, 0], proba_val[:, 1]) >= t).mean()
        gap = abs(cov - target_coverage)
        if gap < best_gap:
            best_gap = gap
            best = float(t)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_maude_product_problem_redacted_textonly")
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENFDA_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-coverage", type=float, default=0.65)
    parser.add_argument("--threshold-min", type=float, default=0.50)
    parser.add_argument("--threshold-max", type=float, default=1.00)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--save-dataset-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running maude_product_problem_redacted_textonly.py with the following parameters:")
    print(f"OUTDIR: {outdir}")
    print(f"OPENFDA_API_KEY: {'<set>' if args.api_key else '<unset>'}")
    print(f"MAX_PER_CLASS: {args.max_per_class}")
    print(f"PAGE_SIZE: {args.page_size}")
    print(f"TARGET_COVERAGE: {args.target_coverage}")
    print(f"SEED: {args.seed}")

    session = make_session()
    df = build_dataset(
        session=session,
        max_per_class=args.max_per_class,
        page_size=args.page_size,
        api_key=args.api_key or None,
    )

    df.to_csv(outdir / "dataset_all.csv", index=False)
    df.to_json(outdir / "dataset_all.jsonl", orient="records", lines=True, force_ascii=False)

    if args.save_dataset_only:
        print(f"Saved dataset only to: {outdir}")
        return

    train_df, temp_df = train_test_split(
        df,
        test_size=0.50,
        random_state=args.seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)

    print("Class counts:", df["label"].value_counts().sort_index().to_dict())

    pipe = make_pipeline(args.seed)
    X_train, X_val, X_test = fit_text_features(pipe, train_df["text"], val_df["text"], test_df["text"])

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

    best_selective = selective_metrics(y_test, proba_test, threshold=chosen_threshold)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    threshold_results: List[Dict[str, Any]] = []
    for threshold in thresholds:
        sel = selective_metrics(y_test, proba_test, threshold=float(threshold))
        threshold_results.append(
            {
                "threshold": float(threshold),
                "coverage": sel["coverage"],
                "accuracy": sel["selective_accuracy"],
                "f1": sel["selective_f1"],
                "precision": sel["selective_precision"],
                "recall": sel["selective_recall"],
            }
        )

    pd.DataFrame(threshold_results).to_csv(outdir / "threshold_sweep.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test,
            "prob_negative": proba_test[:, 0],
            "prob_positive": proba_test[:, 1],
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
            "selective_confusion_matrix": best_selective["selective_confusion_matrix"],
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
        "class_counts": {
            "No product problem": int((df["label"] == 0).sum()),
            "Product problem": int((df["label"] == 1).sum()),
        },
        "split_counts": {
            "train": {
                "No product problem": int((train_df["label"] == 0).sum()),
                "Product problem": int((train_df["label"] == 1).sum()),
            },
            "val": {
                "No product problem": int((val_df["label"] == 0).sum()),
                "Product problem": int((val_df["label"] == 1).sum()),
            },
            "test": {
                "No product problem": int((test_df["label"] == 0).sum()),
                "Product problem": int((test_df["label"] == 1).sum()),
            },
        },
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "task": "MAUDE product problem detection from redacted early narrative text",
        "label_definition": "product_problem_flag mapped to binary class",
        "text_source": "MAUDE narrative text, truncated before boilerplate sections and redacted for explicit outcome words",
        "abstention": "predict only when max class probability exceeds threshold",
        "threshold": chosen_threshold,
        "seed": args.seed,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(
        {
            "vectorizer": pipe.vectorizer,
            "svd": pipe.svd,
            "scaler": pipe.scaler,
            "model": clf,
            "threshold": chosen_threshold,
        },
        outdir / "model.joblib",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {outdir}")
    print(f"Saved threshold sweep to: {outdir / 'threshold_sweep.csv'}")
    print(f"Saved predictions to: {outdir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
