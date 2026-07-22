"""MAUDE product-problem detection with a LEARNED-abstention MLP (PyTorch).

Port of maude_product_problem_redacted_tfidf_mlp.py (Prof. Ameri's script). The
data generation is reused VERBATIM -- download MAUDE device/event records from the
openFDA API, redact label-leaking words, TF-IDF -> TruncatedSVD -> StandardScaler,
class-balanced, stratified train/val/test split. Everything through
build_feature_matrix() is unchanged.

What differs from the original:
  * The sklearn MLPClassifier is replaced by a single-head PyTorch MLP with the
    SAME shape (hidden (256, 128), ReLU, Adam, lr 1e-3, L2/weight-decay 1e-4).
    "Same number of heads" as the original: ONE classification head.
  * The head is widened by one "abstain" column (output width 2 real classes + 1),
    and the model is trained with the selective-classification ("abstention") term
    -log(o * p_y + r) from mlp.py -- a LEARNED reject option, not a post-hoc
    confidence threshold. --o is the payoff (r=0 reduces the term to cross-entropy).

Evaluation reports BOTH selective-classification curves on the same trained model,
so the learned reject is measured head-to-head against the original's method:
  * learned reject : accept a record when its abstain prob r < h.
  * confidence baseline : accept when max class prob >= threshold (the ORIGINAL
    script's rule, applied to this model's renormalized 2-class probabilities).
gain_vs_conf = selective_acc(learned reject) - selective_acc(confidence) at a
matched target coverage -- the "does the learned reject add anything over
confidence thresholding" question. (On the SMT synthetic data this gap is <= 0;
this script is how we check it on the real MAUDE task.)

Run:
  python maude_product_problem_abstention_mlp.py --outdir out --o 2.0
  python maude_product_problem_abstention_mlp.py --save-dataset-only   # cache data
"""

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from requests.adapters import HTTPAdapter
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
from sklearn.preprocessing import StandardScaler
from urllib3.util.retry import Retry

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

BASE_URL = "https://api.fda.gov"
DEFAULT_SEED = 7

# ---- model defaults (match the original sklearn MLPClassifier as closely as possible)
HIDDEN = (256, 128)          # hidden_layer_sizes
LR = 1e-3                    # learning_rate_init
WEIGHT_DECAY = 1e-4          # alpha (sklearn L2 penalty)
BATCH = 32                   # batch_size
EPOCHS = 80                  # max_iter
PATIENCE = 10               # n_iter_no_change
DROPOUT = 0.0               # sklearn MLP has no dropout; use L2 (weight decay) only
DEFAULT_O = 2.0             # abstention payoff

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

# --------------------------------------------------------------------------- #
# Label scheme (4-class, from MAUDE event_type + a narrative severity split).
#
# MAUDE `event_type` is one of {Death, Injury, Malfunction, Other, No answer}.
# Deaths / Injuries / Malfunctions map to native classes; Other / No-answer /
# blank are DROPPED. MAUDE has no native basic-vs-serious injury field (an
# `Injury` MDR is already a regulatory "serious injury" under 21 CFR 803), so the
# basic/serious split is a PROXY: an Injury whose RAW narrative (before redaction)
# hits a serious-injury cue (FDA definition -- life-threatening, permanent
# impairment, or medical/surgical intervention) is labeled serious, else basic.
# Integers are severity-ordered.
# --------------------------------------------------------------------------- #
LABEL_NAMES = {0: "Malfunction", 1: "Basic injury", 2: "Serious injury", 3: "Death"}

SEVERE_INJURY_PATTERNS = [
    r"\blife[- ]threatening\b",
    r"\bpermanent(ly)?\b",
    r"\bdisab(led|ility|ling)\b",
    r"\bsurg(ery|ical|eon|eries)\b",
    r"\boperat(ion|ive|ing room)\b",
    r"\breoperation\b",
    r"\bhospitaliz(ed|ation)\b",
    r"\badmitted (to (the )?(hospital|icu|emergency))\b",
    r"\bintensive care\b",
    r"\bicu\b",
    r"\bintubat(ed|ion)\b",
    r"\bventilat(or|ed|ion)\b",
    r"\bresuscitat(ed|ion)\b",
    r"\bcardiac arrest\b",
    r"\bamputat(ed|ion)\b",
    r"\bparaly(sis|zed|sed)\b",
    r"\bcoma(tose)?\b",
    r"\bunconscious\b",
    r"\bhemorrhage\b",
    r"\bhaemorrhage\b",
    r"\btransfusion\b",
    r"\bsepsis\b",
    r"\bseptic\b",
    r"\bemergency (surgery|room|department|intervention)\b",
    r"\bintervention (was )?(required|needed|necessary)\b",
]


def is_serious_injury(raw_text: str) -> bool:
    """True if the RAW (un-redacted) narrative hits a serious-injury cue."""
    low = (raw_text or "").lower()
    return any(re.search(p, low) for p in SEVERE_INJURY_PATTERNS)


def event_label(event_type_raw: str, raw_text: str) -> Optional[int]:
    """Map MAUDE event_type (+ narrative severity for injuries) to a class int.
    Returns None for Other / No-answer / blank (those records are dropped)."""
    et = (event_type_raw or "").strip().lower()
    if "malfunction" in et:
        return 0
    if "death" in et:
        return 3
    if "injur" in et:                       # matches "Injury"/"injuries"
        return 2 if is_serious_injury(raw_text) else 1
    return None                             # Other / No answer provided / blank


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data generation -- lifted VERBATIM from the original redacted_tfidf script
# --------------------------------------------------------------------------- #

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
            "User-Agent": "maude-product-problem-abstention/1.0",
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
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace(" ", " ")
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
        "product_problem_flag": parse_flag(record.get("product_problem_flag")),  # kept for provenance
        "event_type_raw": event_type,
        "text_raw": text,
        "text": clean_text(text),
        # 4-class label from event_type + narrative severity (severity read on RAW text)
        "label": event_label(event_type, text),
        "raw_json": json.dumps(record, ensure_ascii=False),
    }
    return row


def build_dataset(
    session: requests.Session,
    api_key: Optional[str],
    max_records: int,
    page_size: int,
) -> pd.DataFrame:
    """Scrape a NATURAL-PROPORTION 4-class dataset from MAUDE.

    Queries only the three native event types (Death/Injury/Malfunction), labels
    each via event_label (injuries split basic/serious on narrative severity), drops
    the unmappable ones, and KEEPS the real class skew -- no per-class balancing.
    Malfunction dominates and Death is rare; that imbalance is intentional (the MLP
    will class-weight later). `max_records` is the total fetch budget.
    """
    records = list(
        iter_openfda_records(
            session,
            "device/event",
            query="event_type:(Death OR Injury OR Malfunction)",
            max_records=max_records,
            page_size=page_size,
            api_key=api_key,
        )
    )
    if not records:
        raise RuntimeError("No MAUDE records returned by openFDA.")

    rows: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc="Processing MAUDE records", unit="rec"):
        row = parse_record(rec)
        if row["label"] is None:            # Other / No-answer / unmappable event_type
            continue
        if not row["text"]:
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable MAUDE records after parsing.")

    # Natural MAUDE proportions: no per-class balancing, just a deterministic shuffle.
    df = df.sample(frac=1.0, random_state=DEFAULT_SEED).reset_index(drop=True)
    return df


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


# --------------------------------------------------------------------------- #
# Learned-abstention MLP (replaces the sklearn MLPClassifier)
# --------------------------------------------------------------------------- #

def abstention_term(logits: torch.Tensor, y: torch.Tensor, o: float) -> torch.Tensor:
    """Selective-classification term on one softmax head (lifted from mlp.py).

    Input: (B, k+1) logits whose LAST column is the abstain output, the true class
           index y in [0, k-1], and the payoff o (> 0).
    Return: mean of -log(o*p_y + r), where p_y is the true-class prob and r the
            abstain prob. r=0 reduces this to cross-entropy + const.

    Computed in log-space so o*p_y + r stays stable when both are tiny:
        log(o*p_y + r) = logsumexp([log p_y + log o, log r])   -- no eps, no spike
    """
    log_probs = torch.log_softmax(logits, dim=1)             # (B, k+1)
    log_py = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)  # log p_y (true class)
    log_r = log_probs[:, -1]                                 # log r (abstain col)
    log_o = torch.log(torch.tensor(o, device=log_probs.device, dtype=log_probs.dtype))
    log_z = torch.logsumexp(torch.stack([log_py + log_o, log_r], 0), 0)
    return -log_z.mean()


class AbstentionMLP(nn.Module):
    """Single-head feedforward MLP, same shape as the original sklearn MLPClassifier
    (hidden (256, 128), ReLU), but the output head is widened by one "abstain" column:
    width = n_classes + 1. The last column is the learned reject prob r.

    "Same number of heads" as the original: ONE classification head. The abstain
    column is an extra output ON that head, not a second head.
    """

    def __init__(self, n_features: int, n_classes: int = 2,
                 hidden: Tuple[int, ...] = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.n_classes = n_classes
        layers: List[nn.Module] = []
        d = n_features
        for w in hidden:
            layers += [nn.Linear(d, w), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = w
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes + 1)   # +1 abstain column

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))           # (B, n_classes + 1) logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return renormalized real-class probabilities (2-col, sum to 1) plus the
        abstain prob. The 2-col probs are the drop-in analogue of sklearn's
        predict_proba, so the ORIGINAL confidence-threshold rule applies unchanged."""
        self.eval()
        probs = F.softmax(self.forward(x), dim=1)          # (B, n_classes + 1)
        real = probs[:, : self.n_classes]
        real = real / real.sum(dim=1, keepdim=True).clamp_min(1e-12)   # renormalize
        return {"proba": real, "abstain_prob": probs[:, self.n_classes]}


def train_abstention_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    o: float,
    seed: int,
    device: str,
    hidden: Tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch: int,
    epochs: int,
    patience: int,
) -> AbstentionMLP:
    """Train the abstention MLP with the -log(o*p_y + r) term, early-stopping on
    validation loss (mirrors the original's early_stopping / n_iter_no_change)."""
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    model = AbstentionMLP(n_features=X_train.shape[1], n_classes=2,
                          hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.int64))
    xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_val.astype(np.int64)).to(device)

    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True, generator=g)

    best_val, best_state, waited = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = abstention_term(model(xb), yb, o)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(abstention_term(model(xv), yv, o))
        print(f"  epoch {ep:2d}  val_loss {val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, waited = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def model_outputs(model: AbstentionMLP, X: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (proba [N,2] renormalized real-class, abstain_prob [N])."""
    out = model.predict(torch.from_numpy(X.astype(np.float32)).to(device))
    return out["proba"].cpu().numpy(), out["abstain_prob"].cpu().numpy()


# --------------------------------------------------------------------------- #
# Selective-classification metrics
# --------------------------------------------------------------------------- #

def selective_metrics(y_true: np.ndarray, proba: np.ndarray, keep: np.ndarray) -> Dict[str, Any]:
    """Metrics on the ACCEPTED (kept) subset. keep is a boolean mask. This is the
    shared core for both the confidence baseline and the learned-reject curve."""
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    coverage = float(keep.mean())
    if keep.sum() == 0:
        return {
            "coverage": 0.0,
            "selective_accuracy": None,
            "selective_f1": None,
            "selective_precision": None,
            "selective_recall": None,
            "selective_balanced_accuracy": None,
        }
    yt, yp = y_true[keep], y_pred[keep]
    precision, recall, _, _ = precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0,
    )
    return {
        "coverage": coverage,
        "selective_accuracy": float(accuracy_score(yt, yp)),
        "selective_f1": float(f1_score(yt, yp, zero_division=0)),
        "selective_precision": float(precision),
        "selective_recall": float(recall),
        "selective_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
    }


def confidence_keep(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Original rule: accept when max class prob >= threshold."""
    return np.max(proba, axis=1) >= threshold


def reject_keep(abstain_prob: np.ndarray, h: float) -> np.ndarray:
    """Learned rule: accept when the abstain prob r < h."""
    return abstain_prob < h


def choose_threshold_by_val(proba_val: np.ndarray, target_coverage: float) -> float:
    """Largest confidence threshold whose val coverage still >= target (most
    selective while meeting coverage). Mirrors the original script."""
    candidates = np.arange(0.50, 0.991, 0.01)
    best = float(candidates[-1])
    for t in candidates:
        if confidence_keep(proba_val, float(t)).mean() >= target_coverage:
            best = float(t)
    return best


def choose_reject_by_val(abstain_val: np.ndarray, target_coverage: float) -> float:
    """Smallest abstain threshold h whose val coverage still >= target (most
    selective while meeting coverage) -- the learned-reject analogue."""
    candidates = np.arange(0.99, 0.009, -0.01)   # descending: coverage falls with h
    best = float(candidates[-1])
    for h in candidates:
        if reject_keep(abstain_val, float(h)).mean() >= target_coverage:
            best = float(h)
    return best


def confidence_sweep(y_true: np.ndarray, proba: np.ndarray,
                     lo: float, hi: float, step: float) -> List[Dict[str, Any]]:
    rows = []
    for t in np.arange(lo, hi + 1e-12, step):
        m = selective_metrics(y_true, proba, confidence_keep(proba, float(t)))
        rows.append({"threshold": float(t), **{k: m[k] for k in (
            "coverage", "selective_accuracy", "selective_f1",
            "selective_precision", "selective_recall", "selective_balanced_accuracy")}})
    return rows


def reject_sweep(y_true: np.ndarray, proba: np.ndarray, abstain_prob: np.ndarray,
                 lo: float, hi: float, step: float) -> List[Dict[str, Any]]:
    rows = []
    for h in np.arange(lo, hi + 1e-12, step):
        m = selective_metrics(y_true, proba, reject_keep(abstain_prob, float(h)))
        rows.append({"h": float(h), **{k: m[k] for k in (
            "coverage", "selective_accuracy", "selective_f1",
            "selective_precision", "selective_recall", "selective_balanced_accuracy")}})
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_maude_product_problem_abstention")
    parser.add_argument("--max-records", "--max-per-class", dest="max_records",
                        type=int, default=40000,
                        help="total records to fetch (natural class proportions)")
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
    # --- abstention / model knobs ---
    parser.add_argument("--o", type=float, default=DEFAULT_O, help="abstention payoff")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--hidden", type=str, default=",".join(map(str, HIDDEN)))
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--save-dataset-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running maude_product_problem_abstention_mlp.py with the following parameters:")
    print(f"OUTDIR: {outdir}")
    print(f"OPENFDA_API_KEY: {'<set>' if args.api_key else '<empty>'}")
    print(f"MAX_RECORDS: {args.max_records}")
    print(f"PAGE_SIZE: {args.page_size}")
    print(f"TARGET_COVERAGE: {args.target_coverage}")
    print(f"ABSTENTION PAYOFF o: {args.o}")
    print(f"DEVICE: {args.device}")
    print(f"Seed: {args.seed}")

    session = make_session()
    df = build_dataset(
        session=session,
        api_key=args.api_key or None,
        max_records=args.max_records,
        page_size=args.page_size,
    )

    df.to_csv(outdir / "dataset_all.csv", index=False)
    df.to_json(outdir / "dataset_all.jsonl", orient="records", lines=True, force_ascii=False)

    label_map = LABEL_NAMES
    print(f"Class counts: {{{', '.join(f'{label_map[k]}={v}' for k, v in df['label'].value_counts().sort_index().items())}}}")

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

    # The single-head model + binary metrics below are still 2-class. The scraper now
    # emits a 4-class label (Malfunction/Basic injury/Serious injury/Death), so the
    # in-script training path is disabled until the MLP is updated for >2 classes.
    # Use --save-dataset-only (or the cluster prep) to build the dataset meanwhile.
    n_classes = int(df["label"].nunique())
    if n_classes > 2:
        raise SystemExit(
            f"dataset has {n_classes} classes ({sorted(df['label'].unique())}); the "
            f"in-script abstention MLP is still binary. Re-run with --save-dataset-only "
            f"to just build the dataset, or update the MLP for multi-class first.")

    X_train, X_val, X_test, feature_names, text_pipe = build_feature_matrix(
        train_df, val_df, test_df,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        svd_components=args.svd_components,
    )

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    hidden = tuple(int(w) for w in args.hidden.split(","))
    model = train_abstention_mlp(
        X_train, y_train, X_val, y_val,
        o=args.o, seed=args.seed, device=args.device,
        hidden=hidden, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, batch=args.batch,
        epochs=args.epochs, patience=args.patience,
    )

    proba_val, abstain_val = model_outputs(model, X_val, args.device)
    proba_test, abstain_test = model_outputs(model, X_test, args.device)

    # ---- forced (full-coverage) metrics: argmax over the 2 real classes on ALL rows
    yhat_test = (proba_test[:, 1] >= 0.5).astype(int)
    std_acc = float(accuracy_score(y_test, yhat_test))
    std_f1 = float(f1_score(y_test, yhat_test, zero_division=0))
    std_bal_acc = float(balanced_accuracy_score(y_test, yhat_test))
    try:
        std_auc = float(roc_auc_score(y_test, proba_test[:, 1]))
    except Exception:
        std_auc = None

    # ---- learned reject: pick h on val to hit target coverage
    chosen_h = choose_reject_by_val(abstain_val, target_coverage=args.target_coverage)
    learned = selective_metrics(y_test, proba_test, reject_keep(abstain_test, chosen_h))

    # ---- confidence baseline: the ORIGINAL rule, applied to THIS model's probs
    chosen_threshold = choose_threshold_by_val(proba_val, target_coverage=args.target_coverage)
    baseline = selective_metrics(y_test, proba_test, confidence_keep(proba_test, chosen_threshold))

    # ---- head-to-head: does the learned reject beat confidence thresholding?
    gain_vs_conf = None
    if learned["selective_accuracy"] is not None and baseline["selective_accuracy"] is not None:
        gain_vs_conf = learned["selective_accuracy"] - baseline["selective_accuracy"]

    # ---- full sweeps for both curves
    pd.DataFrame(reject_sweep(y_test, proba_test, abstain_test,
                              0.01, 0.99, args.threshold_step)).to_csv(
        outdir / "reject_sweep.csv", index=False)
    pd.DataFrame(confidence_sweep(y_test, proba_test,
                                  args.threshold_min, args.threshold_max, args.threshold_step)).to_csv(
        outdir / "threshold_sweep.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "y_true": y_test,
            "prob_no_problem": proba_test[:, 0],
            "prob_problem": proba_test[:, 1],
            "abstain_prob": abstain_test,
            "predicted_label": yhat_test,
            "accepted_learned": reject_keep(abstain_test, chosen_h),
            "accepted_confidence": confidence_keep(proba_test, chosen_threshold),
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
        "learned_reject": {
            "abstain_threshold_h": chosen_h,
            "mean_abstain_prob": float(abstain_test.mean()),
            **learned,
        },
        "confidence_baseline": {
            "threshold": chosen_threshold,
            **baseline,
        },
        "gain_vs_conf": gain_vs_conf,
        "validation_coverage_target": args.target_coverage,
        "abstention_payoff_o": args.o,
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
        "task": "MAUDE event severity classification from narrative",
        "label_definition": "event_type -> {0 Malfunction, 1 Basic injury, 2 Serious "
                            "injury, 3 Death}; injuries split by narrative severity",
        "text_source": "MAUDE narrative text redacted to remove obvious label cues",
        "model": "TF-IDF + SVD + PyTorch MLP with LEARNED abstention (-log(o*p_y + r))",
        "heads": "single classification head (2 classes + 1 abstain column)",
        "abstention": "learned reject column trained end-to-end; accept when r < h",
        "comparison": "learned reject vs original max-prob confidence threshold at matched coverage",
        "features": feature_names,
        "redaction": {
            "leaky_patterns": LEAKY_PATTERNS,
            "boilerplate_patterns": BOILERPLATE_PATTERNS,
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_features": X_train.shape[1],
                "n_classes": 2,
                "hidden": hidden,
                "dropout": args.dropout,
                "o": args.o,
            },
        },
        outdir / "model.pt",
    )
    joblib.dump(
        {
            "tfidf": text_pipe.tfidf,
            "svd": text_pipe.svd,
            "scaler": text_pipe.scaler,
            "abstain_threshold_h": chosen_h,
            "confidence_threshold": chosen_threshold,
            "feature_names": feature_names,
        },
        outdir / "text_pipeline.joblib",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {outdir}")
    print(f"Saved reject sweep to: {outdir / 'reject_sweep.csv'}")
    print(f"Saved threshold sweep to: {outdir / 'threshold_sweep.csv'}")
    print(f"Saved predictions to: {outdir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
