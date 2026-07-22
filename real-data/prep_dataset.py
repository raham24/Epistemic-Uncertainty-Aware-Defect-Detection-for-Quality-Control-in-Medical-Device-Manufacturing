"""Scrape MAUDE and build the cached dataset -- the data-prep step of the sweep.

This is the DATA GENERATION half, decoupled from training (mirrors generator.py in
the synthetic pipeline). It runs the professor's redacted TF-IDF data pipeline
VERBATIM -- download openFDA device/event records, redact label-leaking words,
class-balance, stratified train/val/test split, TF-IDF -> TruncatedSVD ->
StandardScaler (all fit on train only) -- then CACHES the result so every training
run in the o-sweep reads the SAME fixed dataset + split + features:

  <cache>/dataset_all.csv      raw balanced dataset (provenance)
  <cache>/dataset_all.jsonl    same, line-delimited
  <cache>/features.npz         X_train/X_val/X_test + y_train/y_val/y_test (float32/int64)
  <cache>/text_pipeline.joblib fitted tfidf + svd + scaler + feature names
  <cache>/dataset_meta.json    counts, split sizes, seed, feature config

Run ONCE (needs network + OPENFDA_API_KEY); the sweep then trains off the cache.

  python real-data/prep_dataset.py --cache real-data/data/maude --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

# import the shared data pipeline (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import maude_product_problem_abstention_mlp as pipe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape MAUDE + cache dataset/features")
    parser.add_argument("--cache", type=str, default="real-data/data/maude",
                        help="output cache directory")
    parser.add_argument("--max-records", "--max-per-class", dest="max_records",
                        type=int, default=40000,
                        help="total records to fetch (natural class proportions)")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENFDA_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=pipe.DEFAULT_SEED)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--svd-components", type=int, default=300)
    args = parser.parse_args()

    pipe.set_seed(args.seed)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print("prep_dataset.py -- scraping MAUDE and caching the dataset")
    print(f"  cache        : {cache}")
    print(f"  api key      : {'<set>' if args.api_key else '<empty> (anonymous)'}")
    print(f"  fetch budget : {args.max_records} records")
    print(f"  seed (split) : {args.seed}")

    # --- MAUDE scrape: 4-class event_type labels, natural proportions ---
    session = pipe.make_session()
    df = pipe.build_dataset(
        session=session,
        api_key=args.api_key or None,
        max_records=args.max_records,
        page_size=args.page_size,
    )

    label_map = pipe.LABEL_NAMES
    counts = df["label"].value_counts().sort_index()
    print(f"  class counts : {{{', '.join(f'{label_map[k]}={v}' for k, v in counts.items())}}}")

    df.to_csv(cache / "dataset_all.csv", index=False)
    df.to_json(cache / "dataset_all.jsonl", orient="records", lines=True, force_ascii=False)

    train_df, val_df, test_df = pipe.make_splits(df, seed=args.seed)
    X_train, X_val, X_test, feature_names, text_pipe = pipe.build_feature_matrix(
        train_df, val_df, test_df,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        svd_components=args.svd_components,
    )

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    # --- cache the feature matrices + targets ---
    np.savez_compressed(
        cache / "features.npz",
        X_train=X_train.astype(np.float32), X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int64), y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
    )
    joblib.dump(
        {"tfidf": text_pipe.tfidf, "svd": text_pipe.svd, "scaler": text_pipe.scaler,
         "feature_names": feature_names},
        cache / "text_pipeline.joblib",
    )

    meta = {
        "seed": args.seed,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(df["label"].nunique()),
        "label_names": label_map,
        "class_counts": {label_map[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "class_proportions": {
            split: {label_map[k]: float(v) for k, v in
                    frame["label"].value_counts(normalize=True).sort_index().items()}
            for split, frame in (("overall", df), ("train", train_df),
                                 ("val", val_df), ("test", test_df))
        },
        "feature_config": {
            "tfidf_max_features": args.tfidf_max_features,
            "tfidf_ngram": [args.tfidf_ngram_min, args.tfidf_ngram_max],
            "svd_components": args.svd_components,
        },
        "redaction": {"leaky_patterns": pipe.LEAKY_PATTERNS,
                      "boilerplate_patterns": pipe.BOILERPLATE_PATTERNS},
    }
    (cache / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print(f"  wrote features.npz ({X_train.shape[0]}+{X_val.shape[0]}+{X_test.shape[0]} "
          f"x {X_train.shape[1]}) + text_pipeline.joblib to {cache}")


if __name__ == "__main__":
    main()
