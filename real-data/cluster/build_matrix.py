"""Build the MAUDE abstention-sweep matrix: one scraped dataset x a structured model
sweep, replicated across multiple random SEEDS.

Single source of truth for the sweep (mirrors cluster/build_matrix.py for the
synthetic pipeline). Unlike the synthetic sweep there is ONE real dataset (the
scraped MAUDE cache), so the axis that varies is the model -- above all the
abstention payoff `o`. Emits two files into real-data/cluster/:

  runs.tsv       one line per training run:  run_id <TAB> train_from_cache args
  manifest.json  the full structured matrix (configs + study tags + output paths)

The sweep is a BASE config plus one-factor-at-a-time STUDIES; shared center points
are de-duplicated (defined once, tagged with every study that wants them):

  payoff     o in 1.0..4.0 step 0.1        -- coverage vs selective accuracy vs o (headline)
  capacity   trunk width/depth             -- does a bigger net change the reject signal
  dropout    dropout in {0,.1,.2,.3}       -- regularization
  lr         Adam lr in {3e-4,1e-3,3e-3}   -- optimization

SEEDS ARE A TOP-LEVEL AXIS: each seed-independent configuration is trained once per
seed in SEEDS, so total runs = distinct configs x len(SEEDS). Set RCA_SEEDS
(comma-separated), e.g. RCA_SEEDS=0,1,2,3,4.

Stdlib only. Re-run after editing the config blocks, then re-submit.
"""

import json
import os
from collections import Counter
from pathlib import Path

# real-data/ is the parent of this cluster/ folder; paths are repo-root-relative.
CLUSTER = Path(__file__).resolve().parent
CACHE = "real-data/data/maude"          # the scraped dataset cache (from prep_dataset.py)
OUT_DIR = "real-data/results/cluster"

# Seeds every configuration is trained at (the top-level replication axis).
SEEDS = [int(s) for s in os.environ.get("RCA_SEEDS", "0,1,2,3,4").split(",") if s.strip() != ""]

# BASE model config (SEED-INDEPENDENT). Mirrors the sklearn MLP shape; the task is
# the 4-class MAUDE severity label, so `loss` and `class_weight` are axes too.
BASE = {
    "loss": "abstention", "o": 2.0, "class_weight": "none",
    "hidden": "256,128", "dropout": 0.0, "lr": 0.001, "epochs": 80,
}

# study axis grids
O_GRID = [round(1.0 + 0.1 * i, 1) for i in range(31)]       # 1.0, 1.1, ... 4.0
HIDDEN_GRID = ["128,64", "256,128", "512,256", "256,128,64"]
DROPOUT_GRID = [0.0, 0.1, 0.2, 0.3]
LR_GRID = [0.0003, 0.001, 0.003]
CW_GRID = ["none", "sqrt", "inverse"]                       # class weighting (skew)

# axes that uniquely identify a CONFIGURATION (the de-dup signature; seed excluded).
AXES = ["loss", "o", "class_weight", "hidden", "dropout", "lr", "epochs"]


def _emit(configs: dict, study: str, **over) -> None:
    """Register one configuration (BASE + overrides) under `study`, de-duplicated by
    the seed-independent axis signature."""
    cfg = {**BASE, **over}
    sig = tuple(cfg[a] for a in AXES)
    if sig in configs:
        configs[sig]["studies"].append(study)
        return
    cfg["studies"] = [study]
    configs[sig] = cfg


def build_configs() -> list[dict]:
    configs: dict = {}
    # core -- the plain cross-entropy baseline (class-weighted) vs base abstention
    _emit(configs, "core", loss="ce", class_weight="sqrt")
    # payoff -- the headline abstention sweep over o
    for o in O_GRID:
        _emit(configs, "payoff", o=o)
    # class_weight -- does weighting the skewed classes change the picture
    for cw in CW_GRID:
        _emit(configs, "class_weight", class_weight=cw)
    # capacity -- trunk width/depth at the base payoff
    for hidden in HIDDEN_GRID:
        _emit(configs, "capacity", hidden=hidden)
    # dropout -- regularization strength
    for do in DROPOUT_GRID:
        _emit(configs, "dropout", dropout=do)
    # lr -- Adam learning rate
    for lr in LR_GRID:
        _emit(configs, "lr", lr=lr)
    return list(configs.values())


def _slug(c: dict, seed: int) -> str:
    """Readable run label: only the axes that differ from BASE, then the seed."""
    parts = ["maude", c["loss"]]
    if c["loss"] == "abstention":
        parts.append(f"o{c['o']:g}")
    if c["class_weight"] != BASE["class_weight"]:
        parts.append("cw-" + c["class_weight"])
    if c["hidden"] != BASE["hidden"]:
        parts.append("h" + c["hidden"].replace(",", "x"))
    if c["dropout"] != BASE["dropout"]:
        parts.append(f"do{c['dropout']:g}")
    if c["lr"] != BASE["lr"]:
        parts.append(f"lr{c['lr']:g}")
    return "_".join(parts) + f"_s{seed}"


def _args(c: dict, seed: int, run_id: str) -> str:
    """The exact train_from_cache.py CLI (every flag explicit -> reproducible)."""
    return (f"--cache {CACHE} --loss {c['loss']} --o {c['o']:g} "
            f"--class-weight {c['class_weight']} --seed {seed} --epochs {c['epochs']} "
            f"--hidden {c['hidden']} --dropout {c['dropout']:g} --lr {c['lr']:g} "
            f"--out {OUT_DIR}/{run_id}.json")


def main() -> None:
    if not SEEDS:
        raise SystemExit("RCA_SEEDS is empty -- set at least one seed, e.g. RCA_SEEDS=0,1,2")

    configs = build_configs()

    run_list = []
    i = 0
    for c in configs:
        for seed in SEEDS:
            i += 1
            run_id = f"{i:04d}_{_slug(c, seed)}"
            run_list.append({
                "run_id": run_id,
                "loss": c["loss"], "o": c["o"], "class_weight": c["class_weight"],
                "seed": seed, "hidden": c["hidden"],
                "dropout": c["dropout"], "lr": c["lr"], "epochs": c["epochs"],
                "studies": sorted(set(c["studies"])),
                "args": _args(c, seed, run_id),
                "metrics": f"{OUT_DIR}/{run_id}.json",
            })

    # runs.tsv : run_id <TAB> train args  (read by line in train.slurm)
    run_lines = [f"{r['run_id']}\t{r['args']}" for r in run_list]
    (CLUSTER / "runs.tsv").write_text("\n".join(run_lines) + "\n")

    manifest = {
        "cache": CACHE, "out_dir": OUT_DIR, "base": BASE, "seeds": SEEDS,
        "n_configs": len(configs), "runs": run_list,
    }
    (CLUSTER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    by_study = Counter(s for r in run_list for s in r["studies"])
    print(f"dataset : 1 (scraped MAUDE cache at {CACHE})")
    print(f"configs : {len(configs)}  (seed-independent)")
    print(f"seeds   : {SEEDS}  ({len(SEEDS)})")
    print(f"runs    : {len(run_list)}  = {len(configs)} configs x {len(SEEDS)} seeds")
    for s, n in sorted(by_study.items()):
        print(f"  study {s:10s}: {n} runs")
    if len(run_list) > 1000:
        print(f"WARNING: {len(run_list)} runs exceeds the usual Slurm MaxArraySize (1001).")
    print("wrote real-data/cluster/runs.tsv, real-data/cluster/manifest.json")


if __name__ == "__main__":
    main()
