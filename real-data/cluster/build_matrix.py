"""Build the MAUDE abstention-sweep matrix: the FULL payoff sweep for EVERY
hyperparameter config, so the notebook can auto-pick the best config and then show its
o-sweep (nothing changes models midway).

Single source of truth for the sweep (mirrors cluster/build_matrix.py for the
synthetic pipeline). There is ONE real dataset (the scraped MAUDE cache). The matrix
is the CROSS PRODUCT of:

  configs   one-factor-from-BASE hyperparameter variants: class_weight, capacity
            (hidden), dropout, lr -- the settings to auto-pick the best from.
  o         the payoff sweep, o in 1.0..5.0 step 0.1, run for EVERY config.

So every config gets its own full o-sweep; the notebook ranks configs (forced macro-F1
in the plain-classifier regime at the largest o), freezes the winner, and draws every
figure from that one config's o-sweep. Emits into real-data/cluster/:

  runs.tsv       one line per training run:  run_id <TAB> train_from_cache args
  manifest.json  the structured matrix (configs + tags + output paths)

SEEDS ARE A TOP-LEVEL AXIS: each seed-independent configuration is trained once per
seed in SEEDS, so total runs = distinct configs x len(SEEDS). Set RCA_SEEDS
(comma-separated), e.g. RCA_SEEDS=0,1,2,3,4. This is a big matrix (configs x 41 o x
seeds); submit.sh auto-chunks the training array past the Slurm array cap.

Stdlib only. Re-run after editing the config blocks, then re-submit.
"""

import json
import os
from pathlib import Path

# real-data/ is the parent of this cluster/ folder; paths are repo-root-relative.
CLUSTER = Path(__file__).resolve().parent
CACHE = "real-data/data/maude"          # the scraped dataset cache (from prep_dataset.py)
OUT_DIR = "real-data/results/cluster"

# Seeds every configuration is trained at (the top-level replication axis).
SEEDS = [int(s) for s in os.environ.get("RCA_SEEDS", "0,1,2,3,4").split(",") if s.strip() != ""]

# the non-o knobs that identify a hyperparameter CONFIG (what we auto-pick the best of).
CONFIG_KEYS = ["class_weight", "hidden", "dropout", "lr"]

# BASE model config (SEED-INDEPENDENT). Mirrors the sklearn MLP shape; the task is
# the 4-class MAUDE severity label, so `loss` and `class_weight` are axes too.
BASE = {
    "loss": "abstention", "o": 2.0, "class_weight": "none",
    "hidden": "256,128", "dropout": 0.0, "lr": 0.001, "epochs": 80,
}

# study axis grids
O_GRID = [round(1.0 + 0.1 * i, 1) for i in range(41)]       # 1.0, 1.1, ... 5.0
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


def config_variants() -> list[dict]:
    """The non-o hyperparameter configs to auto-pick the best from: BASE plus each knob
    varied ONE AT A TIME (class_weight, capacity, dropout, lr). De-duplicated by the
    (class_weight, hidden, dropout, lr) signature."""
    seen: dict = {}

    def add(**over):
        cfg = {k: over.get(k, BASE[k]) for k in CONFIG_KEYS}
        seen.setdefault(tuple(cfg[k] for k in CONFIG_KEYS), cfg)

    add()                                       # BASE center point
    for cw in CW_GRID:
        add(class_weight=cw)
    for hidden in HIDDEN_GRID:
        add(hidden=hidden)
    for do in DROPOUT_GRID:
        add(dropout=do)
    for lr in LR_GRID:
        add(lr=lr)
    return list(seen.values())


def build_configs() -> list[dict]:
    """Cross every config with the full o-sweep -> one (config, o) entry per run.
    Everything is tagged 'payoff' (the notebook groups by the config knobs itself)."""
    configs: dict = {}
    for variant in config_variants():
        for o in O_GRID:
            _emit(configs, "payoff", o=o, **variant)
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
        "n_config_variants": len(config_variants()),
        "n_configs": len(configs), "runs": run_list,
    }
    (CLUSTER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"dataset : 1 (scraped MAUDE cache at {CACHE})")
    print(f"configs : {len(config_variants())} hyperparameter variants x {len(O_GRID)} o "
          f"= {len(configs)} (config, o) points")
    print(f"seeds   : {SEEDS}  ({len(SEEDS)})")
    print(f"runs    : {len(run_list)}  = {len(configs)} points x {len(SEEDS)} seeds")
    if len(run_list) > 1000:
        print(f"note    : {len(run_list)} runs exceeds the usual Slurm array cap (~1001); "
              f"submit.sh auto-chunks the training array.")
    print("wrote real-data/cluster/runs.tsv, real-data/cluster/manifest.json")


if __name__ == "__main__":
    main()
