"""Build the cluster experiment matrix: dataset variants x a structured model sweep.

Single source of truth for the sweep. Emits three files into cluster/:

  datasets.tsv   one line per dataset variant:  name <TAB> generator args
  runs.tsv       one line per training run:      run_id <TAB> dataset <TAB> mlp args
  manifest.json  the full structured matrix (configs + study tags + output paths)

The sweep is a primary grid plus several one-factor-at-a-time STUDIES around a
single BASE config, so each hyperparameter is isolated. Shared center points are
de-duplicated (trained once, tagged with every study that wants them):

  core         every dataset x {cascade, abstention(o=2)} x CORE_SEEDS  -- headline
  payoff       abstention o in 1.0..4.0 step 0.1 on probe datasets      -- coverage vs o
  capacity     trunk width/depth on probe datasets                      -- model size
  dropout      dropout in {0,.1,.2,.3} on probe datasets                -- regularization
  lr           Adam lr in {3e-4,1e-3,3e-3} on probe datasets            -- optimization
  class_weight {none,sqrt,inverse} on imbalanced/baseline               -- minority recall

The SLURM arrays (gen_data.slurm, train.slurm) read the .tsv files by line number
($SLURM_ARRAY_TASK_ID); the analysis notebook reads manifest.json. Edit the config
blocks below, then re-run `python cluster/build_matrix.py` and re-submit.

Stdlib only -- runs anywhere, no heavy deps.
"""

import json
from collections import Counter
from pathlib import Path

# repo root = parent of this file's folder (cluster/). Paths in the manifest are
# repo-root-relative; every script cd's to the root before using them.
ROOT = Path(__file__).resolve().parent.parent
CLUSTER = ROOT / "cluster"
DATA_DIR = "data/cluster"
OUT_DIR = "results/cluster"

# --------------------------------------------------------------------------- #
# Dataset variants: name -> generator CLI args (single tokens, NO spaces inside a
# flag value). Each is generated once. Spans class balance, difficulty, process
# spread, and one re-seeded data draw (data-level variance).
# --------------------------------------------------------------------------- #
DATASETS = {
    "baseline":      "",                                                    # paper balance, bayes 0.045
    "balanced":      "--priors no_defect=0.5,open_circuit=0.25,solder_bridging=0.25",
    "imbalanced":    "--priors no_defect=0.94,open_circuit=0.03,solder_bridging=0.03",
    "mid":           "--bayes-error 0.07",                                  # mildly harder
    "hard":          "--bayes-error 0.09",                                  # harder labels
    "harder":        "--bayes-error 0.12",                                  # hardest labels
    "wide":          "--sigma-scale 1.5",                                   # wider process spread
    "hard_wide":     "--bayes-error 0.09 --sigma-scale 1.5",                # harder + wider
    "balanced_hard": "--priors no_defect=0.5,open_circuit=0.25,solder_bridging=0.25 --bayes-error 0.09",
    "baseline_s7":   "--seed 7",                                            # same config, different data draw
}

# --------------------------------------------------------------------------- #
# BASE model config -- every study varies exactly ONE axis around this. The
# values mirror mlp.py's own defaults, so a base run == `python mlp.py` on that
# dataset (keeps the sweep anchored to the documented baseline).
# --------------------------------------------------------------------------- #
BASE = {
    "loss": "cascade", "o": None, "seed": 0,
    "hidden": "256,256", "dropout": 0.1, "class_weight": "none",
    "lr": 0.001, "epochs": 40,
}

CORE_SEEDS = [0, 1, 2]      # tighter variance band on the headline grid
STUDY_SEEDS = [0, 1]        # focused studies: 2 seeds is enough to read the trend

# probe datasets each study sweeps over (every name must be a key in DATASETS)
PROBE_PAYOFF = ["baseline", "hard", "imbalanced"]
PROBE_TUNING = ["baseline", "hard"]            # capacity / dropout / lr
PROBE_CW = ["imbalanced", "baseline"]          # class weighting matters most when skewed

# study axis grids
O_GRID = [round(1.0 + 0.1 * i, 1) for i in range(31)]   # 1.0, 1.1, ... 4.0 (step 0.1)
HIDDEN_GRID = ["128,128", "256,256", "512,512", "256,256,256", "512,512,256"]
DROPOUT_GRID = [0.0, 0.1, 0.2, 0.3]
LR_GRID = [0.0003, 0.001, 0.003]
CW_GRID = ["none", "sqrt", "inverse"]

# the config axes that uniquely identify a run (the de-dup signature)
AXES = ["dataset", "loss", "o", "seed", "hidden", "dropout", "class_weight",
        "lr", "epochs"]


def _emit(runs: dict, study: str, dataset: str, **over) -> None:
    """Register one run (config = BASE + dataset + overrides) under `study`.

    De-duplicated by the full axis signature: if an identical config was already
    emitted (a shared center point), just tag it with this study too.
    """
    cfg = {**BASE, "dataset": dataset, **over}
    sig = tuple(cfg[a] for a in AXES)
    if sig in runs:
        runs[sig]["studies"].append(study)
        return
    cfg["studies"] = [study]
    runs[sig] = cfg


def build_runs() -> list[dict]:
    """The full de-duplicated run list (insertion-ordered: core first)."""
    runs: dict = {}

    # core -- the headline grid: every dataset x {cascade, abstention@o=2} x seeds
    for ds in DATASETS:
        for seed in CORE_SEEDS:
            _emit(runs, "core", ds, loss="cascade", seed=seed)
            _emit(runs, "core", ds, loss="abstention", o=2.0, seed=seed)

    # payoff -- abstention coverage/accuracy vs the payoff o
    for ds in PROBE_PAYOFF:
        for o in O_GRID:
            for seed in STUDY_SEEDS:
                _emit(runs, "payoff", ds, loss="abstention", o=o, seed=seed)

    # capacity -- trunk width/depth (cascade)
    for ds in PROBE_TUNING:
        for hidden in HIDDEN_GRID:
            for seed in STUDY_SEEDS:
                _emit(runs, "capacity", ds, hidden=hidden, seed=seed)

    # dropout -- regularization strength (cascade)
    for ds in PROBE_TUNING:
        for do in DROPOUT_GRID:
            for seed in STUDY_SEEDS:
                _emit(runs, "dropout", ds, dropout=do, seed=seed)

    # lr -- Adam learning rate (cascade)
    for ds in PROBE_TUNING:
        for lr in LR_GRID:
            for seed in STUDY_SEEDS:
                _emit(runs, "lr", ds, lr=lr, seed=seed)

    # class_weight -- defect-head reweighting for minority recall (cascade)
    for ds in PROBE_CW:
        for cw in CW_GRID:
            for seed in STUDY_SEEDS:
                _emit(runs, "class_weight", ds, class_weight=cw, seed=seed)

    return list(runs.values())


def _slug(c: dict) -> str:
    """Readable run label: dataset_loss + only the axes that differ from BASE,
    then the seed. Base runs stay short (`baseline_cascade_s0`); varied runs are
    self-describing (`hard_cascade_h512x512_s1`)."""
    parts = [c["dataset"], c["loss"]]
    if c["o"] is not None:
        parts.append(f"o{c['o']:g}")
    if c["hidden"] != BASE["hidden"]:
        parts.append("h" + c["hidden"].replace(",", "x"))
    if c["dropout"] != BASE["dropout"]:
        parts.append(f"do{c['dropout']:g}")
    if c["lr"] != BASE["lr"]:
        parts.append(f"lr{c['lr']:g}")
    if c["class_weight"] != BASE["class_weight"]:
        parts.append("cw-" + c["class_weight"])
    return "_".join(parts) + f"_s{c['seed']}"


def _args(c: dict) -> str:
    """The exact mlp.py CLI for this run -- every flag explicit, so a run is fully
    reproducible from runs.tsv regardless of mlp.py's current defaults."""
    a = (f"--loss {c['loss']} --seed {c['seed']} --epochs {c['epochs']} "
         f"--hidden {c['hidden']} --dropout {c['dropout']:g} "
         f"--class-weight {c['class_weight']} --lr {c['lr']:g}")
    if c["o"] is not None:
        a += f" --o {c['o']:g}"
    return a


def main() -> None:
    CLUSTER.mkdir(exist_ok=True)

    runs_cfg = build_runs()
    bad = sorted({c["dataset"] for c in runs_cfg} - set(DATASETS))
    if bad:                                  # a study probe references a missing dataset
        raise SystemExit(f"studies reference unknown dataset(s): {bad}")

    run_list = []
    for i, c in enumerate(runs_cfg, start=1):
        run_id = f"{i:03d}_{_slug(c)}"
        run_list.append({
            "run_id": run_id, "dataset": c["dataset"],
            "loss": c["loss"], "o": c["o"], "seed": c["seed"],
            "hidden": c["hidden"], "dropout": c["dropout"],
            "class_weight": c["class_weight"], "lr": c["lr"], "epochs": c["epochs"],
            "studies": sorted(set(c["studies"])),
            "args": _args(c),
            "model": f"{OUT_DIR}/{run_id}.pt",
            "metrics": f"{OUT_DIR}/{run_id}.json",
        })

    # datasets.tsv : name <TAB> generator args
    ds_lines = [f"{name}\t{args}" for name, args in DATASETS.items()]
    (CLUSTER / "datasets.tsv").write_text("\n".join(ds_lines) + "\n")

    # runs.tsv : run_id <TAB> dataset <TAB> mlp args  (read by line in train.slurm)
    run_lines = [f"{r['run_id']}\t{r['dataset']}\t{r['args']}" for r in run_list]
    (CLUSTER / "runs.tsv").write_text("\n".join(run_lines) + "\n")

    # manifest.json : everything the notebook needs (configs + study tags + paths)
    manifest = {
        "data_dir": DATA_DIR, "out_dir": OUT_DIR, "base": BASE,
        "datasets": {name: {"args": args, "csv": f"{DATA_DIR}/{name}.csv"}
                     for name, args in DATASETS.items()},
        "runs": run_list,
    }
    (CLUSTER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # report
    by_study = Counter(s for r in run_list for s in r["studies"])
    n_abst = sum(r["loss"] == "abstention" for r in run_list)
    print(f"datasets : {len(DATASETS)}")
    print(f"runs     : {len(run_list)}  "
          f"({len(run_list) - n_abst} cascade, {n_abst} abstention)")
    for s, n in sorted(by_study.items()):
        print(f"  study {s:13s}: {n} runs")
    print("wrote cluster/datasets.tsv, cluster/runs.tsv, cluster/manifest.json")


if __name__ == "__main__":
    main()
