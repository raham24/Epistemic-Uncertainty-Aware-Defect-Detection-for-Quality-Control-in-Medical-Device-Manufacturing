"""Build the cluster experiment matrix: dataset variants x model runs.

Single source of truth for the sweep. Emits three files into cluster/:

  datasets.tsv   one line per dataset variant:  name <TAB> generator args
  runs.tsv       one line per training run:      run_id <TAB> dataset <TAB> mlp args
  manifest.json  the full structured matrix (paths included) for the notebook

The SLURM arrays (gen_data.slurm, train.slurm) and the local runner read the .tsv
files by line number ($SLURM_ARRAY_TASK_ID); the analysis notebook reads
manifest.json. Edit DATASETS / LOSSES / O_VALUES / SEEDS below to change the
sweep, then re-run `python cluster/build_matrix.py` and re-submit.

Stdlib only -- runs anywhere, no heavy deps.
"""

import json
from pathlib import Path

# repo root = parent of this file's folder (cluster/). Paths in the manifest are
# repo-root-relative; every script cd's to the root before using them.
ROOT = Path(__file__).resolve().parent.parent
CLUSTER = ROOT / "cluster"
DATA_DIR = "data/cluster"
OUT_DIR = "results/cluster"

# --- dataset variants: name -> generator CLI args (no --out, no spaces in a
# single flag value). Vary the class balance ("split") and the difficulty. ---
DATASETS = {
    "baseline":      "",                                                    # paper balance, bayes 0.045
    "balanced":      "--priors no_defect=0.5,open_circuit=0.25,solder_bridging=0.25",
    "hard":          "--bayes-error 0.09",                                  # harder labels
    "hard_wide":     "--bayes-error 0.09 --sigma-scale 1.5",                # harder + wider spread
    "balanced_hard": "--priors no_defect=0.5,open_circuit=0.25,solder_bridging=0.25 --bayes-error 0.09",
}

# --- model sweep ---
LOSSES = ["cascade", "abstention"]   # both loss functions
O_VALUES = [2.0, 4.0]                 # abstention payoff sweep (cascade ignores o)
SEEDS = [0, 1]                        # repeats for variance
EPOCHS = 40


def iter_runs():
    """Yield one config dict per training run (the Cartesian sweep)."""
    for ds in DATASETS:
        for loss in LOSSES:
            o_values = O_VALUES if loss == "abstention" else [None]
            for o in o_values:
                for seed in SEEDS:
                    run_id = f"{ds}__{loss}"
                    args = f"--loss {loss} --seed {seed} --epochs {EPOCHS}"
                    if o is not None:
                        run_id += f"_o{o:g}"
                        args += f" --o {o}"
                    run_id += f"_s{seed}"
                    yield {
                        "run_id": run_id, "dataset": ds, "loss": loss,
                        "o": o, "seed": seed, "epochs": EPOCHS, "args": args,
                        "model": f"{OUT_DIR}/{run_id}.pt",
                        "metrics": f"{OUT_DIR}/{run_id}.json",
                    }


def main():
    CLUSTER.mkdir(exist_ok=True)

    # datasets.tsv : name <TAB> args
    ds_lines = [f"{name}\t{args}" for name, args in DATASETS.items()]
    (CLUSTER / "datasets.tsv").write_text("\n".join(ds_lines) + "\n")

    # runs.tsv : run_id <TAB> dataset <TAB> args
    run_list = list(iter_runs())
    run_lines = [f"{r['run_id']}\t{r['dataset']}\t{r['args']}" for r in run_list]
    (CLUSTER / "runs.tsv").write_text("\n".join(run_lines) + "\n")

    # manifest.json : everything the notebook needs (configs + output paths)
    manifest = {
        "data_dir": DATA_DIR, "out_dir": OUT_DIR,
        "datasets": {name: {"args": args, "csv": f"{DATA_DIR}/{name}.csv"}
                     for name, args in DATASETS.items()},
        "runs": run_list,
    }
    (CLUSTER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    n_abst = sum(r["loss"] == "abstention" for r in run_list)
    print(f"datasets : {len(DATASETS)}")
    print(f"runs     : {len(run_list)}  ({len(run_list) - n_abst} cascade, {n_abst} abstention)")
    print("wrote cluster/datasets.tsv, cluster/runs.tsv, cluster/manifest.json")


if __name__ == "__main__":
    main()
