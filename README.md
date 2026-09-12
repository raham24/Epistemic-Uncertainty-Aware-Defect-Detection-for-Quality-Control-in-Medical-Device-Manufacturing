# Epistemic Uncertainty-Aware Defect Detection for Quality Control in Medical Device Manufacturing

Code for the paper's experiments. A shared-trunk multi-head MLP with an optional
**learned-abstention** head (selective classification: predict, or decline when
uncertain) is studied on two pipelines:

1. **Synthetic (SMT) benchmark** — a calibrated generator produces labeled boards;
   a 3-head cascade MLP (defect → mechanism → per-parameter risk) is trained across
   difficulty regimes and payoff levels.
2. **Real-data (MAUDE) case study** — adverse-event narratives from the openFDA MAUDE
   API are featurized (TF-IDF → SVD → standardize) and classified by severity with the
   same abstention MLP.

Both are deterministic given a seed; results are reported as means over seeds. Each
pipeline has a single-machine path and a dependency-chained SLURM sweep, and each ends
in an analysis notebook that renders the paper figures.

## Layout

```
generator.py                 synthetic board generator (calibrated posterior labels)
mlp.py                        synthetic cascade MLP: model, both losses, train/evaluate CLI
domain/smt_paper.yaml         the synthetic domain spec (authoritative; generator reads it)
cluster/                      synthetic SLURM sweep + notebook builder
real-data/                    MAUDE scrape/featurize + abstention MLP + train-from-cache
real-data/cluster/            MAUDE SLURM sweep + notebook builder
synthetic_analysis.ipynb      synthetic figures  (built by cluster/build_analysis_nb.py)
real_data_analysis.ipynb      MAUDE figures      (built by real-data/cluster/build_analysis_nb.py)
environment.yml               conda environment (env `paper`, Python 3.11)
misc/                         pytest unit tests (generator/model)
```

Generated artifacts are gitignored and not distributed: datasets (`data/`), sweep
outputs (`results/cluster/`, `real-data/results/cluster/`), the feature cache, and
figures (`figs/`).

## Environment

```bash
conda env create -f environment.yml
conda activate paper
```
PyTorch, NumPy, pandas, scikit-learn, matplotlib, PyYAML, joblib. Training runs on CPU,
CUDA, or Apple MPS via `--device {cpu,cuda,mps}`.

## 1. Synthetic (SMT) pipeline

`generator.py` samples boards from an explicit posterior `p(y|x)` calibrated to a target
Bayes error and class priors; `mlp.py` trains the cascade. `domain/smt_paper.yaml` is the
single source of truth (parameters, spec limits, causal map, priors, Eq. 8 risk).

**Single run:**
```bash
python generator.py --out data/baseline.csv --seed 0
python mlp.py --data data/baseline.csv --loss cascade --seed 0 --device cpu
python mlp.py --data data/baseline.csv --loss abstention --o 2.0 --seed 0   # selective variant
```
Difficulty/balance overrides (never mutate the spec): `--bayes-error` (label difficulty),
`--priors` (class balance), `--sigma-scale`/`--sigma` (process spread).

**Full sweep (SLURM):**
```bash
bash cluster/submit.sh          # build matrix -> gen-data array (CPU) -> train array (GPU), chained
# local fallback (no SLURM):
bash cluster/run_local.sh
```
`cluster/build_matrix.py` emits the run matrix (`manifest.json` + `runs.tsv` +
`datasets.tsv`): the loss × difficulty-regime grid plus one-factor studies (payoff `o`,
capacity, dropout, learning rate, class weighting), each replicated over `RCA_SEEDS`
(default `0,1,2,3,4`). Outputs land in `data/cluster/` and `results/cluster/`.

## 2. Real-data (MAUDE) pipeline

`real-data/prep_dataset.py` scrapes and featurizes **once** (needs network + an openFDA
API key), caching a fixed dataset + split + feature matrix so every run reads identical
data. `train_from_cache.py` trains the abstention MLP off that cache (4-class severity:
Malfunction / Basic injury / Serious injury / Death).

**Prep + sweep (SLURM):**
```bash
export OPENFDA_API_KEY=...                 # https://open.fda.gov/apis/authentication/
bash real-data/cluster/submit.sh           # build matrix -> scrape once -> train array, chained
SKIP_PREP=1 bash real-data/cluster/submit.sh   # reuse an existing cache
# local fallback:
bash real-data/cluster/run_local.sh
```
`real-data/cluster/build_matrix.py` sweeps the full payoff range `o` for every
hyperparameter config (class weighting, capacity, dropout, lr), replicated over seeds.
Outputs land in `real-data/results/cluster/`.

## 3. Cluster jobs (SLURM)

Both sweeps follow the same shape and are configured entirely through `env.sh` (`RCA_*`
variables — partition, account, QOS, GPUs, seeds, array throttle/size):

- **Synthetic** (`cluster/`): `gen_data.slurm` (CPU array, one dataset per task) →
  `train.slurm` (GPU array, one model per task); the train array waits on the data array
  via a SLURM dependency. `submit.sh` builds the matrix, submits both, and chunks the
  training array under the account's array/submit caps.
- **MAUDE** (`real-data/cluster/`): `prep.slurm` (scrape/featurize once) → `train.slurm`
  (GPU array); the train array waits on the scrape. `SKIP_PREP=1` reuses the cache.

Placement is empty by default, so jobs land on the cluster's default partition; set the
`RCA_*` vars in the respective `env.sh` for your site. `run_local.sh` reproduces either
sweep sequentially without SLURM.

## 4. Analysis notebooks

Each notebook walks up to find its sweep manifest, then reads the per-run metrics and
renders every figure to `figs/fig_*.{png,pdf}`:

- `synthetic_analysis.ipynb` ← `cluster/manifest.json` + `results/cluster/*.json`
  (feature-separability cells also read `data/cluster/*.csv`).
- `real_data_analysis.ipynb` ← `real-data/cluster/manifest.json` +
  `real-data/results/cluster/*.json`.

Run the corresponding sweep first so those files exist, then run the notebook
top-to-bottom. The notebooks are generated by the `build_analysis_nb.py` scripts in each
`cluster/` directory — edit those and re-run to regenerate.

## 5. Network architectures and settings

### Synthetic cascade MLP (`mlp.py`, `CascadeMLP`)
- **Input:** 6 standardized process-parameter features.
- **Shared trunk:** 2 hidden layers **(256, 256)**, each `Linear → ReLU → Dropout(0.1)`.
- **Head 1 — defect:** `Linear(256 → 3)` → softmax (+1 reject column when abstaining).
- **Head 2 — mechanism:** `Linear(256 + Head1 → 9)` → softmax (joint printing×reflow, 7 realizable).
- **Head 3 — risk:** `Linear(256 + Head1 + Head2 → 6)` → 6 independent sigmoids (graded risk).
- **Loss:** `CE(defect) + CE(mech) + BCE(risk)`; abstention swaps the defect CE for `−log(o·p_y + r)`.
- **Optimizer:** Adam, lr **1e-3**, batch **128**, ≤ **40 epochs** (early stop, patience 6), no weight decay.

### Real-data abstention MLP (`maude_product_problem_abstention_mlp.py`, `AbstentionMLP`)
- **Input:** text → TF-IDF (50k features, 1–2 grams, min_df 2, max_df 0.95) → TruncatedSVD **(300)** → StandardScaler.
- **Shared trunk:** 2 hidden layers **(256, 128)**, each `Linear → ReLU`.
- **Head:** `Linear(128 → 4)` → softmax (+1 reject column when abstaining).
- **Loss:** class-weighted cross-entropy; abstention uses `−log(o·p_y + r)`.
- **Optimizer:** Adam, lr **1e-3**, weight decay **1e-4**, batch **32**, ≤ **80 epochs** (early stop, patience 10).

Activations: ReLU in every hidden layer; softmax on class heads, sigmoid on the risk head.

### Hardware
- Development / CPU: Apple Silicon MacBook (PyTorch MPS).
- Sweeps: a SLURM HPC cluster, one GPU per run, NVIDIA **H100 / A100 / A30**.
- Determinism: fixed per-run seeds; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Tests

```bash
pytest misc/          # generator + model unit tests (deterministic given a seed)
```
