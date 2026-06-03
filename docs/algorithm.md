# SMT Synthetic Generator + Analysis — Algorithm

This document describes, in pseudocode, (A) how the synthetic SMT dataset is
generated, (B) how class/label distributions are summarized, and (C) how the
Bayes error is estimated. It also lists the improvements over the original
scratch generator so they are easy to explain to the team.

Everything domain-specific (parameters, specs, causal map, correlations, risk
constants, priors) lives in `domain/smt_paper.yaml`. The code carries no
domain values; swapping the spec for `insulin_pump.yaml` re-targets the whole
pipeline with no code change.

---

## A. Data generation

```
INPUT  : spec (YAML), seed
OUTPUT : table of N records, each with 6 parameters, defect label,
         per-stage mechanism labels, per-parameter risk targets,
         exact posterior p(y|x), and provenance.

# --- one-time setup -------------------------------------------------------
Corr  <- assemble 6x6 correlation matrix from spec.correlations
Corr  <- nearest_PD(Corr)              # eigenvalue-clip to a valid PD matrix
L     <- cholesky(Corr)

# --- 1. correlated process draw (Gaussian copula) ------------------------
for each batch b in 0..n_batches-1:
    rng   <- Generator(SeedSequence(seed).spawn()[b])   # independent stream
    Z     <- rng.standard_normal(records_per_batch, 6) @ Lᵀ   # ~ N(0, Corr)
    # 2. non-stationary drift, in sigma units:
    wear     <- wear_coef * (t_global / N)              # equipment aging trend
    diurnal  <- diurnal_coef * sin(2π (t + shift_offset) / period)
    X_b      <- nominal + sigma * (Z + wear + diurnal)  # parameter values
collect X <- concat(X_b)

# --- 3. normalized deviations --------------------------------------------
dev[:,j] <- (X[:,j] - nominal_j) / half_width_j         # |dev| = 1 at spec limit

# --- 4. defect posterior p(y|x) ------------------------------------------
for each defect d (≠ no_defect):
    score_d <- Σ over causal_edges(d): weight * relu( signed_dev )
               # signed_dev = +dev if edge.direction == high else -dev
logit(no_defect) <- 0 + offset(no_defect)        # no_defect is the reference
logit(d)         <- gain * score_d + offset(d)
# two-stage calibration:
gain    <- bisect so that  E_x[1 - max_y p(y|x)] == target_bayes_error   # separability
offset  <- IPF so that     E_x[p(y)] == prior(y)  for every class        # class balance
p(y|x)  <- softmax( [logit(no_defect), logit(d1), logit(d2)] )

# --- 5. labels, mechanisms, risk -----------------------------------------
y         <- sample Categorical(p(y|x))  with a dedicated RNG stream
for each stage s:
    mech_s <- argmax over causal_edges(y, s) of weight*relu(signed_dev), else no_mechanism
for each parameter j:
    risk_j <- graded_risk( |dev[:,j]| )    # two-sided, see section D

# --- 6. provenance + split -----------------------------------------------
attach generator_version, seed, batch_id, build_timestamp, board_id
split <- batch-grouped 70/15/15 (assign whole batches to train/val/test)
```

The posterior `p(y|x)` is stored as columns `p_no_defect`, `p_open_circuit`,
`p_solder_bridging`. This is what makes the Bayes error computable exactly.

---

## B. Class / label distributions (histograms)

```
- defect class balance      : count(y) / N           vs spec priors
- per-stage mechanism balance: count(mech_s) / N      per stage
- parameter distributions   : histogram(X[:,j]) with LSL/USL/nominal lines
- class-conditional overlays: density of X[:,j] split by y  (separability)
```

---

## C. Difficulty + defect-head metrics (paper comparison)

The generator samples `y ~ Categorical(p(y|x))` with a **known** posterior, and
the model sees exactly the features the posterior is built from. So the Bayes
error (oracle ceiling) is available in closed form, and we report the paper's
Section V-A metrics (accuracy + weighted-F1) against it.

```
# Exact Bayes error (oracle ceiling), Monte-Carlo over the known posterior:
R*  <- mean_x [ 1 - max_y p(y|x) ]              # overall and on the test split

# Bayes-optimal classifier (the ceiling): argmax of the posterior
y_opt <- argmax_y p(y|x)
acc, weighted_F1, macro_F1, per_class_PRF <- metrics(y_opt, y_sampled)   on test

# Basic MLP (same model family as the paper):
y_mlp <- MLP.fit(scaled X_train, y_train).predict(scaled X_test)
acc, weighted_F1, macro_F1, per_class_PRF <- metrics(y_mlp, y_test)

# Reference: paper Section V-A = {accuracy 95.00%, weighted-F1 95.36%}
# Reference: majority baseline = always predict the most common class
```

Interpretation: the paper's accuracy/F1 should sit between the strong learner
and the Bayes-optimal ceiling for the data to track the paper's difficulty. F1
(weighted + per-class) matters more than accuracy here because the defect
classes are imbalanced — the minority classes (open / bridging) drive the F1.
Compute everything on the **same test split** for a like-for-like comparison.

---

## D. Graded risk target (paper Eq. 8, with 9–10)

The paper's exact ground-truth risk, two-sided in `|Δ| = |deviation|` (a spec
violation is either USL or LSL, so risk is a function of distance to the *nearer*
limit; causal *direction* lives in the edges, not the risk magnitude):

```
z = min(1, max(0, (|Δ| - Δ1)/(Δ2 - Δ1)))               # Eq. 9
t = max(0, |Δ| - Δ2)                                   # Eq. 10
P = p_L + (p_M - p_L)·z^2 + (p_H - p_M)·(1 - e^{-κ·t}) # Eq. 8
```

`Δ2 = 1.0` is the spec limit. Risk = `p_L` (≈0.05) in the safe region, rises
quadratically to `p_M` (≈0.70) **at the spec limit**, and asymptotes to `p_H`
(≈0.99) out of spec — matching the paper's stated behavior. `p_L,p_M,p_H` are the
paper's stated values; `Δ1, Δ2, κ` are in the spec and remain a documented
reproducibility gap (the paper gives the form, not the constants).

---

## Improvements over the original scratch generator (`sample.py`)

| # | Original (`sample.py`) | Improved (`generator.py`) | Why it matters |
|---|------------------------|---------------------------|----------------|
| 1 | All domain values hard-coded as Python constants | Read from `domain/smt_paper.yaml` | Single source of truth; ontology/eval mirror it; medical-device port is a file swap |
| 2 | 3 latent factors linearly mapped to 6 params (correlations implicit) | Correlated multivariate-Gaussian draw (Gaussian copula with Gaussian marginals) with a specified correlation matrix + nearest-PD correction | Spec correlation is imposed on the latent draw and is auditable; drift adds independent structure on top, so final-value correlations are slightly diluted but recoverable. Swap in inverse-CDF marginals later for non-Gaussian shapes (medical port) |
| 3 | `iterrows()` over 200k rows + per-row RNG construction | Fully vectorized, per-batch RNG streams | Seconds instead of minutes |
| 4 | Per-row RNG seed `int(t + batch*17)` → collisions; `datetime.now()` in data | `SeedSequence(seed).spawn()` streams; fixed build timestamp | Byte-reproducible given a seed; no collisions |
| 5 | Class balance emergent, uncontrolled | All three classes calibrated to the paper's exact Table I priors (IPF on logit offsets) | Matches the paper's class distribution, not just the no-defect rate |
| 6 | Labeler sampled labels but did not expose `p(y|x)` | Exact posterior emitted as columns | Bayes error computable in closed form |
| 7 | Difficulty uncontrolled | Logit gain auto-solved so the Bayes floor hits a target (0.045 → oracle ceiling ~95.5%), making the paper's reported 95.00% defect-head accuracy achievable | Synthetic data tracks the paper's difficulty regime, not an arbitrary one |
| 8 | Ad-hoc risk shape | Risk is exactly the paper's Eq. (8)–(10); two-sided `P(out of spec)`, direction kept in causal edges for chains | Faithful to the paper; correct semantics for risk and RCA chains |
| 9 | No provenance fields | `generator_version`, `seed`, `batch_id`, `build_timestamp`, `board_id` per record | Audit discipline the methodology requires |
| 10 | Two stages as named Python vars; partial causal map | Stages derived from spec; full causal map matching paper Fig. 2 (all 6 params, mirror directions) | Reconfigurable; faithful to the paper's causal structure |

## Reproducibility gaps (carried from the paper, documented for revision)

- Risk-function constants `Δ1, Δ2, κ` — the paper gives the form (Eq. 8–10) and
  `p_L,p_M,p_H` (0.05/0.70/0.99) but not these; `Δ2 = 1.0` (spec limit), `Δ1 = 0.5`,
  `κ = 3.0` chosen so risk hits ≈0.70 at the limit and asymptotes to 0.99.
- Defect-labeling rule — not described; implemented as a calibrated logit model
  over causal-edge scores producing a proper posterior, with the gain solved to a
  target Bayes error so the oracle accuracy ceiling (~95.5%) brackets the paper's
  reported 95.00% (the paper's number is achievable, sitting just below the ceiling).
- Correlation matrix, sigmas, drift coefficients, causal-edge weights — not
  released; chosen to be process-plausible and to land the paper's Table I balance.
