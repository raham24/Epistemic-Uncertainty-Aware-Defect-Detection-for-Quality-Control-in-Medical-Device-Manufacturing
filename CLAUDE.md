# Medical Device Neurosymbolic RCA — Project Context

## Mission

Replicate the neurosymbolic root-cause analysis pipeline from Shenoy & Ameri's 2026 IEEE TSM paper (DOI 10.1109/TSM.2026.3673999) in a medical device context, grounded in MAUDE adverse-event data, and integrated with the existing MAUDEKG knowledge graph hosted on okn.us as part of NSF Proto-OKN.

The original paper applies the methodology to surface-mount assembly (SMT) — solder paste printing and reflow stages, six monitored process parameters, two defect classes. This project ports the methodology to medical devices while keeping the pipeline architecture identical.

Differentiator from the original paper: every RCA hypothesis produced by this pipeline can be linked to real MAUDE adverse-event records as supporting evidence, leveraging MAUDEKG. The original paper has no real-world evidence backing — every hypothesis points only to its own synthetic measurements. That linkage is the methodological contribution.

## Three Core Commitments (Non-Negotiable)

**1. MAUDE is a knowledge source, never training data.** Mine MAUDE narratives via LLM extraction to populate the synthetic generator's domain spec (defect vocabulary, mechanism vocabulary, causal map edges, empirical class priors). Never train an MLP directly on MAUDE narratives. MAUDE's data shape (free-text, post-market, biased reporting, no ground truth) makes it categorically unsuitable as training data for the paper's methodology.

**2. Replicate SMT first.** Before writing any medical device code, build a working SMT reproduction. Verify the multi-head MLP hits roughly the paper's Table II numbers on SMT-shaped synthetic data. This is the safety net — if the medical device port fails downstream, the SMT replication still stands as a publishable result.

**3. Thin slice end-to-end before deep work.** Get the full pipeline (generator → MLP → KG → RCA packet) running with stub components first. Don't perfect any one component before the whole thing runs end-to-end. The goal is to always have a working system to demo.

## Device Class

**Primary scope: external insulin pumps.**

Rationale:
- High MAUDE volume (thousands of reports per year)
- Well-defined failure modes: occlusion, motor failure, battery failure, dose accuracy errors, alarm failures, software faults
- Standards-anchored specs: IEC 60601-2-24 (infusion pumps), AAMI/ANSI ID26, FDA guidance
- Continuous monitorable parameters (motor current, occlusion pressure, battery voltage, reservoir level, dose accuracy) that map naturally onto the SMT per-board pattern
- Mature engineering literature on failure mechanisms

Fallbacks if insulin pumps don't yield enough MAUDE volume or pattern clarity: broader infusion pumps, continuous glucose monitors, automated external defibrillators.

Avoid for v1: implantables, surgical robots, anything software-dominated. Failure-mode parameterization is much harder for these.

## Architecture

Two strictly decoupled layers, coupled through a single boundary artifact.

### Neural pipeline

Multi-head feedforward MLP with shared trunk and task-specific output heads.

- **Shared trunk:** fully-connected layers, ReLU activations. Produces latent representation `h = f_θ(x)`.
- **Defect head:** softmax over defect classes. Argmax for selection.
- **Stage-wise mechanism heads:** one softmax head per manufacturing stage. Argmax per stage.
- **Parameter risk heads:** independent sigmoid per monitored parameter. Outputs continuous risk in (0, 1) representing the calibrated probability that the parameter is in violation of its specification.

Critical: the risk heads regress onto a graded continuous target from the ground-truth function (Eq. 8 in the paper), not a binary in-spec/out-of-spec label. Loss is BCE with soft targets, not classification.

Loss function (paper does not specify; inferred):
```
L = λ_d · CE(defect) + Σ_s λ_m · CE(mechanism_s) + Σ_j λ_r · BCE(risk_j)
```
Class imbalance handling needed for the defect head (paper has ~87.7% no-defect; expect similar shape for medical devices).

### Symbolic pipeline

- **Ontology:** OWL 2 (Protégé for authoring). Five top-level classes: `Defect`, `Mechanism`, `Parameter`, `ParameterViolation`, `ConformanceAssessment`. Import PROV-O for provenance.
- **Causal map:** Ishikawa-style cause-effect graph encoded as OWL object-property assertions. Defect → Mechanism(s) → ParameterViolation(s).
- **Triple store:** target the FABRIC/FRINK SPARQL endpoint already provisioned for MAUDEKG.
- **Rule engine:** SWRL if reasoner supports it; otherwise thin Python layer over SPARQL CONSTRUCT queries. Functionally equivalent.
- **Reasoning is deterministic** — probabilities never gate rule firing. They are annotations on instantiated individuals used for ranking and audit.

### Evidence interface (the boundary)

Each instantiated individual (defect, mechanism, parameter violation) is bound to exactly one `ConformanceAssessment` evidence object containing:
- `hasProbability` (neural output)
- `hasAssessmentInput` (links to spec individuals and observation individuals)
- `hasSourceModel` (model version identifier)
- `performed_on` (PCB/device instance identifier)
- Context identifiers (batch/shift id, inference timestamp)

This is the single artifact that bridges neural and symbolic layers.

### RCA packet

Output of the end-to-end pipeline for each test instance:
- Predicted defect + stage-wise mechanisms
- Ranked list of `defect → mechanism → parameter_violation` chains
- Each chain scored by terminal parameter risk
- Provenance pointers per hypothesis (observation values, spec bounds, timestamp, model version)
- **Extension beyond the paper:** linked MAUDE adverse-event records from MAUDEKG matching each hypothesis's failure pattern

## Implementation Status (As-Built — SMT baseline, Phases 2–4 done)

The SMT replication (Commitment 2) is built and running at scale on GPU. This section is the ground truth for what exists; it extends/deviates from the aspirational Architecture above.

### Code layout (actual — NOT the `src/…` target below)
Two self-contained root files are the whole pipeline; no cross-imports, each reads `domain/smt_paper.yaml`:
- `generator.py` — synthetic generator (copula + drift + calibrated labels + graded risk). `VERSION="smt-gen-2.0"`.
- `mlp.py` — multi-head cascade model, both losses, train/evaluate/CLI. `MODEL_VERSION="mlp-cascade-1.0"`.
- `domain/smt_paper.yaml` — THE spec (every field consumed; no dead fields).
- `cluster/` — HPC sweep infra (tracked). `misc/` — pytest unit tests (gitignored). `v1/` — archived first cut (untouched). `docs/` — gitignored notes (`generator_pseudocode.md`, `project_brief.md`).
- Env: conda env `paper`. Local dev on macOS (`mps`); training on the GPU cluster.

### Generator — calibrated label model (resolves the paper's biggest gap)
Defect labels are **sampled from an explicit posterior `p(y|x)`**, not thresholded:
- Ishikawa causal scores → logits `ℓ_d = gain·score_d + offset_d` (no_defect = reference 0).
- `gain` is **binary-searched so the realized Bayes error hits `label_model.target_bayes_error`** (default 0.045 → ~95.5% optimal-accuracy ceiling, reproducing the paper's ~95%).
- `offset_d` fit by **IPF so class marginals match the priors**.
- The exact posterior is emitted (`p_<defect>` columns), so the **Bayes-optimal accuracy is known in closed form** — the yardstick the model is measured against. This calibration is ours, not the paper's.
- Separable knobs: difficulty = `--bayes-error` (gain); class balance = `--priors` (offsets); process spread = `--sigma`/`--sigma-scale` (physical — drives risk targets + spec-violation rate; the label calibration absorbs it, so sigma does NOT change defect difficulty). Batch-grouped leak-free train/val/test split.

### Model — 3-head TRUE cascade (deviates from the "per-stage heads" sketch above)
- Trunk → **head1 defect** (3-way) → **head2 mechanism** = ONE **joint 9-class** softmax over the `<printing>__<reflow>` Cartesian product (7 realizable) → **head3 risk** (6 independent sigmoids, graded-BCE targets).
- True cascade: each downstream head reads the trunk latent ⊕ the upstream heads' **soft softmax distributions** (differentiable, end-to-end, no teacher forcing).
- Defaults: hidden (256,256), dropout 0.1, Adam lr 1e-3, batch 128, 40 epochs + early stop.

### Two losses (`--loss`)
- `cascade` (default) — CE(defect)+CE(mech)+BCE(risk), the paper's form.
- `abstention` — selective-classification term `−log(o·p_y + r)` (logsumexp form) on the **defect head ONLY**; the mechanism head reverts to plain CE. Two-head abstention is commented out (not deleted) in `mlp.py` — flip it back to restore. `--o` = payoff; only the defect head is widened by one "abstain" column. Do NOT use the word "gambler" anywhere.

### Results / findings (SMT)
- **Cascade tracks the Bayes ceiling:** ~0.95 defect accuracy, gap-to-ceiling ~0.002 across all regimes; joint-mechanism ~0.90–0.95; risk MAE ~0.011. Faithful Table-II replication.
- **Abstention is a careful NEGATIVE result:** it loses to cascade on forced (full-coverage) accuracy, and at matched coverage it does **not** beat simply confidence-thresholding the cascade (`gain_vs_conf ≤ 0` almost everywhere). Because cascade recovers the true posterior, its softmax confidence is already a near-Bayes-optimal reject signal, leaving nothing for the learned reject head to add. This is a finding to report honestly, not a bug. Forced accuracy = argmax over real classes on ALL boards; selective accuracy = accuracy only on non-abstained boards (`r < h`, default h=0.5) and is meaningless without its coverage.

### Cluster sweep (`cluster/`, Star HPC at Hofstra)
Sweep matrix, `submit.sh`, `RCA_*` knobs, Star HPC partition/QOS facts, and the analysis notebook: **`cluster/CLAUDE.md`** (loaded when working under `cluster/`).

## Phase Plan (10 weeks, June–August 2026)

| Phase | Weeks | Focus | Deliverable |
|-------|-------|-------|-------------|
| 0 | 1 | Device class lock | Scoping doc with defect/parameter vocabulary and standards citations |
| 1 | 1–2 | MAUDE knowledge extraction | YAML domain spec with vocabularies, causal map, empirical class priors |
| 2 | 3 | SMT replication baseline | Working SMT pipeline producing Table II-like numbers |
| 3 | 4–5 | Medical device generator | 200k-record synthetic dataset with labels and provenance |
| 4 | 6 | Neural training | Trained MLP checkpoint, evaluation report |
| 5 | 7 | Symbolic pipeline | OWL ontology, populated triple store, rule engine, sample RCA packets |
| 6 | 8 | MAUDEKG provenance integration | Extended RCA packets with real-world evidence linking |
| 7 | 9 | Evaluation + bmedesign.org hook | Full metrics, demo page |
| 8 | 10 | Documentation + paper | Documentation package, paper draft outline |

Minimum viable result if anything goes wrong:
- Phases 0–2 complete → working SMT reproduction (publishable replication)
- Through Phase 4 → working medical device pipeline producing paper's metrics (publishable)
- Through Phase 6 → MAUDE-grounded evidence linking (headline contribution)

The risky parts (FABRIC/FRINK integration, bmedesign.org hook) are sequenced last on purpose. Phase 4 is a complete unit on its own.

## Technical Stack

Runtime and core libraries: see `environment.yml` (conda env `paper`).

**Semantic stack:**
- `owlready2` or `rdflib` — Python OWL/RDF manipulation
- Apache Jena Fuseki — local triple store for development
- FABRIC/FRINK SPARQL endpoint — production target
- Protégé — ontology authoring (out-of-band, not in code)

**LLM extraction (Phase 1):**
- Anthropic API (Claude Opus for narrative extraction)
- Structured output for clean tuple extraction from MAUDE narratives

## Domain Spec Contract

Single YAML file is the contract between every component. The generator reads it. The ontology mirrors it. Evaluation references it. **Full schema: `domain/CLAUDE.md`** (loaded when working under `domain/`).

When in doubt about generator behavior, the YAML is authoritative.

## Coding Standards

- **Style:** Black-formatted Python, type hints required on public functions, docstrings on modules and classes.
- **Tests:** `pytest` for unit tests on the generator, model, and rule engine. Synthetic generator should be deterministic given a fixed seed — test this explicitly.
- **Provenance everywhere:** every emitted record carries `generator_version`, `seed`, `batch_id`, `timestamp`. Every model output carries `model_version`. Every RCA packet carries pointers back to source data. Non-negotiable — mirrors the paper's audit discipline.
- **YAML config over hard-coded constants.** Anything domain-specific lives in the spec file, not in code.
- **Random seed control:** all stochastic operations accept and pass through a seed parameter. No global RNG state.

## Project Structure (target)

The target layout for the FULL system (incl. the symbolic layer, Phases 5-8) lives in the `target-structure` skill (`.claude/skills/target-structure/SKILL.md`). The SMT baseline that exists today is NOT that `src/…` layout — see Implementation Status (As-Built) above.

## Open Questions (Decisions Pending)

- Final device class confirmation. Insulin pumps recommended but contingent on MAUDE volume sanity-check in Phase 1.
- Final parameter list. Need to commit before generator implementation. Source: IEC 60601-2-24 and MAUDE patterns.
- Stage decomposition for medical devices. Options: keep two stages like the paper (delivery / dose accuracy), collapse to single stage, or expand to three (delivery / dose accuracy / alarm subsystem).
- Whether to use PyTorch or sklearn for the prototype MLP. PyTorch preferred for full implementation regardless.
- Specific OKN team interaction model. Raham has team members at okn.us who built MAUDEKG — need to clarify division of work for Phase 5–6 integration.

## Working Preferences

- **Heavy commands are run by Raham himself.** Data generation, training, and cluster submissions: provide the command, do not run it.
- **Do not push to git.** He pushes.
- **Editor:** Raham uses vim exclusively. Do not suggest `nano` or other editors.
- **No emojis** in any output.
- **Technical density is fine.** Don't dilute explanations for accessibility; he's a CS grad student with thesis work in CUDA/cryptography/ZKP and ML coursework.
- **Concrete over abstract.** When discussing approaches, prefer concrete code/spec examples over conceptual descriptions.
- **Honest about limitations.** Flag reproducibility gaps in the paper directly. Don't paper over uncertainty.

## Key References

- **Original paper:** Shenoy, M., & Ameri, F. (2026). "Uncertainty-Aware Neurosymbolic Root-Cause Analysis for Surface-Mount Assembly." IEEE Transactions on Semiconductor Manufacturing. DOI: 10.1109/TSM.2026.3673999
- **Companion paper (unpublished):** [10] in the references — physics-informed provenance-aware RCA with corrective action support
- **MAUDE database:** https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/search.cfm
- **MAUDEKG endpoint:** okn.us (via FRINK registry, FABRIC infrastructure)
- **NSF Proto-OKN context:** Hofstra contribution under Theme 1 (Knowledge Graph) operating under Theme 2 (FABRIC)
- **Target user community:** Hofstra senior capstone medical device design course (https://www.bmedesign.org/)
- **PROV-O specification:** https://www.w3.org/TR/prov-o/
- **IEC 60601-2-24:** Particular requirements for infusion pumps and controllers
- **AAMI/ANSI ID26:** Medical electrical equipment infusion devices

## Reproducibility Gaps in the Original Paper (Important)

The paper omits several implementation details that will need to be re-derived:

1. **Loss function:** never explicitly stated. Inferred above.
2. **Knowledge graph technology:** "graph database" mentioned but no product named. Linguistic tells (use of "individuals") strongly suggest OWL/RDF stack. Mukund Shenoy at ASU may be willing to share details.
3. **Synthetic generator code:** not released. The Gaussian copula approach is described but specific correlation matrix, spec bounds, drift parameters, and risk-function constants (`p_L`, `p_M`, `p_H`, `κ`, `Δ_1`, `Δ_2`) are not provided. Reasonable defaults documented in this file's risk_function section.
4. **MLP architecture details:** trunk depth, width, dropout, learning rate, batch size, optimizer all unspecified. Start with conservative defaults (2 hidden layers of 256, Adam, lr=1e-3, batch=128, dropout=0.1).
5. **Defect labeling rule:** how defect labels are assigned from parameter values is never described. Build a simple logit-based rule that produces the paper's ~87% no-defect class balance.

**Status (SMT baseline):** gaps 1, 3, 4, 5 are RESOLVED — see Implementation Status (As-Built). Loss functions (both `cascade` and `abstention`), the generator, the MLP architecture + hyperparameters, and a calibrated label model (labels sampled from `p(y|x)`, not a threshold) are all implemented. Gap 2 (knowledge-graph technology) remains open, deferred to Phases 5–6.

When implementing, document these decisions explicitly so they can be revised when more information becomes available.
