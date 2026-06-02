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

**Language:** Python 3.11+

**Core libraries:**
- `numpy`, `scipy.stats`, `scipy.linalg` — sampling, distributions, nearest-PD correction for correlation matrix
- `pandas` — per-record table assembly
- `pytorch` — multi-head MLP (sklearn acceptable for prototype but PyTorch for the real implementation)
- `pyyaml` — domain spec config

**Semantic stack:**
- `owlready2` or `rdflib` — Python OWL/RDF manipulation
- Apache Jena Fuseki — local triple store for development
- FABRIC/FRINK SPARQL endpoint — production target
- Protégé — ontology authoring (out-of-band, not in code)

**LLM extraction (Phase 1):**
- Anthropic API (Claude Opus for narrative extraction)
- Structured output for clean tuple extraction from MAUDE narratives

## Domain Spec Contract

Single YAML file is the contract between every component. The generator reads it. The ontology mirrors it. Evaluation references it. Schema:

```yaml
device_class: external_insulin_pump
standards:
  - IEC 60601-2-24
  - AAMI/ANSI ID26
defects:
  - name: no_failure
    prior: 0.85  # calibrated from MAUDE
  - name: occlusion_event
    prior: 0.07
  # ...
mechanisms:
  stage_1_delivery:
    - name: pump_motor_stall
    - name: tubing_kink
    - name: no_mechanism
  stage_2_dose_accuracy:
    - name: stepper_drift
    - name: no_mechanism
parameters:
  - id: motor_current
    nominal: 120.0
    usl: 180.0
    lsl: 60.0
    unit: mA
    stage: stage_1_delivery
  - id: occlusion_pressure
    nominal: 100.0
    usl: 250.0
    unit: mmHg
    stage: stage_1_delivery
  # ... 4-6 parameters total
causal_edges:
  - from: occlusion_event
    via: pump_motor_stall
    parameter: motor_current
    direction: high
  # ...
correlations:
  motor_current: { occlusion_pressure: 0.7, battery_voltage: -0.3 }
  # ...
risk_function:
  p_L: 0.05
  p_M: 0.70
  p_H: 0.99
  kappa: 0.015
generator:
  n_records: 200000
  n_batches: 100
  records_per_batch: 2000
  seed: 42
```

When in doubt about generator behavior, the YAML is authoritative.

## Coding Standards

- **Style:** Black-formatted Python, type hints required on public functions, docstrings on modules and classes.
- **Tests:** `pytest` for unit tests on the generator, model, and rule engine. Synthetic generator should be deterministic given a fixed seed — test this explicitly.
- **Provenance everywhere:** every emitted record carries `generator_version`, `seed`, `batch_id`, `timestamp`. Every model output carries `model_version`. Every RCA packet carries pointers back to source data. Non-negotiable — mirrors the paper's audit discipline.
- **YAML config over hard-coded constants.** Anything domain-specific lives in the spec file, not in code.
- **Random seed control:** all stochastic operations accept and pass through a seed parameter. No global RNG state.

## Project Structure (target)

```
medical-device-rca/
├── CLAUDE.md                       ← this file
├── pyproject.toml
├── README.md
├── domain/
│   ├── insulin_pump.yaml          ← THE domain spec
│   └── smt_paper.yaml             ← SMT replication baseline
├── src/
│   ├── generator/                  ← synthetic data generation
│   │   ├── sampler.py             ← Gaussian copula
│   │   ├── drift.py               ← wear, diurnal, batch effects
│   │   ├── labeler.py             ← defect/mechanism label assignment
│   │   └── risk.py                ← Equation 8 implementation
│   ├── model/                      ← multi-head MLP
│   │   ├── architecture.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── ontology/                   ← OWL building
│   │   ├── build.py
│   │   └── individuals.py
│   ├── reasoning/                  ← rule engine
│   │   ├── rules.py
│   │   └── chain_builder.py
│   ├── maudekg/                    ← SPARQL integration
│   │   ├── extract.py             ← LLM extraction from narratives
│   │   └── evidence_linker.py     ← link RCA hypotheses to MAUDE
│   └── rca_packet/
│       └── builder.py
├── tests/
├── data/                           ← gitignored, generated
└── docs/
    ├── plan.md
    └── domain-spec.md
```

## Open Questions (Decisions Pending)

- Final device class confirmation. Insulin pumps recommended but contingent on MAUDE volume sanity-check in Phase 1.
- Final parameter list. Need to commit before generator implementation. Source: IEC 60601-2-24 and MAUDE patterns.
- Stage decomposition for medical devices. Options: keep two stages like the paper (delivery / dose accuracy), collapse to single stage, or expand to three (delivery / dose accuracy / alarm subsystem).
- Whether to use PyTorch or sklearn for the prototype MLP. PyTorch preferred for full implementation regardless.
- Specific OKN team interaction model. Raham has team members at okn.us who built MAUDEKG — need to clarify division of work for Phase 5–6 integration.

## Working Preferences

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

When implementing, document these decisions explicitly so they can be revised when more information becomes available.
