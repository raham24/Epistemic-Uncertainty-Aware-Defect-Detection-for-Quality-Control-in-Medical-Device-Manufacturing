---
name: target-structure
description: Target directory layout for the FULL neurosymbolic system including the symbolic layer (Phases 5-8). Use when planning where new ontology, reasoning, MAUDEKG, or RCA-packet code should live. The SMT baseline that exists today does NOT follow this layout.
---


> Target for the FULL system (incl. the symbolic layer). The SMT baseline that exists
> today is NOT this `src/…` layout — it is two self-contained root files
> (`generator.py`, `mlp.py`) plus `domain/`, `cluster/`, `v1/` (archived). See
> Implementation Status (As-Built) above.

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
