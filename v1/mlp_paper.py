"""The paper's exact multi-head MLP (SMT replication baseline).

Three head types, exactly as Shenoy & Ameri describe them:
  - defect head    : softmax over defect classes, cross-entropy loss   (Eq. 1)
  - mechanism heads: one softmax-per-stage, cross-entropy loss         (Eq. 2)
  - risk head      : P independent sigmoids, BCE with soft targets     (Eq. 3)

Composite loss L = lam_d*CE(defect) + sum_s lam_m*CE(mech_s) + lam_r*BCE(risk).
No abstention -- this is the safety-net baseline that should land on the paper's
Table II numbers. The model, data pipeline, train loop, and eval all live in
mlp_common; this file only pins the version and the (single) loss choice.

Run: python mlp_paper.py            # train + evaluate on the test split
     python mlp_paper.py --help     # all knobs
"""

from mlp_common import MultiHeadLoss, run_cli

MODEL_VERSION = "mlp-paper-v0.1"     # stamped into outputs (provenance)

# only the paper loss is available here (no abstention)
LOSSES = {"multihead": MultiHeadLoss}


def main() -> None:
    # hand the registry + version to the shared CLI driver
    run_cli(MODEL_VERSION, LOSSES, default_loss="multihead",
            description="Paper baseline multi-head MLP (SMT)")


if __name__ == "__main__":
    main()
