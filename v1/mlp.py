"""Our multi-head MLP: the abstention ("gambler") loss on BOTH classification
heads, the paper's sigmoid risk head untouched.

Same three head types as the paper (defect, per-stage mechanism, per-parameter
risk), but the two CLASSIFICATION heads (defect AND each mechanism head) train
with the selective-classification gambler loss instead of plain cross-entropy.
Each of those heads gets one extra "abstain" output so the model can flag an
ambiguous board / stage instead of guessing. The risk head stays sigmoid + BCE,
exactly as in the paper.

  - defect head    : softmax over m+1 (m classes + abstain), gambler loss
  - mechanism heads: one softmax-per-stage over k+1, gambler loss
  - risk head      : P independent sigmoids, BCE with soft targets   (unchanged)

Gambler term per head: -log(o*p_y + r), p_y = true-class prob, r = abstain prob.
o is the payoff; r=0 reduces it to cross-entropy + const, so larger o -> predict
more / abstain less. The model, data pipeline, train loop, and eval live in
mlp_common; this file only adds the abstention loss and wires it into the CLI.

The paper baseline (no abstention) lives in mlp_paper.py.

Run: python mlp.py                       # train with the abstention loss (o=2)
     python mlp.py --loss multihead      # plain CE baseline, for comparison
     python mlp.py --o 4                  # higher payoff -> fewer abstentions
"""

import torch

# re-export the shared pieces so existing scripts keep importing them from `mlp`
from mlp_common import (LAMBDA_D, LAMBDA_M, LAMBDA_R, MultiHeadLoss,  # noqa: F401
                        evaluate, gambler_term, load_model, run_cli, train)

MODEL_VERSION = "mlp-v0.1"           # stamped into outputs (provenance)


class AbstentionLoss(MultiHeadLoss):
    """Selective-classification "gambler" loss (main.pdf) on BOTH classification
    heads. Each of the defect head and every mechanism head is one column wider
    (the abstain output), and its term becomes -log(o*p_y + r) instead of CE.
    r=0 reduces each term to cross-entropy + const. Larger o -> predict more,
    abstain less. The risk term is inherited unchanged (still sigmoid + BCE)."""

    # tells train() to build the model with the extra abstain outputs
    ABSTAIN = True

    def __init__(self, class_weight: torch.Tensor | None = None, o: float = 2.0,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        # class_weight is accepted only so train()'s call signature still fits;
        # the gambler term has no per-class weight (abstention, not reweighting,
        # is how this loss copes with imbalance), so we pass None to the base
        super().__init__(class_weight=None, lam_d=lam_d, lam_m=lam_m, lam_r=lam_r)
        self.o = float(o)                                   # payoff (>0)

    def defect_term(self, out: dict, y_defect: torch.Tensor) -> torch.Tensor:
        # gambler loss on the m+1-wide defect head
        return gambler_term(out["defect"], y_defect, self.o)

    def mech_term(self, logits: torch.Tensor, y_mech_s: torch.Tensor) -> torch.Tensor:
        # gambler loss on one stage's k+1-wide mechanism head
        return gambler_term(logits, y_mech_s, self.o)


# loss registry: add a subclass here and pick it with --loss
LOSSES = {"multihead": MultiHeadLoss, "abstention": AbstentionLoss}


def main() -> None:
    # our model defaults to the abstention loss; multihead is kept for comparison
    run_cli(MODEL_VERSION, LOSSES, default_loss="abstention",
            description="Multi-head MLP with abstention loss (SMT)")


if __name__ == "__main__":
    main()
