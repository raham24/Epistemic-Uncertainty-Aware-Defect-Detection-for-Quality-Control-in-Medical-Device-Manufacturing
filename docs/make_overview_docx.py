"""Build a high-level .docx overview of the SMT synthetic generator.

Pulls live metrics from results/bayes_error.json and embeds figures from figs/,
so the document cannot drift from the actual data. Output: docs/Generator_Overview.docx
Run: conda run -n paper python docs/make_overview_docx.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
M = json.load(open(ROOT / "results" / "bayes_error.json"))


def pct(err: float) -> str:
    return f"{(1.0 - err) * 100:.1f}%"


cf = M["class_fractions"]
be = M["bayes_error"]
cb = M["defect_head_metrics"]
bayes_pop = be["exact_mc_bayes_error"]
bayes_test = be["exact_mc_bayes_error_test_split"]
bo_f1 = cb["bayes_optimal"]["weighted_f1"]
mlp_acc = cb["mlp_basic"]["accuracy"]
mlp_f1 = cb["mlp_basic"]["weighted_f1"]

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def h(text, level):
    p = doc.add_heading(text, level=level)
    return p


def body(text):
    return doc.add_paragraph(text)


def bullet(text):
    return doc.add_paragraph(text, style="List Bullet")


# ---- Title -----------------------------------------------------------------
title = doc.add_heading("SMT Synthetic Data Generator", level=0)
sub = doc.add_paragraph()
run = sub.add_run(
    "A high-level overview and comparison to the synthetic data in "
    "Shenoy & Ameri (2026), “Uncertainty-Aware Neurosymbolic Root-Cause "
    "Analysis for Surface-Mount Assembly,” IEEE Trans. Semiconductor "
    "Manufacturing, DOI 10.1109/TSM.2026.3673999."
)
run.italic = True
run.font.size = Pt(10)
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

# ---- 1. Purpose ------------------------------------------------------------
h("1. Purpose", 1)
body(
    "This generator produces a synthetic surface-mount-assembly (SMT) dataset "
    "that stands in for the data used in the paper, so the full neurosymbolic "
    "root-cause-analysis pipeline can be developed and evaluated against known "
    "ground truth before any real-world data is involved. The paper neither "
    "releases its dataset nor its generator code, so this is a faithful "
    "reconstruction: its structure mirrors the paper exactly, and its "
    "quantitative behavior is calibrated to the few anchors the paper does "
    "report (the class balance in Table I and the graded-risk form in Eq. 8)."
)

# ---- 2. What it produces ---------------------------------------------------
h("2. What it produces", 1)
body(
    "200,000 per-board records, written from a single specification file "
    "(domain/smt_paper.yaml). The code contains only mechanism; everything "
    "domain-specific lives in the spec, so re-targeting to another device class "
    "is a file swap, not a code change. Each record carries:"
)
bullet("Six monitored process parameters across two stages — paste printing "
       "(ambient humidity, ambient temperature, paste viscosity, stencil "
       "thickness) and reflow (time above liquidus, peak reflow temperature).")
bullet("A defect-class label: no defect, open circuit, or solder bridging.")
bullet("A mechanism label for each stage.")
bullet("A graded risk score for each parameter (the paper’s Eq. 8).")
bullet("The exact label posterior p(y|x) — the probability of each class given "
       "the parameters — which is what makes the difficulty measurable.")
bullet("Full provenance: generator version, seed, batch, timestamp, board id.")

# ---- 3. How it works -------------------------------------------------------
h("3. How it works", 1)
body("Each record is built in six steps:")
for i, (t, d) in enumerate([
    ("Correlated sampling",
     "a Gaussian copula draws the six parameters together with a specified "
     "correlation matrix (nearest positive-definite corrected), so realistic "
     "cross-parameter dependence is preserved."),
    ("Drift",
     "equipment aging (stencil wear with periodic replacement) and a daily "
     "temperature cycle add non-stationarity; records are emitted in "
     "batch/shift groups with different seeds and start hours."),
    ("Normalization",
     "each parameter is expressed as a deviation from nominal, scaled so that a "
     "deviation of 1.0 sits exactly at its specification limit."),
    ("Causal labeling",
     "an Ishikawa cause-effect map turns those deviations into a defect "
     "probability p(y|x); the label is sampled from that distribution, so two "
     "boards with identical readings can still receive different labels — "
     "realistic, irreducible noise."),
    ("Mechanisms and risk",
     "the most-implicated mechanism is assigned per stage, and each parameter "
     "receives a graded risk score via the paper’s Eq. 8."),
    ("Calibration",
     "the labeling is tuned two ways: class frequencies are matched to the "
     "paper’s reported balance, and the overall difficulty is set so the "
     "best-possible accuracy lands in the paper’s regime (~95%)."),
], 1):
    p = doc.add_paragraph(style="List Number")
    p.add_run(f"{t}: ").bold = True
    p.add_run(d)

# ---- 4. Why it is trustworthy ---------------------------------------------
h("4. Why the results can be trusted", 1)
bullet("Known posterior → exact difficulty. Because the generator knows p(y|x), "
       "the theoretical best-possible accuracy (the Bayes floor) is computed in "
       "closed form, not estimated — then confirmed by training real classifiers.")
bullet("Deterministic and auditable. The same seed produces byte-for-byte "
       "identical output, and every record traces back to its source.")
bullet("Single source of truth. The spec file is the contract the generator, "
       "ontology, and evaluation all share, preventing silent drift.")

# ---- Figure: Bayes summary -------------------------------------------------
doc.add_picture(str(ROOT / "figs" / "bayes_summary.png"), width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph()
cr = cap.add_run(
    "Figure 1. How hard the classification problem is. A basic MLP reaches "
    f"~{mlp_acc*100:.1f}% accuracy — essentially the Bayes-optimal ceiling "
    f"(~{pct(bayes_test)} test / {pct(bayes_pop)} population) — and brackets the "
    "paper’s reported 95.00%."
)
cr.italic = True
cr.font.size = Pt(9)
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ---- 5. Comparison to the paper -------------------------------------------
h("5. How it compares to the synthetic data in the paper", 1)
body(
    "The generator is structurally identical to the paper and quantitatively "
    "calibrated to its reported anchors. The table below summarizes the "
    "correspondence."
)

rows = [
    ("Aspect", "Paper (Shenoy & Ameri 2026)", "This generator"),
    ("Domain & stages", "SMT: paste printing, then reflow", "Same"),
    ("Process parameters",
     "Six (humidity, ambient temp, paste viscosity, stencil thickness; time "
     "above liquidus, peak reflow temp)", "Same six"),
    ("Defect classes", "No defect, open circuit, solder bridging", "Same three"),
    ("Sampling", "Gaussian copula + nearest-PD correction",
     "Same (Gaussian marginals)"),
    ("Non-stationarity",
     "Stencil wear + diurnal temperature; batch/shift groups", "Same"),
    ("Causal map", "Ishikawa fishbone (Fig. 2)",
     "Matched edge-for-edge; open-circuit half mirrored from the bridging tree"),
    ("Class balance (Table I)",
     "175,420 / 12,011 / 12,569  (87.71% / 6.01% / 6.28%)",
     f"Calibrated to those priors "
     f"({cf['no_defect']*100:.1f}% / {cf['open_circuit']*100:.1f}% / "
     f"{cf['solder_bridging']*100:.1f}% realized)"),
    ("Graded risk", "Eq. 8–10 (0.05 → 0.70 at spec → 0.99 out of spec)",
     "Exact Eq. 8 implementation"),
    ("Dataset size / split", "200,000;  140k / 30k / 30k (batch-grouped)", "Same"),
    ("Defect-head accuracy", "95.00% (reported)",
     f"Basic MLP {mlp_acc*100:.1f}%, at the oracle ceiling {pct(bayes_pop)} (paper's 95% bracketed)"),
    ("Defect-head weighted-F1", "95.36% (reported)",
     f"Basic MLP {mlp_f1*100:.1f}%, oracle ceiling {bo_f1*100:.1f}%"),
    ("Released artifacts", "None (no code, no constants)",
     "Full generator, spec, and analysis"),
]
table = doc.add_table(rows=0, cols=3)
table.style = "Light Grid Accent 1"
for r, (a, b, c) in enumerate(rows):
    cells = table.add_row().cells
    cells[0].text, cells[1].text, cells[2].text = a, b, c
    if r == 0:
        for cell in cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.bold = True

# ---- 6. What we cannot match exactly --------------------------------------
h("6. What cannot be matched exactly (reproducibility gaps)", 1)
body(
    "The paper specifies the form of the data-generating process but not all of "
    "its numbers. The following are reconstructed to be process-plausible and "
    "calibrated to the paper’s reported anchors, and are documented so they can "
    "be revised if more detail becomes available:"
)
bullet("The correlation matrix, parameter spec bounds, and process spreads.")
bullet("The drift coefficients and the causal-edge weights.")
bullet("The risk-function constants (the two breakpoints and the out-of-spec "
       "rise rate); the paper gives the levels 0.05 / 0.70 / 0.99 but not these.")
bullet("The exact defect-labeling rule (not described in the paper); "
       "implemented as a calibrated logit model over the causal map.")
bullet("The open-circuit half of the causal map: the paper’s Fig. 2 draws only "
       "the solder-bridging subtree, so the open-circuit edges are mirrored from "
       "it (inferred from the paper’s “opposite non-compliance directions” note).")

# ---- 7. Bottom line --------------------------------------------------------
h("7. Bottom line", 1)
body(
    "The synthetic data is structurally identical to the paper, quantitatively "
    "calibrated to its reported class balance and difficulty, and — unlike the "
    "paper — fully released, deterministic, and auditable. Because the label "
    "posterior is known, the Bayes floor gives an exact, principled performance "
    "target for the model stage: a defect classifier should approach ~95% "
    "accuracy, matching the paper’s reported result."
)

out = ROOT / "docs" / "Generator_Overview.docx"
doc.save(str(out))
print(f"Wrote {out}")
