from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


fig = plt.figure(figsize=(20, 12), dpi=220)
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")

fig.patch.set_facecolor("#f6f8fb")
ax.set_facecolor("#f6f8fb")

navy = "#1f3a5f"
gray = "#5c677d"
card_edge = "#d7deea"
shadow = "#dbe5f2"


def card(x, y, w, h, title, body, fc, ec=card_edge, tc=navy, bc=gray, title_fs=12.5, body_fs=9.2):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.08, y - 0.08),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=0,
            facecolor=shadow,
            alpha=0.55,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.36,
        title,
        ha="center",
        va="top",
        fontsize=title_fs,
        fontweight="bold",
        color=tc,
        zorder=3,
    )
    ax.text(
        x + 0.18,
        y + h - 0.72,
        body,
        ha="left",
        va="top",
        fontsize=body_fs,
        color=bc,
        zorder=3,
        linespacing=1.25,
    )


def arrow(x1, y1, x2, y2, color=gray, ms=18):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=1.7,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.0",
            zorder=4,
        )
    )


ax.text(
    10,
    11.35,
    "Group 5 Capstone Responsible AI Pipeline",
    ha="center",
    va="center",
    fontsize=24,
    fontweight="bold",
    color=navy,
)
ax.text(
    10,
    10.95,
    "Data flow from HMDA input to governance-ready artifacts",
    ha="center",
    va="center",
    fontsize=12.5,
    color=gray,
)

card(0.55, 8.95, 3.15, 1.25, "1. Input Data", "Cleaned HMDA dataset\nsource: data/processed\nraw target: action_taken", "#eaf2ff", ec="#9fb9f2")
card(4.15, 8.95, 3.15, 1.25, "2. Target Mapping", "1, 2 -> positive class (1)\n3 -> negative class (0)\ninvalid outcomes dropped", "#edf9f0", ec="#9ad4a8")
card(7.75, 8.95, 3.15, 1.25, "3. Mode Control", "use_full_dataset=True -> full run\nelse stratified sample if rows > max_rows", "#fff7e8", ec="#f0c36b")
card(11.35, 8.95, 3.15, 1.25, "4. Modeling Frame", "drop protected + configured columns\nremove constant / high-cardinality features", "#f1f4ff", ec="#aab8ff")
card(14.95, 8.95, 4.45, 1.25, "5. Train / Val / Test Split", "stratified 60 / 20 / 20 split\nconsistent random_state for reproducibility", "#eefbf7", ec="#9adbc5")

card(0.55, 6.6, 3.3, 1.35, "6. Preprocessing", "numeric: median impute + scale\ncategorical: most frequent + one-hot", "#eef4ff", ec="#9fb9f2")
card(4.15, 6.6, 3.3, 1.35, "7. Candidate Models", "Logistic Regression\nRandom Forest", "#f8f0ff", ec="#cfb4ff")
card(7.75, 6.6, 3.3, 1.35, "8. Threshold Tuning", "select validation threshold via Youden J\nthen predict on train / val / test", "#fff2f0", ec="#f1b2a6")
card(11.35, 6.6, 3.3, 1.35, "9. Performance Metrics", "AUC, accuracy, log loss, F1\nTPR / TNR / FPR / FNR / positive rate", "#f2fbff", ec="#9fd0e6")
card(14.95, 6.6, 4.45, 1.35, "10. Transparency Outputs", "logistic coefficients\npermutation importance", "#f4f9ee", ec="#b6d79b")

card(0.55, 4.0, 4.1, 1.38, "11. Fairness Audit", "group AIR, mean effect, SMD\nFPR / FNR by protected attribute\nintersectional race x sex analysis", "#fef6ef", ec="#efc08a")
card(5.0, 4.0, 4.1, 1.38, "12. Robustness Checks", "PSI + KS drift report\nGaussian perturbation stress test\nrecord AUC drop", "#f0fbff", ec="#8fd3e8")
card(9.45, 4.0, 4.1, 1.38, "13. Model Selection", "composite score = test AUC - overfit penalty - fairness penalty\nchoose best model", "#f2f2ff", ec="#b6b8ff")
card(13.9, 4.0, 5.5, 1.38, "14. Exported Artifacts", "model.joblib, metadata.json\nperformance, fairness, drift tables\nROC + feature importance figures\ngovernance memo", "#eef9f2", ec="#97d2a2")

arrow(3.7, 9.58, 4.15, 9.58, color=gray)
arrow(7.3, 9.58, 7.75, 9.58, color=gray)
arrow(10.9, 9.58, 11.35, 9.58, color=gray)
arrow(14.5, 9.58, 14.95, 9.58, color=gray)

arrow(2.2, 6.6, 2.6, 5.38, color="#e09f3e")
arrow(6.65, 6.6, 6.95, 5.38, color="#159a9c")
arrow(10.4, 6.6, 11.4, 5.38, color="#2f6fed")
arrow(16.0, 6.6, 16.65, 5.38, color="#2f9e44")

ax.text(4.04, 8.45, "clean", fontsize=9, color=gray, ha="center")
ax.text(7.95, 8.45, "control", fontsize=9, color=gray, ha="center")
ax.text(11.48, 8.45, "prepare", fontsize=9, color=gray, ha="center")
ax.text(15.05, 8.45, "split", fontsize=9, color=gray, ha="center")

ax.text(10, 0.9, "Static architecture export for slides and reports", ha="center", va="center", fontsize=10.5, color=gray)
ax.text(10, 0.58, "Generated from the same pipeline logic documented in TECHNICAL_PIPELINE_DOCUMENTATION.md", ha="center", va="center", fontsize=8.8, color="#73829b")

png_path = OUT_DIR / "architecture_pipeline.png"
svg_path = OUT_DIR / "architecture_pipeline.svg"
fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
fig.savefig(svg_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
plt.close(fig)
print(png_path)
print(svg_path)
