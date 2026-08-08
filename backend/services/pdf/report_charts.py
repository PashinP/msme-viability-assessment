"""
Chart generators for the PDF report.
All functions return a file path to a temporary PNG image.
"""
import os, tempfile, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TMPDIR = tempfile.mkdtemp(prefix="msme_report_")
LABEL_NAMES = {0: "Critical", 1: "At-Risk", 2: "Stable", 3: "Growing", 4: "Thriving"}
CLASS_COLORS = ["#d32f2f", "#f57c00", "#388e3c", "#1976d2", "#7b1fa2"]
USD_TO_INR = 83


def _save(fig, name):
    path = os.path.join(TMPDIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def probability_chart(probabilities: dict) -> str:
    """Bar chart of class probabilities."""
    labels = list(probabilities.keys())
    vals = list(probabilities.values())
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [CLASS_COLORS[i] for i in range(len(labels))]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Probability (%)", fontsize=10)
    ax.set_title("Prediction Probability Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(v * 100 for v in vals) + 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "prob_dist")


def radar_chart(scores: dict) -> str:
    """6-axis radar chart for readiness dimensions."""
    categories = list(scores.keys())
    values = list(scores.values()) + [list(scores.values())[0]]
    N = len(categories)
    angles = [n / N * 2 * math.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.plot(angles, values, "o-", linewidth=2.5, color="#1976d2")
    ax.fill(angles, values, alpha=0.2, color="#1976d2")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, fontweight="500")
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7, color="#666")
    ax.set_ylim(0, 100)
    ax.grid(color="#ccc", linewidth=0.5)
    ax.set_title("Loan Readiness Dimensions", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save(fig, "radar")


def shap_chart(contributions: dict, label: str) -> str:
    """Horizontal bar chart of SHAP feature contributions."""
    FEAT_LABELS = {
        "Term": "Loan Term", "NoEmp": "Employees", "NewExist": "Business Type",
        "CreateJob": "Jobs Created", "RetainedJob": "Jobs Retained",
        "DisbursementGross": "Loan Amount", "UrbanRural": "Location",
        "RevLineCr": "Revolving Credit", "LowDoc": "Low Documentation",
        "SBA_Appv": "SBA Guarantee", "GrAppv": "Gross Approved",
    }
    sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [FEAT_LABELS.get(f, f) for f, _ in sorted_feats]
    vals = [v for _, v in sorted_feats]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ["#ef5350" if v > 0 else "#43a047" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.6, edgecolor="white")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Impact on Prediction (SHAP Value)", fontsize=10)
    ax.set_title(f"Feature Contributions → {label}", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "shap")


def term_sensitivity_chart(term_analysis: dict) -> str:
    """Line chart: predicted class vs loan term."""
    data = term_analysis.get("all_terms", [])
    if not data:
        return None
    terms = [d["term_months"] for d in data]
    classes = [d["predicted_class"] for d in data]
    confs = [d["confidence"] * 100 for d in data]
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.plot(terms, classes, "o-", color="#1976d2", linewidth=2, markersize=6, label="Predicted Class")
    ax1.set_xlabel("Loan Term (months)", fontsize=10)
    ax1.set_ylabel("Predicted Class (0-4)", fontsize=10, color="#1976d2")
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(["Critical", "At-Risk", "Stable", "Growing", "Thriving"], fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(terms, confs, "s--", color="#f57c00", linewidth=1.5, markersize=4, alpha=0.7, label="Confidence %")
    ax2.set_ylabel("Confidence (%)", fontsize=10, color="#f57c00")
    ax1.set_title("Term Sensitivity Analysis", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "term_sens")


def amount_sensitivity_chart(amount_analysis: dict) -> str:
    """Line chart: predicted class vs loan amount."""
    data = amount_analysis.get("amount_analysis", [])
    if not data:
        return None
    amounts = [d["amount"] * USD_TO_INR / 100000 for d in data]  # in lakhs
    classes = [d["predicted_class"] for d in data]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ["#43a047" if d["is_safe"] else "#ef5350" for d in data]
    ax.bar(range(len(amounts)), classes, color=colors, width=0.6)
    ax.set_xticks(range(len(amounts)))
    ax.set_xticklabels([f"₹{a:.1f}L" for a in amounts], fontsize=8, rotation=45)
    ax.set_ylabel("Predicted Class", fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["Critical", "At-Risk", "Stable", "Growing", "Thriving"], fontsize=8)
    ax.set_title("Amount Sensitivity (Green = Safe, Red = Risky)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "amt_sens")


def peer_comparison_chart(success_rate: float, baseline_rate: float) -> str:
    """Bar chart comparing user's peer success rate vs baseline."""
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(
        ["Your Peer Group", "Overall Baseline"],
        [success_rate * 100, baseline_rate * 100],
        color=["#1976d2", "#9e9e9e"], width=0.4, edgecolor="white"
    )
    for bar, v in zip(bars, [success_rate * 100, baseline_rate * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Repayment Success Rate (%)", fontsize=10)
    ax.set_title("Peer Comparison: Your Profile vs Baseline", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save(fig, "peer_comp")
