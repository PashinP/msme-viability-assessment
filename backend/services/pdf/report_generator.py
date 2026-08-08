"""
MSME Business Health Report — Reimagined PDF Generator
======================================================
Inspired by medical blood-test reports: simple visual indicators on page 1,
deep technical detail in the middle, a draft loan application, and a clear
executive summary at the end.

Built on ReportLab Platypus for professional text flow and layout.
"""
import io, uuid, math, tempfile, os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether,
    NextPageTemplate
)
from reportlab.platypus.flowables import Flowable


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR SYSTEM — Professional, high-contrast, readable
# ═══════════════════════════════════════════════════════════════════════════════

# Primary
PRIMARY    = colors.HexColor("#1B4F72")   # Deep corporate blue
PRIMARY_LT = colors.HexColor("#D6EAF8")   # Light blue tint
ACCENT     = colors.HexColor("#2E86C1")   # Accent blue

# Status colors (carefully chosen for readability on white)
S_GREEN    = colors.HexColor("#1E8449")
S_GREEN_BG = colors.HexColor("#EAFAF1")
S_YELLOW   = colors.HexColor("#B7950B")
S_YELLOW_BG= colors.HexColor("#FEF9E7")
S_ORANGE   = colors.HexColor("#CA6F1E")
S_ORANGE_BG= colors.HexColor("#FDF2E9")
S_RED      = colors.HexColor("#C0392B")
S_RED_BG   = colors.HexColor("#FDEDEC")

# Neutral grays
G50  = colors.HexColor("#FAFAFA")
G100 = colors.HexColor("#F5F5F5")
G200 = colors.HexColor("#EEEEEE")
G300 = colors.HexColor("#E0E0E0")
G400 = colors.HexColor("#9E9E9E")
G500 = colors.HexColor("#757575")
G600 = colors.HexColor("#616161")
G700 = colors.HexColor("#424242")
G800 = colors.HexColor("#212121")
WHITE = colors.white

# Maps
GRADE_COLORS = {"A": S_GREEN, "B": ACCENT, "C": S_YELLOW, "D": S_ORANGE, "F": S_RED}
GRADE_BG     = {"A": S_GREEN_BG, "B": PRIMARY_LT, "C": S_YELLOW_BG, "D": S_ORANGE_BG, "F": S_RED_BG}
STATUS_COLORS = {
    "strong": S_GREEN, "moderate": S_YELLOW,
    "needs_attention": S_ORANGE, "critical": S_RED, "unknown": G400,
}
STATUS_BG = {
    "strong": S_GREEN_BG, "moderate": S_YELLOW_BG,
    "needs_attention": S_ORANGE_BG, "critical": S_RED_BG, "unknown": G100,
}

USD_TO_INR = 83
GRADE_MAP = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}
W, H = A4
MARGIN = 2.0 * cm
DOC_W = W - 2 * MARGIN  # Usable content width


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def safe(t, n=None):
    if not isinstance(t, str): t = str(t or "")
    t = t.replace("\u20b9","Rs.").replace("\u2014","-").replace("\u2019","'")
    t = "".join(c if ord(c) < 256 else " " for c in t).strip()
    if n and len(t) > n: t = t[:n-1] + "..."
    return t or "N/A"

def fmt_inr(v):
    try:
        v = float(v or 0)
        if v >= 1e7: return f"Rs.{v/1e7:.2f} Cr"
        if v >= 1e5: return f"Rs.{v/1e5:.1f}L"
        if v >= 1e3: return f"Rs.{v/1e3:.0f}K"
        return f"Rs.{v:,.0f}"
    except: return "N/A"

def status_label(s):
    return s.replace("_"," ").title() if s else "Unknown"

def hex_of(c):
    """Get hex string like '#1E8449' from a reportlab color."""
    return "#" + c.hexval()[2:]


# ═══════════════════════════════════════════════════════════════════════════════
# PARAGRAPH STYLES (centralized, consistent)
# ═══════════════════════════════════════════════════════════════════════════════

def build_styles():
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=28,
                                textColor=PRIMARY, leading=34),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=14,
                                   textColor=G500, leading=18),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16,
                             textColor=G800, leading=20, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13,
                             textColor=PRIMARY, leading=17, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10,
                             textColor=G700, leading=14, spaceAfter=3),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                               textColor=G700, leading=14, spaceAfter=4,
                               alignment=TA_JUSTIFY),
        "body_sm": ParagraphStyle("body_sm", fontName="Helvetica", fontSize=8,
                                  textColor=G600, leading=12, spaceAfter=3),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.5,
                                textColor=G500, leading=10, spaceAfter=2),
        "italic": ParagraphStyle("italic", fontName="Helvetica-Oblique", fontSize=8,
                                 textColor=G500, leading=12),
        "label": ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=7,
                                textColor=G400, leading=9),
        "value": ParagraphStyle("value", fontName="Helvetica-Bold", fontSize=12,
                                textColor=G800, leading=15),
        "coach": ParagraphStyle("coach", fontName="Helvetica-Oblique", fontSize=8.5,
                                textColor=ACCENT, leading=12, leftIndent=8,
                                borderPadding=4),
    }

ST = build_styles()  # Module-level styles


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM FLOWABLES
# ═══════════════════════════════════════════════════════════════════════════════

class HorizontalBar(Flowable):
    """Rounded progress bar with score label."""
    def __init__(self, value, max_val=10, width=200, height=10,
                 fill_color=ACCENT, bg_color=G200, show_label=True):
        super().__init__()
        self.value = min(value or 0, max_val)
        self.max_val = max_val
        self.width = width
        self.height = height
        self.fill = fill_color
        self.bg = bg_color
        self.show_label = show_label

    def wrap(self, *args): return self.width + (30 if self.show_label else 0), self.height + 4
    def draw(self):
        c = self.canv
        r = self.height / 2
        # Background
        c.setFillColor(self.bg)
        c.roundRect(0, 2, self.width, self.height, r, fill=1, stroke=0)
        # Fill
        frac = min(self.value / self.max_val, 1.0)
        fill_w = max(r * 2, self.width * frac)
        c.setFillColor(self.fill)
        c.roundRect(0, 2, fill_w, self.height, r, fill=1, stroke=0)
        # Label
        if self.show_label:
            c.setFillColor(G700)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(self.width + 6, 3, f"{self.value}/{self.max_val}")


class SectionBanner(Flowable):
    """Clean section header with left accent and number."""
    def __init__(self, number, title, accent_color=PRIMARY, width=None):
        super().__init__()
        self.number = str(number)
        self.title = safe(title)
        self.accent = accent_color
        self.w = width or DOC_W
        self.h = 28

    def wrap(self, *args): return self.w, self.h
    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(G50)
        c.rect(0, 0, self.w, self.h, fill=1, stroke=0)
        # Left accent
        c.setFillColor(self.accent)
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        # Number circle
        c.setFillColor(self.accent)
        c.circle(20, self.h/2, 9, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(20, self.h/2 - 3, self.number)
        # Title
        c.setFillColor(G800)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(36, self.h/2 - 4, self.title)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS (Matplotlib → PNG → ReportLab Image)
# ═══════════════════════════════════════════════════════════════════════════════

TMPDIR = tempfile.mkdtemp(prefix="msme_report_")

def _save(fig, name):
    p = os.path.join(TMPDIR, f"{name}_{uuid.uuid4().hex[:6]}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(),
                edgecolor="none")
    plt.close(fig)
    return p


def chart_health_gauge(score, color_hex, size=100):
    """Large semicircular gauge for health card."""
    fig, ax = plt.subplots(figsize=(size/72, size/72))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.6, 1.4)

    theta_bg = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#E0E0E0", linewidth=16,
            solid_capstyle="round")
    frac = min(max(score / 100, 0), 1.0)
    theta_fill = np.linspace(np.pi, np.pi - frac * np.pi, 100)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color_hex, linewidth=16,
            solid_capstyle="round")
    ax.text(0, 0.18, str(score), ha="center", va="center", fontsize=30,
            fontweight="bold", color="#212121")
    ax.text(0, -0.18, "/100", ha="center", va="center", fontsize=11,
            color="#9E9E9E")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return _save(fig, "gauge")


def chart_shap_waterfall(contribs, w=380, h=160):
    """SHAP waterfall bar chart."""
    items = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    rename = {
        "DisbursementGross":"Loan Amount", "GrAppv":"Loan Approved",
        "SBA_Appv":"SBA Guarantee", "NoEmp":"Employees",
        "NewExist":"Biz Maturity", "UrbanRural":"Location",
        "RevLineCr":"Revolving Credit", "LowDoc":"Low Doc",
        "RetainedJob":"Jobs Retained", "CreateJob":"Jobs Planned", "Term":"Loan Term"
    }
    names = [rename.get(k, k) for k, _ in items]
    vals  = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(w/72, h/72))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bar_colors = ["#C0392B" if v > 0 else "#1E8449" for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=bar_colors, edgecolor="none", height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="#9E9E9E", linewidth=0.8)
    ax.set_xlabel("SHAP Value (red = risk, green = strength)", fontsize=8, color="#757575")
    for bar, v in zip(ax.patches, vals):
        x = v + (0.003 if v >= 0 else -0.003)
        ax.text(x, bar.get_y() + bar.get_height()/2, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7, color="#424242")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#E0E0E0")
    ax.tick_params(colors="#424242", labelsize=8)
    plt.tight_layout(pad=0.3)
    return _save(fig, "shap")


def chart_radar(features, size=190):
    """6-axis radar chart."""
    def compute(f):
        emp = f.get("NoEmp",0) + f.get("CreateJob",0) + f.get("RetainedJob",0)
        gr = max(f.get("GrAppv",1), 1)
        return {
            "Loan Term": min(100, (f.get("Term",84)/240)*100),
            "Employment": min(100, (emp/30)*100),
            "Maturity": 80 if f.get("NewExist",1)==1 else 30,
            "SBA Cover": min(100, (f.get("SBA_Appv",0)/gr)*120),
            "Location": {1:80, 0:50, 2:60}.get(f.get("UrbanRural",0), 50),
            "Documentation": max(0, 100 - (40 if f.get("LowDoc",0) else 0)),
        }
    scores = compute(features)
    cats = list(scores.keys())
    vals = list(scores.values())
    N = len(cats)
    angles = [n/N*2*math.pi for n in range(N)]
    vals_c = vals + [vals[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(size/72, size/72), subplot_kw={"projection":"polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(angles_c, vals_c, color="#2E86C1", linewidth=2)
    ax.fill(angles_c, vals_c, color="#2E86C1", alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25","50","75","100"], fontsize=6, color="#9E9E9E")
    ax.grid(color="#E0E0E0", linestyle="-", linewidth=0.5)
    ax.spines["polar"].set_color("#E0E0E0")
    plt.tight_layout(pad=0.2)
    return _save(fig, "radar")


def chart_probability(probs, w=340, h=120):
    """Horizontal bar chart of class probabilities."""
    labels = list(probs.keys())
    vals = [probs[k]*100 for k in labels]
    bar_c = ["#EF4444","#F97316","#F59E0B","#3B82F6","#10B981"]
    fig, ax = plt.subplots(figsize=(w/72, h/72))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(labels, vals, color=bar_c[:len(labels)], edgecolor="none", height=0.45)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2, f"{v:.1f}%",
                va="center", fontsize=8, fontweight="bold", color="#424242")
    ax.set_xlim(0, max(vals)*1.25 if vals else 100)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#757575")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#E0E0E0")
    ax.tick_params(labelsize=8, colors="#424242")
    plt.tight_layout(pad=0.3)
    return _save(fig, "probs")


def chart_finance_breakdown(rev, exp, emi, new_emi, w=360, h=90):
    """Stacked bar showing revenue allocation."""
    if rev <= 0: return None
    surplus = max(0, rev - exp - emi - new_emi)
    fig, ax = plt.subplots(figsize=(w/72, h/72))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cat = ["Revenue\nAllocation"]
    
    # Cleaner professional colors
    ax.barh(cat, [exp], color="#CBD5E1", edgecolor="none", height=0.4, label="Expenses")
    ax.barh(cat, [emi], left=[exp], color="#FBBF24", edgecolor="none", height=0.4, label="Existing EMI")
    ax.barh(cat, [new_emi], left=[exp+emi], color="#3B82F6", edgecolor="none", height=0.4, label="New EMI (est.)")
    ax.barh(cat, [surplus], left=[exp+emi+new_emi], color="#10B981", edgecolor="none", height=0.4, label="Surplus")
    
    # Legend on top, spread out
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.05), frameon=False, ncol=4)
    ax.set_xlim(0, rev * 1.05)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#E0E0E0")
    ax.tick_params(axis="x", colors="#9E9E9E", labelsize=8)
    plt.tight_layout(pad=0.2)
    return _save(fig, "finance")


def chart_peer_bars(success_rate, baseline, w=240, h=110):
    """Peer comparison bar chart."""
    fig, ax = plt.subplots(figsize=(w/72, h/72))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cats = ["Your Peers", "Market Avg"]
    vals = [success_rate*100, baseline*100]
    bars = ax.bar(cats, vals, color=["#2E86C1","#9E9E9E"], width=0.4, edgecolor="none")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#424242")
    ax.set_ylim(0, max(vals)*1.3)
    ax.set_ylabel("Success Rate (%)", fontsize=8, color="#757575")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#E0E0E0")
    ax.tick_params(colors="#424242", labelsize=9)
    plt.tight_layout(pad=0.3)
    return _save(fig, "peer")


def chart_pipeline(w=420, h=100):
    """3-step AI pipeline diagram."""
    fig, ax = plt.subplots(figsize=(w/72, h/72))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)

    # Three boxes
    boxes = [
        (0.3, "1. Data\nIngestion", "#D6EAF8", "#2E86C1"),
        (3.7, "2. ML\nScoring", "#D5F5E3", "#1E8449"),
        (7.1, "3. Insight\nGeneration", "#FADBD8", "#C0392B"),
    ]
    for x, text, bg, border in boxes:
        rect = plt.Rectangle((x, 0.4), 2.4, 1.7, facecolor=bg,
                             edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x+1.2, 1.25, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color="#212121", zorder=3)

    # Arrows between boxes
    for x in [2.7, 6.1]:
        ax.annotate("", xy=(x+0.6, 1.25), xytext=(x, 1.25),
                    arrowprops=dict(arrowstyle="->", color="#9E9E9E", lw=1.5))

    # Labels below
    labels = [
        (1.5, "NLP + Document\nParsing"),
        (4.9, "XGBoost +\nLightGBM"),
        (8.3, "SHAP +\nPrescriptions"),
    ]
    for x, txt in labels:
        ax.text(x, 0.1, txt, ha="center", va="center", fontsize=7,
                color="#757575", style="italic")

    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return _save(fig, "pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE STYLE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def clean_table_style(header_bg=PRIMARY):
    return TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), header_bg),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, G50]),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 8),
        ("TEXTCOLOR",     (0,1), (-1,-1), G700),
        ("GRID",          (0,0), (-1,-1), 0.5, G200),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── PAGE 1: COVER ─────────────────────────────────────────────────────────────

def page_cover(pred, assessment, features, context, report_id, now):
    s = ST
    label  = pred["predicted_label"]
    grade  = GRADE_MAP[pred["predicted_class"]]
    gc     = GRADE_COLORS[grade]
    gc_bg  = GRADE_BG[grade]
    biz    = safe(context.get("business_name", "Valued Client"), 40)
    industry = safe(context.get("industry_sector","N/A"), 25)

    story = [Spacer(1, 50)]

    # Brand line
    story.append(Paragraph(
        "<font color='#2E86C1'>MSME VIABILITY ASSESSMENT ENGINE</font>", s["label"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(color=PRIMARY, thickness=3, width="100%"))
    story.append(Spacer(1, 14))

    # Title
    story.append(Paragraph("Business Health Report", s["title"]))
    story.append(Paragraph("Loan Readiness Diagnostic", s["subtitle"]))
    story.append(Spacer(1, 30))

    # Prepared for
    prep = [
        ["PREPARED FOR", "INDUSTRY", "REPORT DATE"],
        [biz, industry, now],
    ]
    t = Table(prep, colWidths=[DOC_W*0.4, DOC_W*0.3, DOC_W*0.3])
    t.setStyle(TableStyle([
        ("TEXTCOLOR", (0,0),(-1,0), G400),
        ("TEXTCOLOR", (0,1),(-1,1), G800),
        ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",  (0,1),(-1,1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,0), 7),
        ("FONTSIZE",  (0,1),(-1,1), 10),
        ("TOPPADDING",(0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LINEBELOW", (0,1),(-1,1), 1, G200),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(t)
    story.append(Spacer(1, 30))

    # Executive summary
    summary = safe(assessment.get("summary",""), 500)
    story.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", s["h3"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(summary,
        ParagraphStyle("exec", fontName="Helvetica", fontSize=10,
                       textColor=G600, leading=16, alignment=TA_JUSTIFY)))
    story.append(Spacer(1, 24))

    # Grade badge + key facts
    gc_hex = hex_of(gc)
    grade_cell = Table([[
        [Paragraph(f"<font size='28' color='{gc_hex}'><b>{grade}</b></font>",
                   ParagraphStyle("gl", alignment=TA_CENTER, leading=34)),
         Paragraph(f"<b>{label}</b>",
                   ParagraphStyle("gn", fontName="Helvetica-Bold", fontSize=10,
                                  textColor=G800, alignment=TA_CENTER)),
         Paragraph(f"{pred['confidence']*100:.0f}% confidence",
                   ParagraphStyle("gc", fontName="Helvetica", fontSize=8,
                                  textColor=G500, alignment=TA_CENTER)),
        ]
    ]], colWidths=[100])
    grade_cell.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), gc_bg),
        ("BOX",(0,0),(-1,-1), 1.5, gc),
        ("TOPPADDING",(0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))

    loan_amt = fmt_inr(features.get("GrAppv",0)*USD_TO_INR)
    revenue  = fmt_inr(context.get("monthly_revenue",0))
    purpose  = safe(context.get("loan_purpose","N/A"), 30)
    years    = str(context.get("years_in_operation",0))

    facts = [["Loan Amount", loan_amt], ["Purpose", purpose],
             ["Monthly Revenue", revenue], ["Years Active", f"{years} years"]]
    facts_tbl = Table(facts, colWidths=[DOC_W*0.22, DOC_W*0.38])
    facts_tbl.setStyle(TableStyle([
        ("TEXTCOLOR",(0,0),(0,-1), G500),
        ("TEXTCOLOR",(1,0),(1,-1), G800),
        ("FONTNAME",(0,0),(0,-1),"Helvetica"),
        ("FONTNAME",(1,0),(1,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LINEBELOW",(0,0),(-1,-2), 0.5, G200),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))

    row = Table([[grade_cell, facts_tbl]], colWidths=[115, DOC_W-115])
    row.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(1,0),(1,0), 20),
    ]))
    story.append(row)

    # Footer
    story.append(Spacer(1, 60))
    story.append(HRFlowable(color=G200, thickness=1, width="100%"))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Report ID: {report_id}  |  Powered by XGBoost + SHAP  |  "
        f"Trained on 897,167 SBA Loans", ST["small"]))
    story.append(PageBreak())
    return story


# ── PAGE 2: HEALTH REPORT CARD (BLOOD TEST STYLE) ────────────────────────────

def page_health_card(assessment, pred, features):
    s = ST
    score = assessment.get("overall_score", 50)
    grade = GRADE_MAP[pred["predicted_class"]]
    gc    = GRADE_COLORS[grade]
    sections = assessment.get("sections", [])

    story = []
    story.append(SectionBanner("1", "Business Health Report Card"))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Like a medical blood test, this page gives you an at-a-glance view of your "
        "business's loan-readiness across six critical dimensions.",
        s["body"]))
    story.append(Spacer(1, 12))

    # Overall gauge
    gauge_path = chart_health_gauge(score, hex_of(gc), 130)
    gauge_row = Table([[
        Image(gauge_path, width=130, height=130),
        [Paragraph(f"<b>Overall Readiness Score</b>", s["h2"]),
         Spacer(1, 4),
         Paragraph(f"Grade <b>{grade}</b> - {pred['predicted_label']}",
                   ParagraphStyle("g_lbl", fontName="Helvetica", fontSize=11,
                                  textColor=gc)),
         Spacer(1, 4),
         Paragraph("This score is calculated from 11 business parameters analyzed "
                   "by our XGBoost machine learning model, calibrated on 897,167 "
                   "historical loan outcomes.", s["body_sm"])]
    ]], colWidths=[145, DOC_W-145])
    gauge_row.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(1,0),(1,0), 15),
    ]))
    story.append(gauge_row)
    story.append(Spacer(1, 16))

    # Section-by-section rows (blood test style)
    story.append(HRFlowable(color=PRIMARY, thickness=2, width="100%"))
    story.append(Spacer(1, 2))

    # Header row
    hdr = Table(
        [["DIMENSION", "SCORE", "STATUS", "WHAT THIS MEANS"]],
        colWidths=[DOC_W*0.22, DOC_W*0.18, DOC_W*0.14, DOC_W*0.46]
    )
    hdr.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0), 7),
        ("TEXTCOLOR",(0,0),(-1,0), G400),
        ("BOTTOMPADDING",(0,0),(-1,0), 4),
        ("TOPPADDING",(0,0),(-1,0), 4),
    ]))
    story.append(hdr)
    story.append(HRFlowable(color=G300, thickness=0.5, width="100%"))

    for sec in sections:
        st_val = sec.get("status","unknown")
        raw_score = sec.get("score")
        sc_v   = raw_score if raw_score is not None else 5
        sc     = STATUS_COLORS.get(st_val, G400)
        sc_bg  = STATUS_BG.get(st_val, G100)
        sc_hex = hex_of(sc)
        sec_name = safe(sec.get("section", "Section"), 30)
        bank_view = safe(sec.get("what_bank_sees",""), 120)

        row_data = [[
            Paragraph(f"<b>{sec_name}</b>", s["h3"]),
            HorizontalBar(sc_v, 10, width=DOC_W*0.12, height=8,
                          fill_color=sc, bg_color=G200, show_label=True),
            Paragraph(f"<font color='{sc_hex}'><b>{status_label(st_val)}</b></font>",
                      ParagraphStyle("st", fontName="Helvetica-Bold", fontSize=9,
                                     textColor=sc)),
            Paragraph(bank_view, s["body_sm"]),
        ]]
        row_tbl = Table(row_data,
                        colWidths=[DOC_W*0.22, DOC_W*0.18, DOC_W*0.14, DOC_W*0.46])
        row_tbl.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LINEBELOW",(0,0),(-1,-1), 0.5, G200),
            ("BACKGROUND",(0,0),(-1,-1), WHITE),
        ]))
        story.append(row_tbl)

    story.append(Spacer(1, 16))

    # Legend
    legend_items = [
        (S_GREEN, "Strong (7-10)"),
        (S_YELLOW, "Moderate (4-6)"),
        (S_ORANGE, "Needs Attention (2-3)"),
        (S_RED, "Critical (0-1)"),
    ]
    legend_cells = []
    for col, label in legend_items:
        legend_cells.append(
            Paragraph(f"<font color='{hex_of(col)}'>|</font> {label}", s["small"])
        )
    leg_tbl = Table([legend_cells], colWidths=[DOC_W/4]*4)
    leg_tbl.setStyle(TableStyle([
        ("TOPPADDING",(0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    story.append(leg_tbl)

    story.append(PageBreak())
    return story


# ── PAGES 3-4: FINANCIAL DEEP DIVE ───────────────────────────────────────────

def page_financial_deep_dive(pred, assessment, features, context):
    s = ST
    story = []
    story.append(SectionBanner("2", "Financial Deep Dive"))
    story.append(Spacer(1, 10))

    # Key metrics grid
    story.append(Paragraph("<b>Key Business Metrics</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 8))

    def metric(lbl, val, col=ACCENT):
        return [
            Paragraph(f"<font color='{hex_of(col)}' size='5'>|</font> "
                      f"<font size='7' color='#757575'>{safe(lbl,22)}</font>",
                      ParagraphStyle("ml", fontName="Helvetica", fontSize=7, leading=10)),
            Paragraph(f"<b>{safe(val,18)}</b>",
                      ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=13,
                                     textColor=G800, leading=16)),
        ]

    cw = (DOC_W - 6) / 4
    r1 = [
        metric("Monthly Revenue",  fmt_inr(context.get("monthly_revenue",0)), S_GREEN),
        metric("Monthly Expenses", fmt_inr(context.get("monthly_expenses",0)), S_RED),
        metric("Existing EMI",     fmt_inr(context.get("existing_debt_emi",0)), S_YELLOW),
        metric("Loan Requested",   fmt_inr(features.get("GrAppv",0)*USD_TO_INR), ACCENT),
    ]
    r2 = [
        metric("Collateral",       fmt_inr(context.get("collateral_value",0)), ACCENT),
        metric("Employees",        str(features.get("NoEmp",0)), ACCENT),
        metric("Years in Business", f"{context.get('years_in_operation',0)} yrs", S_GREEN),
        metric("ITR Filed",        f"{context.get('tax_filing_years',0)} yrs", G500),
    ]
    m_tbl = Table([r1, r2], colWidths=[cw]*4, rowHeights=[52, 52])
    m_tbl.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1), 0.5, G200),
        ("INNERGRID",(0,0),(-1,-1), 0.5, G200),
        ("BACKGROUND",(0,0),(-1,-1), WHITE),
        ("TOPPADDING",(0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",(0,0),(-1,-1), 10),
        ("RIGHTPADDING",(0,0),(-1,-1), 10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(m_tbl)
    story.append(Spacer(1, 12))

    # Compliance checklist
    story.append(Paragraph("<b>Compliance Checklist</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 6))

    checks = [
        ("GST Registered",     context.get("has_gst", False),    "Critical for bank eligibility"),
        ("Udyam Registration", context.get("has_udyam", False),  "Unlocks subsidies and lower rates"),
        ("ITR Filed",          context.get("tax_filing_years",0)>0, "Required by all banks"),
        ("Business > 1 Year",  context.get("years_in_operation",0)>1, "Reduces perceived risk"),
    ]
    comp_data = [["Item", "Status", "Why It Matters"]]
    for name, ok, reason in checks:
        comp_data.append([name, "YES" if ok else "NO", reason])
    ct = Table(comp_data, colWidths=[DOC_W*0.28, DOC_W*0.12, DOC_W*0.60])
    ts = clean_table_style()
    for i, (_, ok, _) in enumerate(checks, 1):
        col = S_GREEN if ok else S_RED
        ts.add("TEXTCOLOR", (1,i),(1,i), col)
        ts.add("FONTNAME",  (1,i),(1,i), "Helvetica-Bold")
    ct.setStyle(ts)
    story.append(ct)
    story.append(Spacer(1, 12))

    # Revenue breakdown chart
    rev = context.get("monthly_revenue", 0)
    exp = context.get("monthly_expenses", 0)
    emi = context.get("existing_debt_emi", 0)
    loan_usd = features.get("GrAppv",0) or features.get("DisbursementGross",0)
    term = features.get("Term", 84)
    new_emi = (loan_usd * USD_TO_INR / max(term, 1)) * 1.1

    if rev > 0:
        story.append(Paragraph("<b>Monthly Revenue Allocation</b>", s["h2"]))
        story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
        story.append(Spacer(1, 4))
        fc_path = chart_finance_breakdown(rev, exp, emi, new_emi, w=DOC_W*0.75, h=80)
        if fc_path:
            story.append(Image(fc_path, width=DOC_W*0.75, height=80, hAlign="LEFT"))
            story.append(Spacer(1, 6))
            dti = ((emi + new_emi) / max(rev - exp, 1)) * 100
            dti_hex = "#C0392B" if dti > 60 else ("#B7950B" if dti > 40 else "#1E8449")
            surplus = rev - exp - emi - new_emi
            story.append(Paragraph(
                f"<b>Debt-to-Income Ratio: <font color='{dti_hex}'>{min(dti,999):.1f}%</font></b>"
                f"  |  Estimated Monthly Surplus: {fmt_inr(max(0,surplus))}", s["body"]))

    # Probability distribution
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>AI Model Confidence Distribution</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 4))
    probs = pred.get("probabilities", {})
    if probs:
        pb_path = chart_probability(probs, w=DOC_W*0.7, h=120)
        story.append(Image(pb_path, width=DOC_W*0.7, height=120, hAlign="LEFT"))

    story.append(PageBreak())
    return story


# ── PAGE 5: DETAILED DIAGNOSIS ───────────────────────────────────────────────

def page_diagnosis(assessment):
    s = ST
    sections = assessment.get("sections", [])
    story = []
    story.append(SectionBanner("3", "Detailed Diagnosis"))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "A deep dive into each dimension. The 'Bank Perspective' shows how a loan "
        "officer would interpret this aspect of your application.", s["body"]))
    story.append(Spacer(1, 10))

    for sec in sections:
        st_val = sec.get("status","unknown")
        sc_v   = sec.get("score", 5)
        sc     = STATUS_COLORS.get(st_val, G400)
        sc_hex = hex_of(sc)
        sec_name = safe(sec["section"], 50)
        diag   = safe(sec.get("diagnosis",""), 800)
        bank   = safe(sec.get("what_bank_sees",""), 300)

        elems = []
        # Title + score
        title_row = Table([[
            Paragraph(f"<b>{sec_name}</b>", s["h2"]),
            Paragraph(f"<font color='{sc_hex}'><b>{status_label(st_val)} - {sc_v}/10</b></font>",
                      ParagraphStyle("sr", fontName="Helvetica-Bold", fontSize=11,
                                     textColor=sc, alignment=TA_RIGHT))
        ]], colWidths=[DOC_W*0.55, DOC_W*0.45-20])
        title_row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        elems.append(title_row)

        # Score bar
        elems.append(HorizontalBar(sc_v, 10, width=DOC_W*0.55, height=8,
                                   fill_color=sc, bg_color=G200, show_label=False))
        elems.append(Spacer(1, 8))

        # Bank perspective
        if bank and bank != "N/A":
            bp = Table([[Paragraph(f"<b>Bank Perspective:</b> {bank}", s["italic"])]],
                       colWidths=[DOC_W-28])
            bp.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1), G50),
                ("TOPPADDING",(0,0),(-1,-1), 6),
                ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                ("LEFTPADDING",(0,0),(-1,-1), 10),
                ("RIGHTPADDING",(0,0),(-1,-1), 10),
            ]))
            elems.append(bp)
            elems.append(Spacer(1, 6))

        # Diagnosis
        elems.append(Paragraph(diag, s["body"]))

        # Card wrapper
        card = Table([[elems]], colWidths=[DOC_W-20])
        card.setStyle(TableStyle([
            ("BOX",(0,0),(-1,-1), 0.5, G200),
            ("LINEBEFORE",(0,0),(0,-1), 3, sc),
            ("TOPPADDING",(0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",(0,0),(-1,-1), 14),
            ("RIGHTPADDING",(0,0),(-1,-1), 10),
            ("BACKGROUND",(0,0),(-1,-1), WHITE),
        ]))
        story.append(KeepTogether(card))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    return story


# ── PAGE 6: STRENGTHS VS WEAKNESSES ──────────────────────────────────────────

def page_strengths_weaknesses(shap_data, assessment):
    s = ST
    story = []
    story.append(SectionBanner("4", "What's Working & What's Not"))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "A clear breakdown of your strongest and weakest areas, derived from both "
        "the AI model's SHAP analysis and the diagnostic assessment.", s["body"]))
    story.append(Spacer(1, 12))

    strengths = []
    weaknesses = []

    # From assessment sections
    for sec in assessment.get("sections", []):
        name = safe(sec["section"], 30)
        score = sec.get("score", 5)
        status = sec.get("status", "unknown")
        bank = safe(sec.get("what_bank_sees",""), 80)
        if status in ("strong",):
            strengths.append((name, f"Score {score}/10", bank))
        elif status in ("critical", "needs_attention"):
            weaknesses.append((name, f"Score {score}/10", bank))

    # From SHAP
    if shap_data:
        rename = {
            "DisbursementGross":"Loan Amount", "GrAppv":"Loan Approved",
            "SBA_Appv":"SBA Guarantee", "NoEmp":"Employees",
            "NewExist":"Biz Maturity", "UrbanRural":"Location",
            "RevLineCr":"Revolving Credit", "LowDoc":"Low Doc",
            "RetainedJob":"Jobs Retained", "CreateJob":"Jobs Planned", "Term":"Loan Term"
        }
        for feat, val in sorted(shap_data.get("feature_contributions",{}).items(),
                                key=lambda x: x[1]):
            fname = rename.get(feat, feat)
            if val < -0.01 and len(strengths) < 6:
                strengths.append((fname, f"SHAP: {val:+.3f}", "Positively impacts approval"))
            elif val > 0.01 and len(weaknesses) < 6:
                weaknesses.append((fname, f"SHAP: {val:+.3f}", "Negatively impacts approval"))

    # Build two-column layout
    def build_col(items, header, col, bg):
        col_hex = hex_of(col)
        rows = [[Paragraph(f"<font color='{col_hex}'><b>{header}</b></font>",
                           ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=11,
                                          textColor=col))]]
        for name, score, detail in items[:6]:
            rows.append([
                [Paragraph(f"<font color='{col_hex}'>|</font> <b>{name}</b>",
                           ParagraphStyle("cn", fontName="Helvetica-Bold", fontSize=9,
                                          textColor=G800, leading=12)),
                 Paragraph(f"{score} - {detail}",
                           ParagraphStyle("cd", fontName="Helvetica", fontSize=8,
                                          textColor=G600, leading=11))]
            ])
        if not items:
            rows.append([Paragraph("No significant items identified.", s["italic"])])
        t = Table(rows, colWidths=[DOC_W*0.46])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), bg),
            ("TOPPADDING",(0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING",(0,0),(-1,-1), 10),
            ("RIGHTPADDING",(0,0),(-1,-1), 10),
            ("LINEBELOW",(0,0),(-1,-2), 0.5, G200),
        ]))
        return t

    str_tbl = build_col(strengths, "STRENGTHS", S_GREEN, S_GREEN_BG)
    wk_tbl  = build_col(weaknesses, "WEAKNESSES", S_RED, S_RED_BG)

    layout = Table([[str_tbl, wk_tbl]], colWidths=[DOC_W*0.48, DOC_W*0.48],
                   spaceBefore=0, spaceAfter=0)
    layout.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
        ("RIGHTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(layout)

    story.append(PageBreak())
    return story


# ── PAGE 7: MODEL TRANSPARENCY ───────────────────────────────────────────────

def page_model_transparency(shap_data, features, assessment):
    s = ST
    story = []
    story.append(SectionBanner("5", "How Our AI Model Works"))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Transparency builds trust. This section explains exactly how we arrived "
        "at your score, what data we used, and how each factor influenced the result.",
        s["body"]))
    story.append(Spacer(1, 10))

    # Pipeline diagram
    story.append(Paragraph("<b>AI Analysis Pipeline</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 6))
    pipe_path = chart_pipeline(w=DOC_W, h=100)
    story.append(Image(pipe_path, width=DOC_W, height=100))
    story.append(Spacer(1, 12))

    # SHAP chart
    story.append(Paragraph("<b>Feature Impact Analysis (SHAP)</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Each bar shows how a specific business parameter pushed the AI's prediction. "
        "<font color='#1E8449'>Green = strength</font>. "
        "<font color='#C0392B'>Red = risk factor</font>.", s["body"]))
    story.append(Spacer(1, 6))

    contribs = shap_data.get("feature_contributions", {}) if shap_data else {}
    if contribs:
        shap_path = chart_shap_waterfall(contribs, w=DOC_W, h=160)
        story.append(Image(shap_path, width=DOC_W, height=160))
        story.append(Spacer(1, 10))

        # Feature table
        rename = {
            "DisbursementGross":"Loan Amount", "GrAppv":"Loan Approved",
            "SBA_Appv":"SBA Guarantee", "NoEmp":"Employees",
            "NewExist":"Biz Maturity", "UrbanRural":"Location",
            "RevLineCr":"Revolving Credit", "LowDoc":"Low Doc",
            "RetainedJob":"Jobs Retained", "CreateJob":"Jobs Planned", "Term":"Loan Term"
        }
        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        ftbl_data = [["Feature", "SHAP Value", "Direction"]]
        for feat, val in sorted_c:
            ftbl_data.append([
                rename.get(feat, feat), f"{val:+.4f}",
                "Risk Factor" if val > 0 else "Strength"
            ])
        ftbl = Table(ftbl_data, colWidths=[DOC_W*0.4, DOC_W*0.25, DOC_W*0.35])
        ts = clean_table_style()
        for i, (_, val) in enumerate(sorted_c, 1):
            col = S_RED if val > 0 else S_GREEN
            ts.add("TEXTCOLOR", (1,i),(2,i), col)
            ts.add("FONTNAME",  (1,i),(2,i), "Helvetica-Bold")
        ftbl.setStyle(ts)
        story.append(ftbl)

    story.append(Spacer(1, 12))

    # Radar chart
    story.append(Paragraph("<b>Business Profile Radar</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 4))
    radar_path = chart_radar(features, 180)

    sections = assessment.get("sections", [])
    if sections:
        dim_rows = [["Dimension","Score","Status"]]
        for sec in sections:
            dim_rows.append([safe(sec["section"],30),
                             f"{sec.get('score',0)}/10",
                             status_label(sec.get("status","unknown"))])
        dim_tbl = Table(dim_rows, colWidths=[DOC_W*0.38, DOC_W*0.12, DOC_W*0.18])
        ts2 = clean_table_style()
        for i, sec in enumerate(sections, 1):
            sc = STATUS_COLORS.get(sec.get("status","unknown"), G400)
            ts2.add("TEXTCOLOR",(2,i),(2,i), sc)
            ts2.add("FONTNAME",(2,i),(2,i),"Helvetica-Bold")
        dim_tbl.setStyle(ts2)

        lay = Table([[Image(radar_path, width=175, height=175), dim_tbl]],
                    colWidths=[185, DOC_W-185])
        lay.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",(0,0),(-1,-1),0),
        ]))
        story.append(lay)
    else:
        story.append(Image(radar_path, width=180, height=180, hAlign="LEFT"))

    # Model credentials
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Model Credentials</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 4))
    cred_data = [
        ["Component","Detail"],
        ["Primary Model","XGBoost (Gradient Boosted Trees)"],
        ["Ensemble","LightGBM secondary model"],
        ["Training Data","897,167 US SBA loans (1962-2014)"],
        ["Features","11 core business/loan parameters"],
        ["Explainability","SHAP Tree Explainer"],
        ["Similarity","K-Nearest Neighbors (K=50)"],
    ]
    cred_tbl = Table(cred_data, colWidths=[DOC_W*0.35, DOC_W*0.65])
    cred_tbl.setStyle(clean_table_style())
    story.append(cred_tbl)

    story.append(PageBreak())
    return story


# ── PAGES 8-9: DRAFT LOAN APPLICATION ────────────────────────────────────────

def page_draft_application(features, context, assessment, pred, schemes):
    s = ST
    story = []
    story.append(SectionBanner("6", "Draft Loan Application"))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Take this to the bank.</b> Below is a pre-filled loan application template "
        "based on the data you've provided. Use it as a starting framework when "
        "approaching lenders. The coaching notes (in blue) suggest how to present "
        "each section for maximum impact.",
        s["body"]))
    story.append(Spacer(1, 12))

    biz_name  = safe(context.get("business_name", "[Your Business Name]"), 40)
    industry  = safe(context.get("industry_sector", "[Your Industry]"), 30)
    years     = context.get("years_in_operation", 0)
    employees = features.get("NoEmp", 0)
    new_jobs  = features.get("CreateJob", 0)
    loan_amt  = fmt_inr(features.get("GrAppv",0) * USD_TO_INR)
    purpose   = safe(context.get("loan_purpose", "[Loan Purpose]"), 40)
    term_m    = features.get("Term", 84)
    rev       = context.get("monthly_revenue", 0)
    exp       = context.get("monthly_expenses", 0)
    emi       = context.get("existing_debt_emi", 0)
    collateral= context.get("collateral_value", 0)
    has_gst   = context.get("has_gst", False)
    has_udyam = context.get("has_udyam", False)
    itr_years = context.get("tax_filing_years", 0)
    grade     = GRADE_MAP[pred["predicted_class"]]

    # Section A: Business Details
    story.append(Paragraph("<b>A. BUSINESS DETAILS</b>", s["h2"]))
    story.append(HRFlowable(color=PRIMARY, thickness=1, width="100%"))
    story.append(Spacer(1, 6))
    biz_data = [
        ["Business Name",          biz_name],
        ["Industry / Sector",      industry],
        ["Years in Operation",     f"{years} years"],
        ["Current Employees",      f"{employees} (Planning to hire {new_jobs} more)"],
        ["Business Type",          "Existing" if features.get("NewExist",1)==1 else "New Venture"],
        ["Location Type",          {1:"Urban",2:"Rural",0:"Not specified"}.get(features.get("UrbanRural",0),"N/A")],
    ]
    bt = Table(biz_data, colWidths=[DOC_W*0.35, DOC_W*0.65])
    bt.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("TEXTCOLOR",(0,0),(0,-1), G500),
        ("TEXTCOLOR",(1,0),(1,-1), G800),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LINEBELOW",(0,0),(-1,-1), 0.5, G200),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(bt)
    if years < 2:
        story.append(Paragraph(
            "Tip: Banks prefer 2+ years of operating history. Emphasize any prior "
            "industry experience of the founders to compensate.", s["coach"]))
    story.append(Spacer(1, 12))

    # Section B: Loan Request
    story.append(Paragraph("<b>B. LOAN REQUEST</b>", s["h2"]))
    story.append(HRFlowable(color=PRIMARY, thickness=1, width="100%"))
    story.append(Spacer(1, 6))
    new_emi_est = (features.get("GrAppv",0) * USD_TO_INR / max(term_m, 1)) * 1.1
    loan_data = [
        ["Amount Requested",       loan_amt],
        ["Purpose",                purpose],
        ["Requested Term",         f"{term_m} months ({term_m//12} years {term_m%12} months)"],
        ["Estimated Monthly EMI",  fmt_inr(new_emi_est)],
    ]
    lt = Table(loan_data, colWidths=[DOC_W*0.35, DOC_W*0.65])
    lt.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("TEXTCOLOR",(0,0),(0,-1), G500),
        ("TEXTCOLOR",(1,0),(1,-1), G800),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LINEBELOW",(0,0),(-1,-1), 0.5, G200),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(lt)
    story.append(Paragraph(
        "Tip: Present a clear utilization plan. Banks want to see exactly how "
        "the funds will be deployed (e.g., 40% equipment, 30% working capital, "
        "30% expansion).", s["coach"]))
    story.append(Spacer(1, 12))

    # Section C: Financial Summary
    story.append(Paragraph("<b>C. FINANCIAL SUMMARY</b>", s["h2"]))
    story.append(HRFlowable(color=PRIMARY, thickness=1, width="100%"))
    story.append(Spacer(1, 6))
    profit = rev - exp
    dti = ((emi + new_emi_est) / max(profit, 1)) * 100
    fin_data = [
        ["Monthly Revenue",      fmt_inr(rev)],
        ["Monthly Expenses",     fmt_inr(exp)],
        ["Monthly Profit",       fmt_inr(profit)],
        ["Existing EMI",         fmt_inr(emi)],
        ["New EMI (estimated)",  fmt_inr(new_emi_est)],
        ["Debt-to-Income Ratio", f"{min(dti,999):.1f}%"],
        ["Surplus After All EMIs", fmt_inr(max(0, profit - emi - new_emi_est))],
    ]
    ft = Table(fin_data, colWidths=[DOC_W*0.35, DOC_W*0.65])
    ft.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("TEXTCOLOR",(0,0),(0,-1), G500),
        ("TEXTCOLOR",(1,0),(1,-1), G800),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LINEBELOW",(0,0),(-1,-1), 0.5, G200),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(ft)
    if dti > 50:
        story.append(Paragraph(
            f"Tip: Your DTI of {dti:.0f}% is above the 50% comfort zone. Consider requesting "
            f"a longer term or smaller amount to bring this below 50%.", s["coach"]))
    story.append(Spacer(1, 12))

    # Section D: Collateral & Documentation
    story.append(Paragraph("<b>D. COLLATERAL & DOCUMENTATION</b>", s["h2"]))
    story.append(HRFlowable(color=PRIMARY, thickness=1, width="100%"))
    story.append(Spacer(1, 6))
    doc_data = [
        ["Collateral Offered",    fmt_inr(collateral) if collateral else "None"],
        ["GST Registration",      "Yes" if has_gst else "No"],
        ["Udyam Registration",    "Yes" if has_udyam else "No"],
        ["ITR Filed",             f"{itr_years} years"],
    ]
    dt = Table(doc_data, colWidths=[DOC_W*0.35, DOC_W*0.65])
    dt.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("TEXTCOLOR",(0,0),(0,-1), G500),
        ("TEXTCOLOR",(1,0),(1,-1), G800),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LINEBELOW",(0,0),(-1,-1), 0.5, G200),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(dt)

    if not has_udyam:
        story.append(Paragraph(
            "Tip: Get Udyam registration (free, online, takes 10 min). It unlocks "
            "CGTMSE collateral-free guarantees and priority sector lending rates.", s["coach"]))
    story.append(Spacer(1, 12))

    # Section E: Government Schemes You Qualify For
    if schemes and schemes.get("schemes"):
        story.append(Paragraph("<b>E. SUPPORTING GOVERNMENT SCHEMES</b>", s["h2"]))
        story.append(HRFlowable(color=PRIMARY, thickness=1, width="100%"))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Mention these to the bank. It shows preparation and unlocks special rates.",
            s["coach"]))
        story.append(Spacer(1, 6))
        for sc in schemes["schemes"][:4]:
            name = safe(sc.get("name",""), 50)
            desc = safe(sc.get("description",""), 200)
            benefits = sc.get("benefits", [])
            rel = sc.get("relevance","low")
            rel_col = S_GREEN if rel == "high" else S_YELLOW
            card_inner = [
                Paragraph(f"<b>{name}</b>", s["h3"]),
                Paragraph(desc, s["body_sm"]),
            ]
            if benefits:
                ben_text = " | ".join([f"+ {safe(b,40)}" for b in benefits[:3]])
                card_inner.append(Paragraph(ben_text,
                    ParagraphStyle("ben", fontName="Helvetica", fontSize=7.5,
                                   textColor=S_GREEN, leading=10)))
            card = Table([[card_inner]], colWidths=[DOC_W-20])
            card.setStyle(TableStyle([
                ("BOX",(0,0),(-1,-1), 0.5, G200),
                ("LINEBEFORE",(0,0),(0,-1), 3, rel_col),
                ("TOPPADDING",(0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("LEFTPADDING",(0,0),(-1,-1), 12),
                ("RIGHTPADDING",(0,0),(-1,-1), 10),
                ("BACKGROUND",(0,0),(-1,-1), WHITE),
            ]))
            story.append(KeepTogether(card))
            story.append(Spacer(1, 6))

    story.append(PageBreak())
    return story


# ── PAGE: ACTION PLAN ────────────────────────────────────────────────────────

def page_action_plan(prescriptions):
    s = ST
    story = []
    story.append(SectionBanner("7", "Action Plan"))
    story.append(Spacer(1, 8))

    if not prescriptions:
        story.append(Paragraph(
            "No critical actions required. Your business profile is already strong.",
            ParagraphStyle("ok", fontName="Helvetica-Bold", fontSize=12, textColor=S_GREEN)))
        story.append(PageBreak())
        return story

    story.append(Paragraph(
        "Prioritized, specific steps to strengthen your loan application. "
        "Address high-priority items first for maximum improvement.", s["body"]))
    story.append(Spacer(1, 10))

    PCOLS = {"high": S_RED, "medium": S_ORANGE, "low": S_GREEN}
    step = 1
    for rx in prescriptions:
        section_name = safe(rx.get("section",""), 40)
        story.append(Paragraph(f"<b>{section_name}</b>", s["h2"]))
        story.append(HRFlowable(color=ACCENT, thickness=1, width="100%"))
        story.append(Spacer(1, 5))

        for sug in rx.get("suggestions", []):
            priority = sug.get("priority","medium")
            pc = PCOLS.get(priority, G400)
            pc_hex = hex_of(pc)
            action = safe(sug.get("action",""), 80)
            detail = safe(sug.get("detail",""), 500)
            impact = safe(sug.get("impact",""), 120)
            difficulty = safe(sug.get("difficulty",""), 30)

            inner_rows = [
                [Paragraph(f"<b>{step}.</b>",
                    ParagraphStyle("sn", fontName="Helvetica-Bold", fontSize=14,
                                   textColor=ACCENT, leading=16)),
                 Paragraph(f"<b>{action}</b>", s["h3"]),
                 Paragraph(f"<font color='{pc_hex}'><b>{priority.title()}</b></font>",
                    ParagraphStyle("pr", fontName="Helvetica-Bold", fontSize=9,
                                   textColor=pc, alignment=TA_RIGHT))],
                [Spacer(1,1), Paragraph(detail, s["body"]), Spacer(1,1)],
            ]
            if impact and impact != "N/A":
                inner_rows.append([Spacer(1,1),
                    Paragraph(f"<b>Impact:</b> {impact}" +
                              (f"  |  <b>Effort:</b> {difficulty}" if difficulty and difficulty != "N/A" else ""),
                              s["coach"]),
                    Spacer(1,1)])

            inner_tbl = Table(inner_rows, colWidths=[22, DOC_W-22-80, 80])
            inner_tbl.setStyle(TableStyle([
                ("SPAN",(1,1),(2,1)),
                ("SPAN",(1,2),(2,2)) if len(inner_rows) > 2 else ("SPAN",(0,0),(0,0)),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("TOPPADDING",(0,0),(-1,-1), 3),
                ("BOTTOMPADDING",(0,0),(-1,-1), 3),
                ("LEFTPADDING",(0,0),(-1,-1), 4),
                ("RIGHTPADDING",(0,0),(-1,-1), 4),
            ]))

            card = Table([[inner_tbl]], colWidths=[DOC_W])
            card.setStyle(TableStyle([
                ("BOX",(0,0),(-1,-1), 0.5, G200),
                ("LINEBEFORE",(0,0),(0,-1), 3, pc),
                ("BACKGROUND",(0,0),(-1,-1), WHITE),
                ("TOPPADDING",(0,0),(-1,-1), 6),
                ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                ("LEFTPADDING",(0,0),(-1,-1), 0),
                ("RIGHTPADDING",(0,0),(-1,-1), 8),
            ]))
            story.append(KeepTogether(card))
            story.append(Spacer(1, 6))
            step += 1
        story.append(Spacer(1, 6))

    story.append(PageBreak())
    return story


# ── PAGE: MARKET CONTEXT ─────────────────────────────────────────────────────

def page_market_context(similar):
    s = ST
    story = []
    story.append(SectionBanner("8", "Market Context & Peer Comparison"))
    story.append(Spacer(1, 10))

    if not similar:
        story.append(Paragraph("Peer comparison data unavailable.", s["italic"]))
        story.append(PageBreak())
        return story

    sr   = similar.get("success_rate", 0)
    base = similar.get("baseline_success_rate", 0)
    total= similar.get("total_similar", 0)
    insight = safe(similar.get("insight",""), 300)
    risk = safe(similar.get("risk_vs_baseline",""), 40)

    story.append(Paragraph(insight, s["body"]))
    story.append(Spacer(1, 10))

    # Peer chart + stats table
    pc_path = chart_peer_bars(sr, base, w=230, h=110)
    stats = [
        ["Metric","Value"],
        ["Businesses Matched", f"{total:,}"],
        ["Peer Success Rate", f"{sr*100:.1f}%"],
        ["Market Baseline", f"{base*100:.1f}%"],
        ["Risk vs Baseline", risk],
    ]
    stats_tbl = Table(stats, colWidths=[DOC_W*0.3, DOC_W*0.25])
    stats_tbl.setStyle(clean_table_style())

    lay = Table([[Image(pc_path, width=220, height=105), stats_tbl]],
                colWidths=[230, DOC_W-230])
    lay.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(lay)

    # Similar businesses
    biz_list = similar.get("similar_businesses", [])
    if biz_list:
        story.append(Spacer(1, 14))
        story.append(Paragraph("<b>Closest Historical Matches</b>", s["h2"]))
        story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
        story.append(Spacer(1, 4))
        biz_data = [["#","Business","State","Outcome","Loan Amt","Match"]]
        for b in biz_list[:6]:
            ok = b.get("outcome","") in ["Paid in Full","P I F"]
            biz_data.append([
                str(b.get("rank","")), safe(b.get("name","N/A"),26),
                safe(b.get("state","")[:12]),
                "Repaid" if ok else "Default",
                fmt_inr(b.get("disbursement",0)),
                f"{b.get('similarity_score',0)*100:.0f}%"
            ])
        bt = Table(biz_data, colWidths=[18, DOC_W*0.3, 50, 50, DOC_W*0.18, 40])
        ts = clean_table_style()
        for i, b in enumerate(biz_list[:6], 1):
            ok = b.get("outcome","") in ["Paid in Full","P I F"]
            ts.add("TEXTCOLOR",(3,i),(3,i), S_GREEN if ok else S_RED)
            ts.add("FONTNAME",(3,i),(3,i),"Helvetica-Bold")
        bt.setStyle(ts)
        story.append(bt)

    story.append(PageBreak())
    return story


# ── FINAL PAGE: EXECUTIVE SUMMARY ────────────────────────────────────────────

def page_executive_summary(pred, assessment, features, context, report_id, now):
    s = ST
    label = pred["predicted_label"]
    grade = GRADE_MAP[pred["predicted_class"]]
    gc    = GRADE_COLORS[grade]
    score = assessment.get("overall_score", 50)

    story = []
    story.append(SectionBanner("9", "Summary & Next Steps"))
    story.append(Spacer(1, 12))

    # Grade box
    gc_hex = hex_of(gc)
    gc_bg  = GRADE_BG[grade]
    summary_box = Table([[
        [Paragraph(f"<font size='20' color='{gc_hex}'><b>{grade}</b></font>",
                   ParagraphStyle("sg", alignment=TA_CENTER, leading=26)),
         Paragraph(f"<b>Score: {score}/100</b>",
                   ParagraphStyle("ss", fontName="Helvetica-Bold", fontSize=10,
                                  textColor=G800, alignment=TA_CENTER))],
        [Paragraph(f"<b>{label}</b><br/>{pred['confidence']*100:.0f}% confidence",
                   ParagraphStyle("sl", fontName="Helvetica", fontSize=10,
                                  textColor=G700, leading=14))]
    ]], colWidths=[100, DOC_W-100])
    summary_box.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1), gc_bg),
        ("BOX",(0,0),(-1,-1), 1.5, gc),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING",(0,0),(-1,-1), 10),
        ("RIGHTPADDING",(0,0),(-1,-1), 10),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 16))

    # Key takeaways
    story.append(Paragraph("<b>KEY TAKEAWAYS</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 8))

    sections = assessment.get("sections", [])
    # Find strongest and weakest
    strong = [sec for sec in sections if sec.get("status") == "strong"]
    weak   = [sec for sec in sections if sec.get("status") in ("critical","needs_attention")]

    takeaways = []
    if strong:
        names = ", ".join([safe(s["section"],20) for s in strong[:3]])
        takeaways.append(f"<b>Your strongest areas are:</b> {names}. These will positively "
                        f"influence a bank's decision.")
    if weak:
        names = ", ".join([safe(s["section"],20) for s in weak[:3]])
        takeaways.append(f"<b>Areas needing immediate attention:</b> {names}. Addressing these "
                        f"before approaching a bank will significantly improve approval chances.")
    takeaways.append(f"<b>AI Confidence:</b> The model is {pred['confidence']*100:.0f}% confident "
                    f"in classifying your business as '{label}'.")
    if context.get("monthly_revenue", 0) > 0:
        rev = context["monthly_revenue"]
        exp = context.get("monthly_expenses", 0)
        profit = rev - exp
        takeaways.append(f"<b>Monthly Profit:</b> {fmt_inr(profit)} — this is the number "
                        f"banks care about most.")

    for t in takeaways:
        story.append(Paragraph(f"  {t}", s["body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 16))

    # Next steps checklist
    story.append(Paragraph("<b>RECOMMENDED NEXT STEPS</b>", s["h2"]))
    story.append(HRFlowable(color=G200, thickness=0.5, width="100%"))
    story.append(Spacer(1, 8))

    steps = [
        "Review the Health Report Card (Page 2) to understand your overall position",
        "Address the items in the Action Plan (Page 8), starting with high-priority fixes",
        "Use the Draft Loan Application (Page 7) as your template when approaching banks",
        "Check the Government Schemes section for subsidies you may qualify for",
        "Consult a chartered accountant to organize your financial documentation",
    ]
    for i, step in enumerate(steps, 1):
        step_tbl = Table([[
            Paragraph(f"<b>{i}</b>",
                ParagraphStyle("sn2", fontName="Helvetica-Bold", fontSize=11,
                               textColor=ACCENT, alignment=TA_CENTER)),
            Paragraph(step, s["body"])
        ]], colWidths=[25, DOC_W-25])
        step_tbl.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story.append(step_tbl)

    story.append(Spacer(1, 20))

    # Disclaimer box
    disc = (
        "This report is generated by an AI system for INFORMATIONAL PURPOSES ONLY. "
        "It does not constitute financial advice, a credit decision, or a loan guarantee. "
        "Predictions are based on statistical patterns from historical SBA loan data. "
        "Consult qualified financial advisors before making decisions."
    )
    disc_tbl = Table([[Paragraph(disc, s["body_sm"])]], colWidths=[DOC_W-20])
    disc_tbl.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1), 1, G300),
        ("BACKGROUND",(0,0),(-1,-1), G50),
        ("TOPPADDING",(0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",(0,0),(-1,-1), 12),
        ("RIGHTPADDING",(0,0),(-1,-1), 12),
    ]))
    story.append(disc_tbl)
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Report ID: {report_id}  |  Generated: {now}", ST["small"]))

    return story


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE TEMPLATE DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(WHITE)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, H-8, W, 8, fill=1, stroke=0)
    canvas.setFillColor(G200)
    canvas.rect(0, 0, W, 2, fill=1, stroke=0)
    canvas.restoreState()

def _draw_inner(canvas, doc):
    canvas.saveState()
    # Top accent
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, H-4, W, 4, fill=1, stroke=0)
    # Header bar
    canvas.setFillColor(G50)
    canvas.rect(0, H-34, W, 30, fill=1, stroke=0)
    canvas.setFillColor(G700)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(MARGIN, H-22, "MSME BUSINESS HEALTH REPORT")
    canvas.setFillColor(G400)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(W-MARGIN, H-22, f"Page {doc.page}")
    canvas.drawRightString(W-MARGIN, H-30, "CONFIDENTIAL")
    # Footer
    canvas.setFillColor(G200)
    canvas.rect(0, 16, W, 0.5, fill=1, stroke=0)
    canvas.setFillColor(G400)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawCentredString(W/2, 7,
        "For informational purposes only. Not financial advice. Generated by MSME AI Advisor.")
    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(features, context, pred, shap_data, similar, assessment,
                    prescriptions, optimizer, schemes) -> bytes:
    """Generate the full Business Health Report PDF. Returns PDF bytes."""
    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    buf = io.BytesIO()

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 26, bottomMargin=MARGIN + 10,
        title="MSME Business Health Report",
        author="MSME AI Advisor",
    )

    cover_frame = Frame(0, 0, W, H, leftPadding=MARGIN,
                        rightPadding=MARGIN, topPadding=20, bottomPadding=20)
    inner_frame = Frame(MARGIN, 22, W - 2*MARGIN, H - 22 - MARGIN - 34,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    cover_tpl = PageTemplate(id="cover", frames=[cover_frame], onPage=_draw_cover)
    inner_tpl = PageTemplate(id="inner", frames=[inner_frame], onPage=_draw_inner)
    doc.addPageTemplates([cover_tpl, inner_tpl])

    # Build the story
    story = []
    story += page_cover(pred, assessment, features, context, report_id, now)
    story.append(NextPageTemplate("inner"))
    story.append(Paragraph('<para/>', ParagraphStyle("sw", fontSize=1)))

    story += page_health_card(assessment, pred, features)
    story += page_financial_deep_dive(pred, assessment, features, context)
    story += page_diagnosis(assessment)
    story += page_strengths_weaknesses(shap_data, assessment)
    story += page_model_transparency(shap_data, features, assessment)
    story += page_draft_application(features, context, assessment, pred, schemes)
    story += page_action_plan(prescriptions)
    story += page_market_context(similar)
    story += page_executive_summary(pred, assessment, features, context, report_id, now)

    doc.build(story)
    buf.seek(0)
    return buf.read()
