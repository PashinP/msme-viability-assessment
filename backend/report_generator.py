"""
PDF Report Generator for MSME Loan Readiness Assessment.
Generates a detailed 10+ page report using fpdf2 + matplotlib charts.
"""
import os, io, math, re, tempfile, uuid
from datetime import datetime
from copy import deepcopy
from fpdf import FPDF
from backend.report_charts import (
    probability_chart, radar_chart, shap_chart,
    term_sensitivity_chart, amount_sensitivity_chart, peer_comparison_chart
)

USD_TO_INR = 83
LABEL_NAMES = {0: "Critical", 1: "At-Risk", 2: "Stable", 3: "Growing", 4: "Thriving"}
GRADE_MAP = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}
GRADE_COLORS_RGB = {"A": (76,175,80), "B": (33,150,243), "C": (255,152,0), "D": (244,67,54), "F": (183,28,28)}
FEAT_LABELS = {
    "Term": "Loan Term (months)", "NoEmp": "Number of Employees",
    "NewExist": "Business Type", "CreateJob": "Jobs to Create",
    "RetainedJob": "Jobs Retained", "DisbursementGross": "Loan Amount (USD)",
    "UrbanRural": "Location Type", "RevLineCr": "Revolving Credit",
    "LowDoc": "Low Documentation", "SBA_Appv": "SBA Guarantee (USD)",
    "GrAppv": "Gross Approved (USD)",
}


def strip_emoji(text):
    """Remove emoji and other non-latin1 chars for PDF compatibility."""
    if not isinstance(text, str): return str(text)
    # Replace rupee sign with Rs.
    text = text.replace('\u20b9', 'Rs.')
    # Remove all non-latin1 characters (emojis etc)
    cleaned = ''.join(c for c in text if ord(c) < 256)
    return cleaned.strip() or 'N/A'


def fmt_inr(usd):
    inr = usd * USD_TO_INR
    if inr >= 1e7: return f"Rs.{inr/1e7:.1f} Cr"
    if inr >= 1e5: return f"Rs.{inr/1e5:.1f}L"
    if inr >= 1e3: return f"Rs.{inr/1e3:.0f}K"
    return f"Rs.{inr:,.0f}"


def fmt_feat_val(key, val):
    if key in ("DisbursementGross", "SBA_Appv", "GrAppv"):
        return f"${val:,.0f} ({fmt_inr(val)})"
    if key == "NewExist": return "Existing" if val == 1 else "New"
    if key == "UrbanRural": return {0: "Undefined", 1: "Urban", 2: "Rural"}.get(int(val), str(val))
    if key == "RevLineCr": return "Yes" if val else "No"
    if key == "LowDoc": return "Yes" if val else "No"
    return str(val)


class MSMEReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 8, "MSME Loan Readiness Report | Confidential", align="L")
            self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 16, 200, 16)
            self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(25, 118, 210)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 118, 210)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(6)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, strip_emoji(text))
        self.ln(2)

    def add_kv_table(self, data, col1_w=80, col2_w=100):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(25, 118, 210)
        self.set_text_color(255, 255, 255)
        self.cell(col1_w, 8, "Parameter", border=1, fill=True)
        self.cell(col2_w, 8, "Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        alt = False
        for k, v in data:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(40, 40, 40)
            if alt: self.set_fill_color(240, 245, 250)
            else: self.set_fill_color(255, 255, 255)
            self.cell(col1_w, 7, str(k), border=1, fill=True)
            self.cell(col2_w, 7, str(v), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            alt = not alt
        self.ln(4)


def generate_report(features, pred, shap_data, similar, red_flags, optimizer, schemes) -> bytes:
    """Generate the full PDF report. Returns PDF bytes."""
    pdf = MSMEReport()
    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    label = pred["predicted_label"]
    grade = GRADE_MAP[pred["predicted_class"]]
    conf = pred["confidence"]
    gc = GRADE_COLORS_RGB[grade]

    # ═══ PAGE 1: COVER ═══
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(25, 25, 60)
    pdf.cell(0, 15, "MSME Loan Readiness", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, "Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(25, 118, 210)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)
    # Grade badge
    pdf.set_fill_color(*gc)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 48)
    x = 80
    pdf.rect(x, pdf.get_y(), 50, 50, style="F")
    pdf.set_xy(x, pdf.get_y() + 5)
    pdf.cell(50, 30, grade, align="C")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(50, 12, label, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_text_color(80, 80, 80)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Confidence: {conf*100:.1f}%  |  Loan: {fmt_inr(features.get('DisbursementGross',0))}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Report ID: {report_id}  |  Generated: {now}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Model: XGBoost (897,167 historical SBA loans)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "This report is generated by the MSME Loan Readiness Coach AI system.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "For informational purposes only. Not a guarantee of loan approval.", align="C", new_x="LMARGIN", new_y="NEXT")

    # ═══ PAGE 2: EXECUTIVE SUMMARY ═══
    pdf.add_page()
    pdf.section_title("Executive Summary")
    verdict = {
        "Critical": "This loan application shows significant risk factors. Major restructuring is recommended before approaching lenders.",
        "At-Risk": "This application has some concerning risk factors that need to be addressed. With the right changes, approval chances can improve significantly.",
        "Stable": "This is a solid application with a reasonable chance of approval. Some minor optimizations could strengthen it further.",
        "Growing": "This is a strong loan application. The business profile matches historically successful loans.",
        "Thriving": "Excellent! This application matches the strongest performing loans in our database of 897K businesses.",
    }
    pdf.body_text(f"Overall Assessment: {label} (Grade {grade})")
    pdf.body_text(verdict.get(label, ""))
    pdf.body_text(f"The AI model predicts this loan application falls into the '{label}' category with {conf*100:.1f}% confidence. This assessment is based on analysis of 897,167 historical Small Business Administration (SBA) loans using an XGBoost classifier with SHAP explainability.")

    # Key numbers
    pdf.sub_title("Key Figures")
    amt = features.get("DisbursementGross", 0)
    pdf.body_text(f"  - Loan Amount Requested: {fmt_inr(amt)} (${amt:,.0f})")
    pdf.body_text(f"  - Loan Term: {features.get('Term', 0)} months ({features.get('Term', 0)/12:.1f} years)")
    pdf.body_text(f"  - Employees: {features.get('NoEmp', 0)} | New Jobs Planned: {features.get('CreateJob', 0)}")
    pdf.body_text(f"  - SBA Guarantee: {fmt_inr(features.get('SBA_Appv', 0))} ({features.get('SBA_Appv',0)/max(features.get('GrAppv',1),1)*100:.0f}% coverage)")

    # Strengths + Risks from SHAP
    if shap_data:
        pdf.sub_title("Top Strengths")
        for f in shap_data.get("top_negative_features", []):
            pdf.body_text(f"  + {FEAT_LABELS.get(f, f)}")
        pdf.sub_title("Top Risk Factors")
        for f in shap_data.get("top_positive_features", []):
            pdf.body_text(f"  ! {FEAT_LABELS.get(f, f)}")

    if red_flags and red_flags.get("flags"):
        n = len(red_flags["flags"])
        nh = sum(1 for f in red_flags["flags"] if f["severity"] == "high")
        pdf.body_text(f"Warnings: {n} issues detected ({nh} critical). See Section 5 for details.")

    # ═══ PAGE 3: BUSINESS PROFILE ═══
    pdf.add_page()
    pdf.section_title("Business Profile & Classification")
    pdf.sub_title("Input Parameters")
    rows = [(FEAT_LABELS.get(k, k), fmt_feat_val(k, features[k])) for k in features if k in FEAT_LABELS]
    pdf.add_kv_table(rows)

    pdf.sub_title("Model Prediction")
    pdf.add_kv_table([
        ("Predicted Class", f"{pred['predicted_class']} - {label}"),
        ("Grade", grade),
        ("Confidence", f"{conf*100:.1f}%"),
        ("Model Used", pred.get("model_used", "XGBoost")),
    ])

    # Probability chart
    try:
        img = probability_chart(pred["probabilities"])
        pdf.image(img, x=15, w=180)
    except Exception as e:
        pdf.body_text(f"(Chart generation error: {e})")

    # ═══ PAGE 4: RADAR CHART ═══
    pdf.add_page()
    pdf.section_title("Readiness Dimensions")
    pdf.body_text("The radar chart below shows your loan application's strength across 6 key dimensions. Higher scores indicate stronger positioning in that area.")

    def compute_radar(f):
        s = {}
        s["Loan Term"] = min(100, (f.get("Term", 84) / 240) * 100)
        emp = f.get("NoEmp", 0) + f.get("CreateJob", 0) + f.get("RetainedJob", 0)
        s["Employment"] = min(100, (emp / 30) * 100)
        s["Maturity"] = 80 if f.get("NewExist", 1) == 1 else 30
        gr = f.get("GrAppv", 1)
        sba = f.get("SBA_Appv", 0)
        s["Guarantee"] = min(100, (sba / max(gr, 1)) * 120)
        s["Location"] = {1: 80, 0: 50, 2: 40}.get(f.get("UrbanRural", 0), 50)
        doc = 100
        if f.get("LowDoc", 0) == 1: doc -= 40
        if f.get("RevLineCr", 0) == 1: doc -= 20
        s["Documentation"] = max(0, doc)
        return s

    scores = compute_radar(features)
    try:
        img = radar_chart(scores)
        pdf.image(img, x=30, w=150)
    except Exception as e:
        pdf.body_text(f"(Chart error: {e})")

    pdf.ln(4)
    pdf.sub_title("Dimension Scores")
    pdf.add_kv_table([(k, f"{v:.0f}/100") for k, v in scores.items()])

    # ═══ PAGE 5: SHAP ANALYSIS ═══
    if shap_data:
        pdf.add_page()
        pdf.section_title("Explainability Analysis (SHAP)")
        pdf.body_text("SHAP (SHapley Additive exPlanations) values show how each feature contributed to the prediction. Green bars are strengths (pushing toward better ratings), red bars are risk factors (pushing toward worse ratings).")
        try:
            img = shap_chart(shap_data["feature_contributions"], label)
            pdf.image(img, x=15, w=180)
        except Exception as e:
            pdf.body_text(f"(Chart error: {e})")
        pdf.ln(4)
        pdf.sub_title("Feature Impact Table")
        contribs = sorted(shap_data["feature_contributions"].items(), key=lambda x: abs(x[1]), reverse=True)
        rows = [(FEAT_LABELS.get(f, f), f"{v:+.4f} ({'Risk' if v > 0 else 'Strength'})") for f, v in contribs]
        pdf.add_kv_table(rows)

    # ═══ PAGE 6: RED FLAGS ═══
    pdf.add_page()
    pdf.section_title("Risk Warnings & Red Flags")
    if red_flags and red_flags.get("flags"):
        for flag in red_flags["flags"]:
            sev = flag["severity"].upper()
            pdf.set_font("Helvetica", "B", 11)
            color = {"HIGH": (244,67,54), "MEDIUM": (255,152,0), "LOW": (33,150,243)}
            c = color.get(sev, (100,100,100))
            pdf.set_text_color(*c)
            pdf.cell(0, 7, strip_emoji(f"[{sev}] {flag['flag']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(60, 60, 60)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.5, strip_emoji(flag["explanation"]))
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(56, 142, 60)
            pdf.multi_cell(0, 5.5, strip_emoji(f"Suggestion: {flag['suggestion']}"))
            pdf.ln(4)
    else:
        pdf.body_text("No red flags detected. Your loan structure looks solid.")

    # ═══ PAGE 7-8: OPTIMIZER ═══
    if optimizer:
        pdf.add_page()
        pdf.section_title("Loan Structure Optimization")
        if optimizer.get("changes") and optimizer.get("improvement", 0) > 0:
            orig = optimizer["original_prediction"]
            opt = optimizer["optimized_prediction"]
            pdf.body_text(f"By restructuring your loan, the predicted outcome improves from {orig['predicted_label']} to {opt['predicted_label']} (confidence: {opt['confidence']*100:.0f}%).")
            pdf.sub_title("Recommended Changes")
            for c in optimizer["changes"]:
                feat = c["feature"]
                ov, nv = c["original"], c["optimized"]
                if feat in ("DisbursementGross", "SBA_Appv", "GrAppv"):
                    pdf.body_text(f"  - {FEAT_LABELS.get(feat, feat)}: {fmt_inr(ov)} -> {fmt_inr(nv)}")
                elif feat == "Term":
                    pdf.body_text(f"  - Loan Term: {ov} months -> {nv} months ({nv/12:.1f} years)")
                else:
                    pdf.body_text(f"  - {FEAT_LABELS.get(feat, feat)}: {ov} -> {nv}")
            # Amount analysis
            amt = optimizer.get("amount_analysis")
            if amt and amt.get("max_safe_amount", 0) > 0:
                pdf.body_text(f"Safe Borrowing Limit: {fmt_inr(amt['max_safe_amount'])} (requested: {fmt_inr(amt['requested_amount'])})")
            term = optimizer.get("term_analysis")
            if term:
                pdf.body_text(f"Optimal Term: {term['recommended_term']} months ({term['recommended_term_years']} years)")
        else:
            pdf.body_text("Your loan structure is already well-optimized. No significant improvements found.")

        # Sensitivity charts
        if optimizer.get("term_analysis"):
            try:
                img = term_sensitivity_chart(optimizer["term_analysis"])
                if img: pdf.image(img, x=15, w=180)
            except: pass
        if optimizer.get("amount_analysis"):
            try:
                img = amount_sensitivity_chart(optimizer["amount_analysis"])
                if img:
                    if pdf.get_y() > 200: pdf.add_page()
                    pdf.image(img, x=15, w=180)
            except: pass

    # ═══ PAGE 9: PEER COMPARISON ═══
    if similar:
        pdf.add_page()
        pdf.section_title("Peer Comparison (Similar Businesses)")
        pdf.body_text(similar.get("insight", ""))
        pdf.add_kv_table([
            ("Businesses Matched", similar.get("total_similar", 0)),
            ("Success Rate (Peers)", f"{similar.get('success_rate', 0)*100:.0f}%"),
            ("Baseline Success Rate", f"{similar.get('baseline_success_rate', 0)*100:.0f}%"),
            ("Risk Level", similar.get("risk_vs_baseline", "N/A")),
            ("Dataset Size", f"{similar.get('dataset_size', 897167):,} loans"),
        ])
        try:
            img = peer_comparison_chart(similar["success_rate"], similar["baseline_success_rate"])
            pdf.image(img, x=30, w=140)
        except: pass
        # Similar businesses table
        bizes = similar.get("similar_businesses", [])
        if bizes:
            pdf.ln(4)
            pdf.sub_title("Top 5 Most Similar Businesses")
            for b in bizes:
                pdf.body_text(f"  {b['rank']}. {b['name']} ({b['state']}) - {b['outcome']} | {b['employees']} employees | {fmt_inr(b['disbursement'])} | Match: {b['similarity_score']*100:.0f}%")

    # ═══ PAGE 10: GOVERNMENT SCHEMES ═══
    if schemes and schemes.get("schemes"):
        pdf.add_page()
        pdf.section_title("Government Schemes Matched")
        pdf.body_text("Based on your business profile, the following government schemes may help reduce your borrowing costs or provide subsidies:")
        for s in schemes["schemes"]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(25, 118, 210)
            rel = "HIGH MATCH" if s["relevance"] == "high" else "PARTIAL MATCH"
            max_inr = s.get("max_amount_inr", 0)
            if max_inr >= 1e7: ms = f"Rs.{max_inr/1e7:.0f} Cr"
            elif max_inr >= 1e5: ms = f"Rs.{max_inr/1e5:.0f}L"
            else: ms = f"Rs.{max_inr/1e3:.0f}K"
            pdf.cell(0, 7, strip_emoji(f"{s['name']} [{rel}] - Up to {ms}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(60, 60, 60)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, strip_emoji(s["description"]))
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(56, 142, 60)
            for b in s["benefits"]:
                pdf.cell(0, 5, strip_emoji(f"  - {b}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(100, 100, 100)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 5, s["url"], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # ═══ LAST PAGE: DISCLAIMER ═══
    pdf.add_page()
    pdf.section_title("Methodology & Disclaimer")
    pdf.sub_title("Model Details")
    pdf.add_kv_table([
        ("Primary Model", "XGBoost (Gradient Boosted Trees)"),
        ("Secondary Model", "LightGBM"),
        ("Training Dataset", "SBA National Dataset (897,167 loans)"),
        ("Features Used", "11 loan/business characteristics"),
        ("Classification", "5-class (Critical, At-Risk, Stable, Growing, Thriving)"),
        ("Explainability", "SHAP (Tree Explainer)"),
        ("Similarity Engine", "K-Nearest Neighbors (K=50)"),
    ])
    pdf.sub_title("Important Disclaimer")
    pdf.body_text("This report is generated by an AI-powered assessment system for INFORMATIONAL PURPOSES ONLY. It does not constitute financial advice, a loan guarantee, or a credit decision. The predictions are based on historical patterns from the US Small Business Administration dataset and may not perfectly reflect current market conditions or individual circumstances.")
    pdf.body_text("Business owners should consult with qualified financial advisors, chartered accountants, and banking professionals before making loan decisions. The AI model's predictions are probabilistic and should be used as one input among many in the decision-making process.")
    pdf.body_text(f"Report generated on {now} | Report ID: {report_id}")

    return pdf.output()
