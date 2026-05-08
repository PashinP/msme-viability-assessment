"""
Build the Appendix HTML document with Jupyter-styled code + screenshots.
Run: python3 build_appendix.py
Output: appendix_final.html (open in Chrome → Print → Save as PDF)
"""
import base64, os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SS_DIR = os.path.join(PROJECT, "appendix_screenshots")
VIZ_DIR = os.path.join(PROJECT, "visualizations")

def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def embed_img(path, caption=""):
    b64 = img_b64(path)
    return f'''<figure>
<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:6px;">
<figcaption>{caption}</figcaption>
</figure>'''

# Read source files
with open(os.path.join(PROJECT, "api", "engine.py")) as f:
    engine_code = f.read()
with open(os.path.join(PROJECT, "api", "optimizer.py")) as f:
    optimizer_code = f.read()

# Escape HTML
def esc(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Appendix — MSME Viability Assessment System</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',sans-serif; color:#1a1a2e; line-height:1.6; max-width:210mm; margin:0 auto; padding:20mm 18mm; background:#fff; }}
h1 {{ font-size:22px; font-weight:700; text-align:center; margin-bottom:4px; }}
h2 {{ font-size:17px; font-weight:700; margin:28px 0 10px; border-bottom:2px solid #7b2ff7; padding-bottom:4px; color:#1a1a2e; page-break-after:avoid; }}
h3 {{ font-size:14px; font-weight:600; margin:18px 0 6px; color:#333; page-break-after:avoid; }}
p, li {{ font-size:11px; }}
.subtitle {{ text-align:center; font-size:12px; color:#666; margin-bottom:20px; }}

/* Jupyter-style code cell */
.jupyter-cell {{ background:#282c34; border-radius:6px; margin:10px 0; overflow:hidden; page-break-inside:avoid; }}
.jupyter-cell .cell-header {{ background:#21252b; padding:4px 12px; font-size:10px; color:#7f8c98; font-family:'JetBrains Mono',monospace; border-bottom:1px solid #373b41; }}
.jupyter-cell .cell-header .tag {{ color:#61afef; }}
.jupyter-cell pre {{ padding:10px 14px; margin:0; font-size:8.5px; line-height:1.45; font-family:'JetBrains Mono',monospace; color:#abb2bf; overflow-x:auto; white-space:pre; }}
.jupyter-cell pre .kw {{ color:#c678dd; }}
.jupyter-cell pre .fn {{ color:#61afef; }}
.jupyter-cell pre .st {{ color:#98c379; }}
.jupyter-cell pre .cm {{ color:#5c6370; font-style:italic; }}
.jupyter-cell pre .nb {{ color:#e5c07b; }}
.jupyter-cell pre .num {{ color:#d19a66; }}
.jupyter-cell pre .op {{ color:#56b6c2; }}

/* Screenshot figures */
figure {{ margin:12px 0; text-align:center; page-break-inside:avoid; }}
figure img {{ max-width:92%; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.12); }}
figcaption {{ font-size:10px; color:#555; margin-top:6px; font-style:italic; }}

.file-label {{ display:inline-block; background:#f0e6ff; color:#5b21b6; padding:2px 8px; border-radius:4px; font-size:10px; font-family:'JetBrains Mono',monospace; margin-bottom:4px; }}
.desc {{ font-size:11px; color:#444; margin:4px 0 8px; }}
hr {{ border:none; border-top:1px solid #e0e0e0; margin:20px 0; }}

@media print {{
  body {{ padding:12mm 15mm; }}
  .jupyter-cell {{ break-inside:avoid; }}
  figure {{ break-inside:avoid; }}
}}
</style>
</head><body>

<h1>Appendix: Source Code & Working Prototype</h1>
<p class="subtitle">MSME Viability Assessment System — Key Implementation Files & Live Screenshots</p>
<hr>

<h2>A. Core Source Code</h2>

<h3>A.1 Prediction Engine with SHAP Explainability & Counterfactual Recommendations</h3>
<span class="file-label">api/engine.py</span>
<p class="desc">Central ML module — loads XGBoost/LightGBM models, provides real-time SHAP-based explainability via TreeExplainer, and implements a DiCE-inspired deterministic grid-search for counterfactual recommendations. All ML logic is concentrated here to keep the API layer thin.</p>

<div class="jupyter-cell">
<div class="cell-header">In [<span class="tag">1</span>]: <span class="tag">api/engine.py</span> — 234 lines</div>
<pre>{esc(engine_code)}</pre>
</div>

<h3>A.2 Loan Structure Optimizer, Red-Flag Detection & Government Scheme Matching</h3>
<span class="file-label">api/optimizer.py</span>
<p class="desc">Three custom-built analytical subsystems: (1) parametric loan structure optimizer sweeping term/amount configurations, (2) rule-based red-flag detection derived from 899,164 historical loan patterns, (3) government scheme matching engine with a hand-curated knowledge base of 10 Indian MSME financing programs.</p>

<div class="jupyter-cell">
<div class="cell-header">In [<span class="tag">2</span>]: <span class="tag">api/optimizer.py</span> — 350 lines</div>
<pre>{esc(optimizer_code)}</pre>
</div>

<hr>
<h2>B. Model Training Results & Visualizations</h2>

<h3>B.1 Model Comparison — 5-Class Benchmark</h3>
<p class="desc">Accuracy, precision, recall, and F1-score across six models trained on 899,164 SBA loan records (test set n=178,385). XGBoost and LightGBM achieve ~93% weighted accuracy.</p>
{embed_img(os.path.join(VIZ_DIR, "model_comparison_table.png"), "Figure B.1: Comparative performance of all trained classifiers on the 5-class viability spectrum.")}

<h3>B.2 5-Class Health Label Distribution</h3>
<p class="desc">Distribution of engineered viability labels: Critical (16.2%), At-Risk (1.2%), Stable (53.3%), Growing (17.4%), Thriving (11.8%).</p>
{embed_img(os.path.join(VIZ_DIR, "multiclass_distribution.png"), "Figure B.2: Class distribution across the full 899,164-record dataset.")}

<h3>B.3 XGBoost Confusion Matrix</h3>
<p class="desc">Confusion matrix for the primary XGBoost classifier. Diagonal values confirm strong per-class accuracy, with At-Risk showing the most confusion due to class imbalance (1.2% representation).</p>
{embed_img(os.path.join(VIZ_DIR, "xgb_confusion_matrix.png"), "Figure B.3: XGBoost 5-class confusion matrix (test set, n=178,385).")}

<h3>B.4 SHAP Global Feature Importance</h3>
<p class="desc">Mean absolute SHAP values showing Term (loan duration), DisbursementGross, and SBA_Appv as the dominant predictors of viability class.</p>
{embed_img(os.path.join(VIZ_DIR, "mc_shap_global_bar.png"), "Figure B.4: Global SHAP feature importance across all five viability classes.")}

<h3>B.5 SHAP Beeswarm Summary Plot</h3>
<p class="desc">Per-instance SHAP contributions colored by feature value. Longer loan terms and higher disbursement amounts consistently push predictions toward healthier classes.</p>
{embed_img(os.path.join(VIZ_DIR, "shap_summary_plot.png"), "Figure B.5: SHAP beeswarm plot — feature-level impact on model output.")}

<hr>
<h2>C. Working Prototype Screenshots</h2>
<p class="desc">The following screenshots are from the deployed MSME Loan Readiness Coach, demonstrating the end-to-end system capabilities in a real-time production environment.</p>

<h3>C.1 Expert Mode — Direct Feature Input</h3>
<p class="desc">The Expert Mode interface allows bank officers to enter structured loan parameters directly. All 11 features are input via the form, and the system returns a complete viability assessment.</p>
{embed_img(os.path.join(SS_DIR, "01_expert_input_form.png"), "Figure C.1: Expert Mode input form with 11 loan parameters.")}

<h3>C.2 Viability Grade, Radar Chart & Prescriptive Recommendations</h3>
<p class="desc">Assessment result showing the viability grade (C/Stable at 99.6% confidence), a radar chart visualizing the business profile across key dimensions, and actionable recommendations to improve the application — including specific loan term extension and documentation suggestions.</p>
{embed_img(os.path.join(SS_DIR, "02_grade_radar_recommendations.png"), "Figure C.2: Viability assessment result with prescriptive improvement recommendations.")}

<h3>C.3 Similar Businesses & Government Scheme Matching</h3>
<p class="desc">The system matches the user's profile against 899K historical records to find similar businesses (KNN-based) and automatically recommends relevant Indian government schemes (MUDRA, NSIC, CGTMSE) based on eligibility criteria.</p>
{embed_img(os.path.join(SS_DIR, "03_similar_biz_govt_schemes.png"), "Figure C.3: Peer comparison and auto-matched government financing schemes.")}

<h3>C.4 SHAP Feature Contribution Analysis</h3>
<p class="desc">Real-time SHAP analysis showing which features pushed the prediction toward or away from the assessed class. The horizontal bar chart reveals Term and SBA_Appv as the strongest contributors for this specific application.</p>
{embed_img(os.path.join(SS_DIR, "04_shap_analysis.png"), "Figure C.4: Per-prediction SHAP waterfall chart and additional scheme matches.")}

<h3>C.5 AI Loan Coach — Conversational Assessment</h3>
<p class="desc">The Loan Coach tab provides a conversational interface where business owners describe their situation in natural language (with Hindi/English support). The AI advisor extracts the 11 required features through dialogue and triggers the assessment pipeline.</p>
{embed_img(os.path.join(SS_DIR, "05_loan_coach_chat.png"), "Figure C.5: Gemini-powered conversational loan readiness assessment interface.")}

<h3>C.6 Analytics Dashboard</h3>
<p class="desc">The Analytics tab aggregates all historical predictions stored in the SQLite audit database. It displays total assessments processed, average model confidence, risk class distribution, and usage trends — providing institutional-level monitoring capabilities.</p>
{embed_img(os.path.join(SS_DIR, "06_analytics_dashboard.png"), "Figure C.6: Real-time analytics dashboard with prediction audit trail.")}

</body></html>'''

out_path = os.path.join(PROJECT, "appendix_final.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"✅ Created: {out_path}")
print(f"   Open in Chrome → File → Print → Save as PDF")
print(f"   File size: {os.path.getsize(out_path) / 1024:.0f} KB")
