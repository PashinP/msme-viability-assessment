"""
MSME Viability Assessment — Enterprise Streamlit Dashboard
============================================================
This frontend queries the FastAPI backend (localhost:8000) for all
ML operations. It provides four modules:

  1. Single Assessment — Interactive loan viability check
  2. SHAP Explanation  — Feature contribution analysis
  3. Batch Processing  — Bulk CSV upload & download
  4. Analytics         — Historical monitoring dashboard
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import os

# ── Configuration ──
API_BASE = os.environ.get("API_URL", "https://msme-viability-assessment.onrender.com")
API_KEY = "msme-dev-key-2024"
HEADERS = {"X-API-Key": API_KEY}

LABEL_NAMES = {0: "Critical", 1: "At-Risk", 2: "Stable", 3: "Growing", 4: "Thriving"}
LABEL_COLORS = {"Critical": "#d32f2f", "At-Risk": "#f57c00", "Stable": "#388e3c",
                "Growing": "#1976d2", "Thriving": "#7b1fa2"}
LABEL_EMOJIS = {"Critical": "🔴", "At-Risk": "🟠", "Stable": "🟢",
                "Growing": "🔵", "Thriving": "🟣"}

# ── Page Setup ──
st.set_page_config(
    page_title="MSME Viability Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.95rem; }

    .health-card {
        padding: 1.4rem; border-radius: 14px; text-align: center;
        color: white; font-weight: 700; font-size: 1.4rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25); margin-bottom: 1rem;
    }

    .metric-box {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c; border-radius: 12px;
        padding: 1rem; text-align: center; color: white;
    }
    .metric-box .label { font-size: 0.78rem; color: #a0a0b0; margin-bottom: 0.2rem; }
    .metric-box .value { font-size: 1.5rem; font-weight: 700; }

    .rec-box {
        background: linear-gradient(135deg, #1b3a2d, #1e4d3a);
        border-left: 4px solid #4caf50; padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0; margin: 0.4rem 0; color: #e0e0e0;
    }
    .rec-box strong { color: #81c784; }

    .analytics-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c; border-radius: 14px;
        padding: 1.5rem; text-align: center; color: white;
    }
    .analytics-card .big-num { font-size: 2.5rem; font-weight: 800; }
    .analytics-card .sub { font-size: 0.85rem; color: #a0a0b0; margin-top: 0.3rem; }

    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──

def api_call(method, endpoint, **kwargs):
    """Make a call to the FastAPI backend."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, headers=HEADERS, timeout=30)
        elif method == "POST":
            r = requests.post(url, headers=HEADERS, timeout=60, **kwargs)
        elif method == "FILE":
            r = requests.post(url, headers={"X-API-Key": API_KEY}, timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ **Cannot connect to the API server.** Make sure the FastAPI backend is running on port 8000.\n\n"
                 "Run this in a terminal:\n```\ncd ~/Desktop/Practicum_Project\npython3 -m uvicorn api.server:app --port 8000\n```")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None


def get_sidebar_inputs():
    """Render the loan application form in the sidebar."""
    st.sidebar.markdown("## 📋 Loan Application")

    st.sidebar.markdown("#### 💰 Loan Details")
    term = st.sidebar.slider("Loan Term (months)", 0, 480, 84, step=6)
    disbursement = st.sidebar.number_input("Disbursement Amount ($)", 1000, 5000000, 100000, step=5000)
    gr_appv = st.sidebar.number_input("Gross Approved ($)", 1000, 5000000, 100000, step=5000)
    sba_appv = st.sidebar.number_input("SBA Guaranteed ($)", 500, 4000000, 75000, step=5000)

    st.sidebar.markdown("#### 👥 Business Details")
    no_emp = st.sidebar.number_input("Employees", 0, 500, 5, step=1)
    create_job = st.sidebar.number_input("Jobs to Create", 0, 200, 2, step=1)
    retained_job = st.sidebar.number_input("Jobs Retained", 0, 200, 3, step=1)
    new_exist = st.sidebar.selectbox("Business Type", [1, 2], format_func=lambda x: {1: "Existing", 2: "New"}[x])
    urban_rural = st.sidebar.selectbox("Location", [1, 2, 0], format_func=lambda x: {1: "Urban", 2: "Rural", 0: "Undefined"}[x])
    rev_line = st.sidebar.selectbox("Revolving Credit", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
    low_doc = st.sidebar.selectbox("Low Doc Loan", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])

    return {
        "Term": term, "NoEmp": no_emp, "NewExist": new_exist,
        "CreateJob": create_job, "RetainedJob": retained_job,
        "DisbursementGross": disbursement, "UrbanRural": urban_rural,
        "RevLineCr": rev_line, "LowDoc": low_doc,
        "SBA_Appv": sba_appv, "GrAppv": gr_appv,
    }


# ── Page Renderers ──

def render_assessment(features):
    """Tab 1: Single assessment + recommendation."""
    predict_btn = st.button("🔍 Assess Viability", use_container_width=True, type="primary")

    if not predict_btn:
        st.markdown("#### 👈 Fill the sidebar and click **Assess Viability**")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**📊 Predict**\n\n5-class viability from 2 models")
        c2.markdown("**🔍 Explain**\n\nSHAP feature contributions")
        c3.markdown("**💡 Fix**\n\nAI-generated improvement plan")
        return

    # Call /predict
    with st.spinner("Running viability assessment..."):
        pred = api_call("POST", "/predict", json=features)
    if not pred:
        return

    label = pred["predicted_label"]
    conf = pred["confidence"]
    color = LABEL_COLORS[label]
    emoji = LABEL_EMOJIS[label]

    # Health badge
    st.markdown(f'<div class="health-card" style="background:{color};">'
                f'{emoji} Viability: {label} — {conf*100:.1f}% confidence</div>',
                unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Health Class", f"{pred['predicted_class']+1} / 5"),
        (c2, "Confidence", f"{conf*100:.1f}%"),
        (c3, "Loan Term", f"{features['Term']} mo"),
        (c4, "Prediction ID", f"#{pred['prediction_id']}"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="label">{lbl}</div>'
                     f'<div class="value">{val}</div></div>', unsafe_allow_html=True)

    # Probability chart
    st.markdown("### Class Probability Distribution")
    proba = pred["probabilities"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    classes = list(proba.keys())
    vals = list(proba.values())
    bar_colors = [LABEL_COLORS[c] for c in classes]
    bars = ax.barh(classes, vals, color=bar_colors, height=0.6, edgecolor='none')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', color='white', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # SHAP
    st.markdown("### 🔍 SHAP Feature Contributions")
    with st.spinner("Computing SHAP explanations..."):
        shap_data = api_call("POST", "/explain", json=features)
    if shap_data:
        contribs = shap_data["feature_contributions"]
        sorted_feats = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        feat_names = [f[0] for f in sorted_feats]
        feat_vals = [f[1] for f in sorted_feats]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        colors = ["#ef5350" if v > 0 else "#66bb6a" for v in feat_vals]
        ax2.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1], height=0.6)
        ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("SHAP Value (impact on prediction)", color='white', fontsize=10)
        ax2.set_title(f"Why the model predicted: {label}", color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown(f"**Top risk factors:** {', '.join(shap_data['top_positive_features'])}")
        st.markdown(f"**Top protective factors:** {', '.join(shap_data['top_negative_features'])}")

    # Recommendations
    if pred["predicted_class"] >= 4:
        st.success("🎉 This MSME is **Thriving** — no improvement needed!")
    else:
        st.markdown("### 💡 Prescriptive Recommendations")
        with st.spinner("Computing optimal interventions..."):
            rec = api_call("POST", "/recommend", json={"application": features})
        if rec:
            target_label = rec["target_label"]
            st.markdown(f"**Goal:** {emoji} {label} → {LABEL_EMOJIS[target_label]} {target_label}")
            if rec["feasible"]:
                for change in rec["changes"]:
                    orig = change["original_value"]
                    new = change["recommended_value"]
                    feat_label = change["feature_label"]
                    direction = change["direction"]
                    if "Amount" in feat_label or "Approved" in feat_label or "Guaranteed" in feat_label:
                        st.markdown(f'<div class="rec-box"><strong>{direction} {feat_label}</strong><br>'
                                    f'${orig:,.0f} → ${new:,.0f}</div>', unsafe_allow_html=True)
                    elif "Term" in feat_label:
                        st.markdown(f'<div class="rec-box"><strong>{direction} {feat_label}</strong><br>'
                                    f'{orig:.0f} months → {new:.0f} months</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="rec-box"><strong>{direction} {feat_label}</strong><br>'
                                    f'{orig:.0f} → {new:.0f}</div>', unsafe_allow_html=True)
            else:
                st.warning("No single or dual-feature change found. This business may need fundamental restructuring.")


def render_batch():
    """Tab 2: Batch CSV processing."""
    st.markdown("### 📁 Batch Processing")
    st.markdown("Upload a CSV file with loan applications to process them all at once.")

    # Template download
    template_cols = "Term,NoEmp,NewExist,CreateJob,RetainedJob,DisbursementGross,UrbanRural,RevLineCr,LowDoc,SBA_Appv,GrAppv"
    sample_rows = (
        "84,5,1,2,3,100000,1,0,0,75000,100000\n"
        "36,2,2,0,1,50000,2,0,0,25000,50000\n"
        "240,10,1,5,8,500000,1,1,0,400000,500000\n"
    )
    template_csv = template_cols + "\n" + sample_rows
    st.download_button("📥 Download Template CSV", template_csv, "msme_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        st.markdown("---")
        df_preview = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.markdown(f"**Preview:** {len(df_preview)} applications found")
        st.dataframe(df_preview.head(), use_container_width=True, hide_index=True)

        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            uploaded.seek(0)
            with st.spinner(f"Processing {len(df_preview)} applications..."):
                start = time.time()
                result = api_call("FILE", "/predict/batch",
                                  files={"file": ("upload.csv", uploaded.getvalue(), "text/csv")})
                elapsed = time.time() - start

            if result:
                st.success(f"✅ Processed **{result['total_processed']}** applications in **{elapsed:.1f}s**"
                           f" (Batch ID: `{result['batch_id']}`)")

                # Summary
                st.markdown("#### Risk Distribution")
                summary = result["summary"]
                cols = st.columns(5)
                for i, (label, count) in enumerate(summary.items()):
                    emoji = LABEL_EMOJIS.get(label, "")
                    color = LABEL_COLORS.get(label, "#555")
                    cols[i].markdown(
                        f'<div class="metric-box"><div class="label">{emoji} {label}</div>'
                        f'<div class="value" style="color:{color};">{count}</div></div>',
                        unsafe_allow_html=True
                    )

                # Results table
                st.markdown("#### Detailed Results")
                rows = []
                for r in result["results"]:
                    rows.append({
                        "ID": r["prediction_id"],
                        "Prediction": f"{LABEL_EMOJIS[r['predicted_label']]} {r['predicted_label']}",
                        "Confidence": f"{r['confidence']*100:.1f}%",
                        "Model": r["model_used"],
                    })
                results_df = pd.DataFrame(rows)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Download results
                csv_out = results_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv_out, "msme_results.csv", "text/csv")


def render_analytics():
    """Tab 3: Historical analytics from the database."""
    st.markdown("### 📈 Analytics Dashboard")

    data = api_call("GET", "/analytics")
    if not data:
        return

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="analytics-card"><div class="big-num">{data["total_predictions"]}</div>'
                f'<div class="sub">Total Predictions</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="analytics-card"><div class="big-num">{data["average_confidence"]*100:.1f}%</div>'
                f'<div class="sub">Average Confidence</div></div>', unsafe_allow_html=True)
    n_classes = len(data.get("class_distribution", {}))
    c3.markdown(f'<div class="analytics-card"><div class="big-num">{n_classes}</div>'
                f'<div class="sub">Risk Classes Seen</div></div>', unsafe_allow_html=True)

    # Class distribution chart
    dist = data.get("class_distribution", {})
    if dist:
        st.markdown("#### Risk Distribution (All Time)")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        labels = list(dist.keys())
        values = list(dist.values())
        colors = [LABEL_COLORS.get(l, "#555") for l in labels]
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Recent predictions table
    recent = data.get("recent_predictions", [])
    if recent:
        st.markdown("#### Recent Predictions")
        recent_df = pd.DataFrame(recent)
        recent_df["confidence"] = recent_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        recent_df["disbursement"] = recent_df["disbursement"].apply(lambda x: f"${x:,.0f}")
        recent_df.columns = ["ID", "Timestamp", "Prediction", "Confidence", "Term", "Disbursement"]
        st.dataframe(recent_df, use_container_width=True, hide_index=True)


# ── Main App ──

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 MSME Viability Assessment System</h1>
        <p>Enterprise AI platform for loan risk stratification · Powered by XGBoost + SHAP + Counterfactual AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar inputs (shared across tabs)
    features = get_sidebar_inputs()

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Assessment",
        "📁 Batch Processing",
        "📈 Analytics"
    ])

    with tab1:
        render_assessment(features)
    with tab2:
        render_batch()
    with tab3:
        render_analytics()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
        "MSME Viability Assessment v1.0 · FastAPI Backend on port 8000 · "
        "Models: XGBoost + LightGBM · Database: SQLite"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
