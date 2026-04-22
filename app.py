"""
MSME Loan Readiness Coach — Streamlit Dashboard v2.1
=======================================================
India-first · Chat-based AI assessment · ₹ Currency · Clean UI
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import time
import os
import math

# ── Configuration ──
API_BASE = os.environ.get("API_URL", "https://msme-viability-assessment.onrender.com")
API_KEY = os.environ.get("MSME_API_KEY", "msme-dev-key-2024")
HEADERS = {"X-API-Key": API_KEY}
USD_TO_INR = 83

LABEL_NAMES = {0: "Critical", 1: "At-Risk", 2: "Stable", 3: "Growing", 4: "Thriving"}
LABEL_COLORS = {"Critical": "#d32f2f", "At-Risk": "#f57c00", "Stable": "#388e3c",
                "Growing": "#1976d2", "Thriving": "#7b1fa2"}
LABEL_EMOJIS = {"Critical": "🔴", "At-Risk": "🟠", "Stable": "🟢",
                "Growing": "🔵", "Thriving": "🟣"}
GRADE_MAP = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}
GRADE_COLORS = {"A": "#4caf50", "B": "#2196f3", "C": "#ff9800", "D": "#f44336", "F": "#b71c1c"}


def fmt_inr(usd):
    """Format USD → ₹ in Indian style."""
    inr = usd * USD_TO_INR
    if inr >= 10000000:
        return f"₹{inr/10000000:.1f} Cr"
    elif inr >= 100000:
        return f"₹{inr/100000:.1f}L"
    elif inr >= 1000:
        return f"₹{inr/1000:.0f}K"
    return f"₹{inr:,.0f}"


# ── Page Setup ──
st.set_page_config(
    page_title="MSME Loan Readiness Coach",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS (cleaner, less boxy) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.95rem; }

    .grade-card {
        padding: 2rem; border-radius: 20px; text-align: center;
        color: white; font-weight: 800; margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .grade-card .grade { font-size: 4rem; line-height: 1; }
    .grade-card .label { font-size: 1.2rem; margin-top: 0.5rem; font-weight: 600; }
    .grade-card .conf { font-size: 0.9rem; opacity: 0.85; margin-top: 0.3rem; font-weight: 400; }

    .insight-strip {
        padding: 1rem 1.5rem; border-radius: 12px; color: white;
        margin: 0.5rem 0; font-size: 0.95rem; line-height: 1.5;
    }

    .scheme-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #3a3a5c; border-radius: 14px;
        padding: 1.2rem; color: white; margin: 0.5rem 0;
    }

    .metric-box {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c; border-radius: 12px;
        padding: 1rem; text-align: center; color: white;
    }
    .metric-box .label { font-size: 0.78rem; color: #a0a0b0; margin-bottom: 0.2rem; }
    .metric-box .value { font-size: 1.5rem; font-weight: 700; }

    .analytics-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c; border-radius: 14px;
        padding: 1.5rem; text-align: center; color: white;
    }
    .analytics-card .big-num { font-size: 2.5rem; font-weight: 800; }
    .analytics-card .sub { font-size: 0.85rem; color: #a0a0b0; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def api_call(method, endpoint, **kwargs):
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, headers=HEADERS, timeout=120)
        elif method == "POST":
            r = requests.post(url, headers=HEADERS, timeout=180, **kwargs)
        elif method == "FILE":
            r = requests.post(url, headers={"X-API-Key": API_KEY}, timeout=240, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        st.warning("⏳ Server is waking up (free tier). Please try again in a moment!")
        return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server. Please wait 30 seconds and retry.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None


def compute_radar_scores(features: dict) -> dict:
    scores = {}
    scores["Loan Term"] = min(100, (features.get("Term", 84) / 240) * 100)
    emp = features.get("NoEmp", 0) + features.get("CreateJob", 0) + features.get("RetainedJob", 0)
    scores["Employment"] = min(100, (emp / 30) * 100)
    scores["Maturity"] = 80 if features.get("NewExist", 1) == 1 else 30
    gr = features.get("GrAppv", 1)
    sba = features.get("SBA_Appv", 0)
    scores["Guarantee"] = min(100, (sba / max(gr, 1)) * 120)
    scores["Location"] = {1: 80, 0: 50, 2: 40}.get(features.get("UrbanRural", 0), 50)
    doc_score = 100
    if features.get("LowDoc", 0) == 1: doc_score -= 40
    if features.get("RevLineCr", 0) == 1: doc_score -= 20
    scores["Documentation"] = max(0, doc_score)
    return scores


def render_radar_chart(scores: dict):
    categories = list(scores.keys())
    values = list(scores.values()) + [list(scores.values())[0]]
    angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))] + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.plot(angles, values, 'o-', linewidth=2, color='#64b5f6')
    ax.fill(angles, values, alpha=0.25, color='#64b5f6')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, color='white', fontweight='500')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7, color='#666')
    ax.set_ylim(0, 100)
    ax.grid(color='#333', linewidth=0.5)
    ax.spines['polar'].set_color('#444')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════
# TAB 1: LOAN READINESS COACH (CHAT)
# ═══════════════════════════════════════════

def render_chat_tab():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "extracted_features" not in st.session_state:
        st.session_state.extracted_features = None
    if "assessment_done" not in st.session_state:
        st.session_state.assessment_done = False
    if "pending_voice" not in st.session_state:
        st.session_state.pending_voice = False

    if not st.session_state.messages:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 16px;
                    padding: 2rem; color: white; margin-bottom: 1rem; border: 1px solid #3a3a5c;">
            <h3 style="margin:0;">👋 Welcome to the Loan Readiness Coach!</h3>
            <p style="margin:0.5rem 0 0; opacity:0.85; line-height:1.6;">
            अपने business के बारे में बताइए — क्या करते हैं, कितने लोग काम करते हैं,
            कितने पैसों की जरूरत है, और किसलिए। हम आपकी loan readiness assess करके
            actionable advice देंगे।<br><br>
            <strong>Type or 🎤 speak:</strong> <em>"मैं जयपुर में चाय की दुकान चलाता हूँ, 3 लोग काम करते हैं, ₹5 लाख चाहिए"</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            display_text = msg["content"]
            if "```json" in display_text:
                display_text = display_text.split("```json")[0].strip()
            st.markdown(display_text)

    if st.session_state.assessment_done and st.session_state.extracted_features:
        render_report_card(st.session_state.extracted_features)
        return

    # ── Voice Input (collapsible) ──
    import streamlit.components.v1 as components
    with st.expander("🎤 Voice Input — click to speak", expanded=False):
        st.caption("Click mic → speak → click mic to stop → copy text → paste below")
        components.html(_get_voice_html(), height=60)
        voice_input = st.text_input("Paste voice transcript here:", key="voice_field",
                                     placeholder="Your speech will appear above — copy and paste here")
        if st.button("📤 Send Voice Message", type="primary", use_container_width=True,
                     disabled=not bool(voice_input and voice_input.strip())):
            st.session_state.pending_voice = voice_input.strip()
            st.rerun()

    # Process pending voice
    if st.session_state.get("pending_voice"):
        msg = st.session_state.pending_voice
        st.session_state.pending_voice = False
        _process_user_message(msg)

    # ── Keyboard Chat Input ──
    if prompt := st.chat_input("अपने business और loan की जरूरत बताइए..."):
        _process_user_message(prompt)


def _process_user_message(prompt: str):
    """Process a user message (from keyboard or voice)."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            result = api_call("POST", "/chat", json={"messages": st.session_state.messages})

        if result:
            response = result["response"]
            display_text = response
            if "```json" in display_text:
                display_text = display_text.split("```json")[0].strip()
            st.markdown(display_text)
            st.session_state.messages.append({"role": "assistant", "content": response})

            if result.get("extraction_complete") and result.get("features_extracted"):
                st.session_state.extracted_features = result["features_extracted"]
                st.session_state.assessment_done = True
                st.rerun()
        else:
            st.warning("Could not get a response. Please try again.")


def _get_voice_html():
    """Return the HTML/JS for the voice input microphone button."""
    return """
    <div style="display:flex; align-items:center; gap:10px; font-family:Inter,sans-serif;">
        <button id="micBtn" onclick="toggleMic()" style="
            width:40px; height:40px; border-radius:50%; border:none;
            background:linear-gradient(135deg,#302b63,#24243e);
            color:white; font-size:18px; cursor:pointer;
            display:flex; align-items:center; justify-content:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.3); transition:all 0.3s;">🎤</button>
        <span id="micStatus" style="font-size:13px; color:#888;">Click 🎤 to speak</span>
    </div>
    <script>
    let recognition=null, listening=false, transcript='';
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(SR){
        recognition=new SR();
        recognition.continuous=true;
        recognition.interimResults=true;
        recognition.lang='en-IN';
        recognition.onstart=()=>{
            listening=true;
            document.getElementById('micBtn').style.background='linear-gradient(135deg,#d32f2f,#f44336)';
            document.getElementById('micStatus').textContent='🔴 Listening...';
            document.getElementById('micStatus').style.color='#f44336';
            transcript='';
        };
        recognition.onresult=(e)=>{
            let interim='';
            for(let i=e.resultIndex;i<e.results.length;i++){
                if(e.results[i].isFinal) transcript+=e.results[i][0].transcript+' ';
                else interim+=e.results[i][0].transcript;
            }
            document.getElementById('micStatus').textContent=transcript.trim()||('🎙️ '+interim);
        };
        recognition.onend=()=>{
            listening=false;
            document.getElementById('micBtn').style.background='linear-gradient(135deg,#302b63,#24243e)';
            if(transcript.trim()){
                document.getElementById('micStatus').textContent='✅ '+transcript.trim();
                document.getElementById('micStatus').style.color='#4caf50';
                // Send to Streamlit via the Streamlit component API
                window.parent.postMessage({type:'streamlit:setComponentValue',value:transcript.trim()},'*');
            } else {
                document.getElementById('micStatus').textContent='Click 🎤 to speak';
                document.getElementById('micStatus').style.color='#888';
            }
        };
        recognition.onerror=(e)=>{
            listening=false;
            document.getElementById('micBtn').style.background='linear-gradient(135deg,#302b63,#24243e)';
            document.getElementById('micStatus').textContent=e.error==='not-allowed'?'⚠️ Allow microphone access':'⚠️ '+e.error;
            document.getElementById('micStatus').style.color='#ff9800';
        };
    } else {
        document.getElementById('micStatus').textContent='⚠️ Voice not supported. Use Chrome.';
        document.getElementById('micStatus').style.color='#ff9800';
    }
    function toggleMic(){
        if(!recognition)return;
        if(listening)recognition.stop(); else recognition.start();
    }
    </script>
    """


# ═══════════════════════════════════════════
# REPORT CARD (shared by Chat + Expert Mode)
# ═══════════════════════════════════════════

def render_report_card(features: dict):
    st.markdown("---")
    st.markdown("## 📊 Your Loan Readiness Report")

    with st.spinner("🔍 Analyzing across 897K historical businesses..."):
        pred = api_call("POST", "/predict", json=features)
        shap_data = api_call("POST", "/explain", json=features)
        similar = api_call("POST", "/similar", json=features)
        red_flags = api_call("POST", "/redflags", json=features)
        optimizer = api_call("POST", "/optimize", json=features)
        schemes = api_call("POST", "/schemes", json=features)

    if not pred:
        st.error("Could not complete assessment. Please try again.")
        return

    label = pred["predicted_label"]
    conf = pred["confidence"]
    grade = GRADE_MAP[pred["predicted_class"]]
    grade_color = GRADE_COLORS[grade]
    loan_inr = fmt_inr(features.get("DisbursementGross", 0))

    # ── Grade + Radar ──
    col_grade, col_radar = st.columns([1, 1.5])

    with col_grade:
        st.markdown(
            f'<div class="grade-card" style="background: linear-gradient(135deg, {grade_color}, {grade_color}cc);">'
            f'<div class="grade">{grade}</div>'
            f'<div class="label">{LABEL_EMOJIS[label]} {label}</div>'
            f'<div class="conf">{conf*100:.1f}% confidence · Loan: {loan_inr}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_radar:
        scores = compute_radar_scores(features)
        fig = render_radar_chart(scores)
        st.pyplot(fig)
        plt.close(fig)

    # ── Red Flags (as clean list, not boxes) ──
    if red_flags and red_flags.get("flags"):
        flags = red_flags["flags"]
        n_high = sum(1 for f in flags if f["severity"] == "high")

        if n_high > 0:
            st.markdown(f"### 🚨 {n_high} Critical Warning{'s' if n_high > 1 else ''} Found")
        else:
            st.markdown(f"### ⚠️ {len(flags)} Item{'s' if len(flags) > 1 else ''} to Review")

        for flag in flags:
            sev_color = {"high": "#f44336", "medium": "#ff9800", "low": "#2196f3"}[flag["severity"]]
            st.markdown(
                f'<div class="insight-strip" style="background:linear-gradient(135deg, '
                f'{"#3d1111" if flag["severity"] == "high" else "#3d2e11" if flag["severity"] == "medium" else "#112d3d"}, '
                f'{"#4d1a1a" if flag["severity"] == "high" else "#4d3a1a" if flag["severity"] == "medium" else "#1a3a4d"}); '
                f'border-left: 3px solid {sev_color};">'
                f'<strong>{flag["emoji"]} {flag["flag"]}</strong><br>'
                f'{flag["explanation"]}<br>'
                f'<span style="color:#81c784;">💡 {flag["suggestion"]}</span></div>',
                unsafe_allow_html=True
            )
    elif red_flags:
        st.success("✅ No red flags! Your loan structure looks solid.")

    # ── Loan Optimizer (clean summary, not boxes) ──
    if optimizer and optimizer.get("changes") and optimizer.get("improvement", 0) > 0:
        orig = optimizer["original_prediction"]
        opt = optimizer["optimized_prediction"]

        st.markdown("### 🎯 How to Improve Your Application")
        st.markdown(
            f'<div class="insight-strip" style="background: linear-gradient(135deg, #0d3320, #164430); '
            f'border-left: 3px solid #4caf50;">'
            f'By restructuring your loan: '
            f'<strong>{LABEL_EMOJIS[orig["predicted_label"]]} {orig["predicted_label"]}</strong> → '
            f'<strong>{LABEL_EMOJIS[opt["predicted_label"]]} {opt["predicted_label"]}</strong> '
            f'({opt["confidence"]*100:.0f}% confidence)</div>',
            unsafe_allow_html=True
        )

        FEAT_LABELS = {
            "Term": "Loan Term", "NoEmp": "Employees", "CreateJob": "Jobs to Create",
            "RetainedJob": "Jobs Retained", "DisbursementGross": "Loan Amount",
            "LowDoc": "Documentation", "SBA_Appv": "Guarantee Amount",
            "GrAppv": "Approved Amount", "RevLineCr": "Revolving Credit",
            "NewExist": "Business Type", "UrbanRural": "Location",
        }

        change_lines = []
        for c in optimizer["changes"]:
            feat = c["feature"]
            fl = FEAT_LABELS.get(feat, feat)
            ov, nv = c["original"], c["optimized"]

            if feat in ("DisbursementGross", "SBA_Appv", "GrAppv"):
                change_lines.append(f"**{fl}:** {fmt_inr(ov)} → {fmt_inr(nv)}")
            elif feat == "Term":
                change_lines.append(f"**{fl}:** {ov} months → {nv} months ({nv/12:.1f} years)")
            elif feat == "LowDoc":
                change_lines.append(f"**{fl}:** {'Minimal' if int(ov) else 'Full'} → {'Minimal' if int(nv) else 'Full docs'}")
            elif feat == "NewExist":
                change_lines.append(f"**{fl}:** {'Existing' if int(ov)==1 else 'New'} → {'Existing' if int(nv)==1 else 'New'}")
            else:
                change_lines.append(f"**{fl}:** {ov} → {nv}")

        st.markdown("\n".join(f"- 📌 {line}" for line in change_lines))

        # Safe amount + term insights
        amt = optimizer.get("amount_analysis")
        if amt and amt.get("max_safe_amount", 0) > 0:
            if amt.get("should_reduce"):
                st.warning(f"⚠️ **Safe borrowing limit:** {fmt_inr(amt['max_safe_amount'])}. "
                           f"You're asking for {fmt_inr(amt['requested_amount'])} — consider reducing.")
            elif amt.get("can_take_more"):
                st.info(f"💰 You could safely borrow up to **{fmt_inr(amt['max_safe_amount'])}** "
                        f"(you're asking for {fmt_inr(amt['requested_amount'])}). Room to grow!")

        term = optimizer.get("term_analysis")
        if term:
            st.info(f"📅 **Best repayment period:** {term['recommended_term']} months ({term['recommended_term_years']} years)")

    elif optimizer:
        st.success("🎯 Your loan structure is already well-optimized!")

    # ── Similar Businesses ──
    if similar:
        st.markdown("### 🏢 Similar Businesses")
        sr = similar["success_rate"]
        if similar["risk_vs_baseline"] == "above_average":
            bc = "#1b5e20"
        elif similar["risk_vs_baseline"] == "average":
            bc = "#e65100"
        else:
            bc = "#b71c1c"

        st.markdown(
            f'<div class="insight-strip" style="background:{bc};">'
            f'{similar["insight"]}</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Matched", similar["total_similar"])
        c2.metric("Success Rate", f"{sr*100:.0f}%")
        c3.metric("Baseline", f"{similar['baseline_success_rate']*100:.0f}%")

        if similar.get("similar_businesses"):
            with st.expander("👀 View Top 5 Similar Businesses"):
                for biz in similar["similar_businesses"]:
                    st.markdown(
                        f'{biz["outcome_emoji"]} **{biz["name"]}** ({biz["state"]}) — '
                        f'{biz["outcome"]} · {biz["employees"]} employees · '
                        f'{fmt_inr(biz["disbursement"])} · Match: {biz["similarity_score"]*100:.0f}%'
                    )

    # ── Government Schemes ──
    if schemes and schemes.get("schemes"):
        st.markdown("### 🏛️ सरकारी योजनाएं (Government Schemes)")
        st.caption("Commercial loan लेने से पहले देखिए — ये programmes आपके पैसे बचा सकते हैं:")

        for scheme in schemes["schemes"]:
            rel_badge = ("🟢 HIGH MATCH" if scheme["relevance"] == "high" else "🟡 PARTIAL MATCH")
            benefits = " · ".join(scheme["benefits"])
            max_inr = scheme.get("max_amount_inr", 0)
            if max_inr >= 10000000:
                max_str = f"₹{max_inr/10000000:.0f} Cr"
            elif max_inr >= 100000:
                max_str = f"₹{max_inr/100000:.0f}L"
            else:
                max_str = f"₹{max_inr/1000:.0f}K"

            st.markdown(
                f'<div class="scheme-card">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">'
                f'<strong>🇮🇳 {scheme["name"]}</strong>'
                f'<span style="font-size:0.8rem;">{rel_badge} · Up to {max_str}</span></div>'
                f'<p style="margin:0.3rem 0; color:#b0b0c0; font-size:0.9rem;">{scheme["description"]}</p>'
                f'<p style="margin:0.2rem 0; font-size:0.85rem; color:#a5d6a7;">{benefits}</p>'
                f'<a href="{scheme["url"]}" target="_blank" style="color:#64b5f6; font-size:0.85rem;">🔗 {scheme["url"]}</a>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── SHAP (collapsible) ──
    if shap_data:
        with st.expander("🔍 Technical: Why this rating? (SHAP Analysis)"):
            contribs = shap_data["feature_contributions"]
            sorted_feats = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            feat_names = [f[0] for f in sorted_feats]
            feat_vals = [f[1] for f in sorted_feats]

            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            colors = ["#ef5350" if v > 0 else "#66bb6a" for v in feat_vals]
            ax2.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1], height=0.6)
            ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
            ax2.set_xlabel("Impact on prediction", color='white', fontsize=10)
            ax2.set_title(f"Feature Contributions → {label}", color='white', fontsize=13, fontweight='bold')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            c1, c2 = st.columns(2)
            c1.success(f"**Strengths:** {', '.join(shap_data['top_negative_features']) or 'None'}")
            c2.error(f"**Risk Factors:** {', '.join(shap_data['top_positive_features']) or 'None'}")

    # Reset
    st.markdown("---")
    if st.button("🔄 Start New Assessment", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.extracted_features = None
        st.session_state.assessment_done = False
        st.rerun()


# ═══════════════════════════════════════════
# TAB 2: BATCH PROCESSING
# ═══════════════════════════════════════════

def render_batch():
    st.markdown("### 📁 Batch Processing")
    st.markdown("Upload a CSV file with loan applications to process them all at once.")

    template_cols = "Term,NoEmp,NewExist,CreateJob,RetainedJob,DisbursementGross,UrbanRural,RevLineCr,LowDoc,SBA_Appv,GrAppv"
    sample_rows = "84,5,1,2,3,100000,1,0,0,75000,100000\n36,2,2,0,1,50000,2,0,0,25000,50000\n"
    st.download_button("📥 Download Template CSV", template_cols + "\n" + sample_rows, "msme_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df_preview = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.markdown(f"**Preview:** {len(df_preview)} applications")
        st.dataframe(df_preview.head(), use_container_width=True, hide_index=True)

        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            uploaded.seek(0)
            with st.spinner(f"Processing {len(df_preview)} applications..."):
                start = time.time()
                result = api_call("FILE", "/predict/batch",
                                  files={"file": ("upload.csv", uploaded.getvalue(), "text/csv")})
                elapsed = time.time() - start

            if result:
                st.success(f"✅ Processed **{result['total_processed']}** applications in **{elapsed:.1f}s**")

                summary = result["summary"]
                cols = st.columns(5)
                for i, (lbl, count) in enumerate(summary.items()):
                    cols[i].metric(f"{LABEL_EMOJIS.get(lbl, '')} {lbl}", count)

                rows = []
                for r in result["results"]:
                    rows.append({
                        "ID": r["prediction_id"],
                        "Prediction": f"{LABEL_EMOJIS[r['predicted_label']]} {r['predicted_label']}",
                        "Confidence": f"{r['confidence']*100:.1f}%",
                        "Model": r["model_used"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# TAB 3: ANALYTICS
# ═══════════════════════════════════════════

def render_analytics():
    st.markdown("### 📈 Analytics Dashboard")
    data = api_call("GET", "/analytics")
    if not data:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", data["total_predictions"])
    c2.metric("Avg Confidence", f"{data['average_confidence']*100:.1f}%")
    c3.metric("Risk Classes", len(data.get("class_distribution", {})))

    dist = data.get("class_distribution", {})
    if dist:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        labels = list(dist.keys())
        values = list(dist.values())
        colors = [LABEL_COLORS.get(l, "#555") for l in labels]
        bars = ax.bar(labels, values, color=colors, width=0.5)
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


# ═══════════════════════════════════════════
# TAB 4: EXPERT MODE
# ═══════════════════════════════════════════

def render_expert_mode():
    st.markdown("### 🔧 Expert Mode — Direct Feature Input")
    st.caption("For bank officers and analysts with structured loan data.")

    with st.form("expert_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            term = st.number_input("Loan Term (months)", 0, 480, 84, step=6)
            disbursement = st.number_input("Disbursement Amount ($)", 1000, 5_000_000, 100000, step=5000,
                                           help=f"≈ {fmt_inr(100000)}")
            gr_appv = st.number_input("Gross Approved ($)", 1000, 5_000_000, 100000, step=5000)
            sba_appv = st.number_input("SBA Guaranteed ($)", 500, 4_000_000, 75000, step=5000)
        with c2:
            no_emp = st.number_input("Employees", 0, 500, 5, step=1)
            create_job = st.number_input("Jobs to Create", 0, 200, 2, step=1)
            retained_job = st.number_input("Jobs Retained", 0, 200, 3, step=1)
            new_exist = st.selectbox("Business Type", [1, 2], format_func=lambda x: {1: "Existing", 2: "New"}[x])
        with c3:
            urban_rural = st.selectbox("Location", [1, 2, 0], format_func=lambda x: {1: "Urban", 2: "Rural", 0: "Undefined"}[x])
            rev_line = st.selectbox("Revolving Credit", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
            low_doc = st.selectbox("Low Doc Loan", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])

        submitted = st.form_submit_button("🔍 Assess Viability", type="primary", use_container_width=True)

    if submitted:
        features = {
            "Term": term, "NoEmp": no_emp, "NewExist": new_exist,
            "CreateJob": create_job, "RetainedJob": retained_job,
            "DisbursementGross": disbursement, "UrbanRural": urban_rural,
            "RevLineCr": rev_line, "LowDoc": low_doc,
            "SBA_Appv": sba_appv, "GrAppv": gr_appv,
        }
        render_report_card(features)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏦 MSME Loan Readiness Coach</h1>
        <p>AI-powered loan assessment · अपने business की बात अपनी भाषा में करें · XGBoost + SHAP + Gemini</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Loan Coach", "📁 Batch Processing", "📈 Analytics", "🔧 Expert Mode"
    ])

    with tab1:
        render_chat_tab()
    with tab2:
        render_batch()
    with tab3:
        render_analytics()
    with tab4:
        render_expert_mode()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
        "MSME Loan Readiness Coach v2.1 · 🇮🇳 India-First · 897K businesses indexed · "
        "XGBoost + LightGBM · SHAP · Gemini 2.5</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
