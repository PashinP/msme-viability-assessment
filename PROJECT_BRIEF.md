# MSME Viability Assessment — Full Project Brief for Cursor Agent

> **Owner**: Pashin P | **Role**: AIML Engineer Portfolio Project  
> **Repo**: https://github.com/PashinP/msme-viability-assessment  
> **Live API**: https://msme-viability-assessment.onrender.com  
> **Tech Stack**: FastAPI backend + Streamlit frontend (upgrading to Vite/React)

---

## 1. WHAT THIS PROJECT IS

An **AI-powered MSME (Micro, Small & Medium Enterprise) Loan Readiness Coach** that:

1. Lets business owners describe their business in **plain language (English or Hindi)** via chat
2. Uses **LLM (Groq Llama 3.3 70B → Gemini 2.0 → offline fallback)** to extract 11 financial features from that conversation
3. Runs the extracted features through a trained **XGBoost model** (92% accuracy, trained on 899K U.S. SBA loan records) to output a **viability grade: Critical / At-Risk / Stable / Growing / Thriving**
4. Explains *why* using **SHAP feature contributions**
5. Tells users *how to improve* using a **counterfactual optimizer** (DiCE-inspired)
6. Shows **similar historical businesses** and their loan outcomes (KNN over 897K records)
7. Matches **Indian government schemes** (MUDRA, MSME, SVANidhi) to their profile
8. Generates a **downloadable PDF report**

The project is on the owner's AIML engineering resume. It must look genuinely impressive, technically deep, and visually stunning — not like a student Streamlit prototype.

---

## 2. CURRENT STATE (What Exists + What's Wrong)

### ✅ What Exists and Works Well

**Backend (FastAPI) — `backend/` directory, runs on Render.com:**
```
backend/
├── server.py          # 12 REST endpoints, auth, CORS, lazy-loading
├── engine.py          # XGBoost/LightGBM/SHAP prediction engine
├── chat_agent.py      # Multi-provider LLM (Groq → Gemini → offline rule-based)
├── optimizer.py       # Red flags, loan optimizer, government schemes
├── similar_engine.py  # KNN similarity over 897K SBA records
├── report_generator.py # 10-page PDF with charts (fpdf2)
├── database.py        # SQLAlchemy — full audit trail of every prediction
├── schemas.py         # Pydantic validation
└── prompts.py         # LLM system prompts
```

**API Endpoints (all require `X-API-Key: msme-dev-key-2024` header):**
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Single loan assessment |
| `/predict/batch` | POST | CSV bulk processing |
| `/explain` | POST | SHAP contributions |
| `/recommend` | POST | Counterfactual changes |
| `/optimize` | POST | Optimal loan structure |
| `/similar` | POST | KNN similar businesses |
| `/redflags` | POST | Structural risk flags |
| `/schemes` | POST | Government scheme matching |
| `/chat` | POST | LLM conversation |
| `/report` | POST | PDF generation |
| `/analytics` | GET | Historical stats |

**Prediction Input Schema (POST `/predict`, `/explain`, `/optimize`, `/similar`, `/redflags`, `/schemes`, `/report`):**
```json
{
  "Term": 84,               // Loan term in months (0-480)
  "NoEmp": 10,              // Number of employees (0-500)
  "NewExist": 1,            // 1=Existing business, 2=New startup
  "CreateJob": 3,           // Jobs this loan will create (0-200)
  "RetainedJob": 10,        // Jobs that will be retained (0-200)
  "DisbursementGross": 150000, // Loan amount in USD
  "UrbanRural": 1,          // 1=Urban, 2=Rural, 0=Undefined
  "RevLineCr": 0,           // Revolving line of credit (0 or 1)
  "LowDoc": 0,              // Low documentation loan (0 or 1)
  "SBA_Appv": 112500,       // SBA guaranteed amount in USD (~75% of loan)
  "GrAppv": 150000          // Gross approved amount in USD
}
```

**Prediction Output:**
```json
{
  "predicted_class": 3,
  "predicted_label": "Growing",
  "confidence": 0.87,
  "probabilities": {
    "Critical": 0.01, "At-Risk": 0.03, "Stable": 0.09,
    "Growing": 0.87, "Thriving": 0.00
  },
  "model_used": "XGBoost",
  "prediction_id": 42
}
```

**Models (stored in `models/`):**
- `xgb_mc.pkl` — XGBoost, 92.4% accuracy, 8MB, **primary**
- `lgbm_mc.pkl` — LightGBM, 92.1% accuracy, 2.6MB, **fallback**
- `scaler_mc.pkl` — StandardScaler
- `metadata.json` — feature names, label map

**Notebook:** `notebooks/msme_viability_analysis.ipynb` — 54 cells covering full ML pipeline

### ❌ What's Wrong / Why It's Not Impressive

1. **Frontend is basic Streamlit** — static matplotlib charts, no animations, no interactivity, looks like every ML student project
2. **No live What-If simulator** — user has to submit a form and wait; no real-time feedback
3. **SHAP chart is a static bar chart** — not interactive, no tooltips, not explanatory to non-technical users
4. **Radar chart is matplotlib** — static, pixelated, not interactive
5. **Grade card is plain colored div** — no animation on reveal
6. **Probability bars don't exist** — just a grade letter, no visualization of the full 5-class probability
7. **Chat UI is standard Streamlit chat** — no typing indicator, no personality, no visual polish
8. **Analytics is a basic bar chart** — no time series, no distribution pie, no filtering
9. **No landing/hero section** — app just starts with tabs, no wow moment
10. **Not deployable as a proper web app** — Streamlit URLs are ugly, no custom domain look

---

## 3. THE VISION — What It Should Become

This should feel like a **premium fintech product**, not a student project. Reference: think [Stripe's dashboard](https://stripe.com), [Linear's UI](https://linear.app), or [Vercel's dark theme](https://vercel.com). The goal is for someone to open this and say "this person clearly knows what they're doing."

### 3.1 User Journey (What The Experience Should Feel Like)

```
LANDING PAGE
│
│  ── Animated hero with live stats (897K businesses indexed, 92% accuracy, 5 viability grades)
│  ── Clear CTA: "Assess Your Loan Readiness" → goes to Chat Coach
│  ── Short visual explainer of how it works (3-step flow)
│
├── CHAT COACH (main flow)
│   │
│   │  ── Full-screen chat interface, dark, premium
│   │  ── Typing indicator animation when AI is thinking
│   │  ── Message bubbles with smooth slide-in animation
│   │  ── Voice input button
│   │  ── When extraction is complete → smooth transition to results
│   │
│   └── ASSESSMENT RESULT (after chat completes)
│       │
│       ├── GRADE REVEAL — animated, dramatic (like a card flip or count-up)
│       │   Grade letter (A/B/C/D/F), viability label, confidence %, loan amount in ₹
│       │
│       ├── PROBABILITY BARS — animated horizontal bars for all 5 classes
│       │   Shows: Critical ▓░░░░ 3% | At-Risk ▓░░░░ 5% | Stable ▓▓░░░ 20% | Growing ▓▓▓▓░ 65% | Thriving ░░░░░ 7%
│       │
│       ├── RADAR CHART — interactive Plotly/Recharts, hover tooltips
│       │   Dimensions: Loan Term, Employment, Business Maturity, SBA Guarantee, Location, Documentation
│       │
│       ├── SHAP WATERFALL CHART — interactive, color-coded, explained in plain English
│       │   Red bars = hurts your score | Green bars = helps your score
│       │   Hovering shows: "This feature pushes your grade DOWN because..."
│       │
│       ├── RED FLAGS — color-coded severity cards (high/medium/low)
│       │
│       ├── HOW TO IMPROVE — counterfactual recommendations panel
│       │   "Change Loan Term from 84 months → 120 months to upgrade: Stable → Growing"
│       │
│       ├── WHAT-IF SIMULATOR — LIVE SLIDERS (the most impressive feature)
│       │   Sliders for key features → prediction updates in real-time without page reload
│       │   Shows grade changing as user drags sliders
│       │
│       ├── SIMILAR BUSINESSES — horizontal scrollable cards
│       │   Each card: business name, state, outcome (✅ paid/❌ defaulted), similarity score
│       │
│       ├── GOVERNMENT SCHEMES — visually rich cards with scheme logos/icons
│       │
│       └── DOWNLOAD PDF — prominent button
│
├── EXPERT MODE (for technical users / bank officers)
│   ── Side-by-side layout: input form LEFT, live result RIGHT
│   ── No submit button needed — result updates as you type (debounced 500ms)
│   ── Full sliders + number inputs
│
├── BATCH UPLOAD
│   ── Drag-and-drop CSV upload
│   ── Live processing table with row-by-row results
│   ── Export results as CSV
│
└── ANALYTICS DASHBOARD
    ── Total predictions, avg confidence, grade distribution
    ── Interactive Plotly charts: pie chart (class distribution), time series (predictions over time)
    ── Recent predictions table
```

---

## 4. TECHNOLOGY DECISIONS

### 4A. Option 1: Vite + React (Recommended for Full Rebuild)
**Use this if doing a proper new frontend.**
- Vite for build tooling
- React for component tree
- Recharts or Plotly.js for charts
- Framer Motion for animations
- Vanilla CSS (no Tailwind) for full control
- Deployed to Vercel (free)
- Backend stays on Render

### 4B. Option 2: Improved Streamlit (Faster, Less Impressive)
**Use this if staying with Streamlit.**
- Replace matplotlib with `plotly` for interactive charts
- Add custom CSS for animations using `st.markdown` with `<style>` tags
- Use `streamlit-extras` for better components
- Add `st.session_state` for What-If simulator (sliders that update on change)
- Stays on Streamlit Community Cloud

**The owner wants Option 4A (React + Vite) — build a proper web app.**

---

## 5. FRONTEND ARCHITECTURE (React + Vite)

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx
    ├── App.jsx                    # Router, layout
    ├── index.css                  # Global design system (CSS variables, fonts)
    │
    ├── pages/
    │   ├── Landing.jsx            # Hero + explainer
    │   ├── ChatCoach.jsx          # Conversational assessment flow
    │   ├── AssessmentResult.jsx   # Full results page
    │   ├── ExpertMode.jsx         # Side-by-side form + live results
    │   ├── BatchUpload.jsx        # CSV batch processing
    │   └── Analytics.jsx         # Historical dashboard
    │
    ├── components/
    │   ├── GradeCard.jsx          # Animated grade reveal
    │   ├── ProbabilityBars.jsx    # 5-class animated bars
    │   ├── RadarChart.jsx         # Plotly radar
    │   ├── ShapWaterfall.jsx      # SHAP waterfall chart
    │   ├── WhatIfSimulator.jsx    # Live slider simulator
    │   ├── RedFlagsPanel.jsx      # Color-coded risk flags
    │   ├── OptimizationPanel.jsx  # Counterfactual improvements
    │   ├── SimilarBusinesses.jsx  # KNN result cards
    │   ├── GovernmentSchemes.jsx  # Scheme matching cards
    │   ├── ChatMessage.jsx        # Individual message bubble
    │   ├── TypingIndicator.jsx    # Animated dots
    │   ├── BatchTable.jsx         # Results table
    │   └── Navbar.jsx             # Navigation
    │
    └── utils/
        ├── api.js                 # All API calls (axios)
        └── formatters.js          # INR formatting, grade colors, etc.
```

---

## 6. DESIGN SYSTEM

### Color Palette
```css
:root {
  /* Backgrounds */
  --bg-primary:    #0a0a0f;   /* near black */
  --bg-secondary:  #111118;   /* card background */
  --bg-tertiary:   #1a1a28;   /* elevated card */
  --bg-glass:      rgba(255, 255, 255, 0.04);  /* glassmorphism */

  /* Brand */
  --accent-purple: #7c3aed;   /* primary brand */
  --accent-blue:   #3b82f6;   /* secondary */
  --accent-cyan:   #06b6d4;   /* highlights */

  /* Grade Colors */
  --grade-critical: #ef4444;  /* red */
  --grade-atrisk:   #f97316;  /* orange */
  --grade-stable:   #22c55e;  /* green */
  --grade-growing:  #3b82f6;  /* blue */
  --grade-thriving: #a855f7;  /* purple */

  /* Text */
  --text-primary:   #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted:     #475569;

  /* Borders */
  --border:         rgba(255, 255, 255, 0.08);
  --border-active:  rgba(124, 58, 237, 0.5);

  /* Gradients */
  --gradient-hero:  linear-gradient(135deg, #0f0c29, #302b63, #24243e);
  --gradient-card:  linear-gradient(135deg, #111118, #1a1a28);
  --gradient-purple: linear-gradient(135deg, #7c3aed, #4f46e5);
}
```

### Typography
```css
/* Import in index.css */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

body { font-family: 'Inter', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }
```

### Key Design Principles
1. **Glassmorphism cards** — `backdrop-filter: blur(16px)`, semi-transparent backgrounds, subtle borders
2. **Smooth animations** — 300ms ease transitions on everything, stagger children
3. **Depth with shadows** — `box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4)`
4. **Hover states** — every interactive element has a visible hover state
5. **Loading skeletons** — never show empty states, always show shimmering placeholders
6. **Micro-animations** — numbers count up on load, bars animate in from left, grade flips in

---

## 7. COMPONENT SPECIFICATIONS

### 7.1 Grade Card Component
```
┌─────────────────────────────────────────────┐
│                                             │
│   ANIMATION: card flips in from back        │
│                                             │
│         ╔═══╗                               │
│         ║ B ║   ← Giant grade letter        │
│         ╚═══╝   (color matches grade)       │
│                                             │
│      🔵 Growing                             │
│      87% confidence                         │
│      Loan: ₹12.5L                          │
│                                             │
└─────────────────────────────────────────────┘
Animation: scale from 0.5 → 1.0, opacity 0 → 1, 400ms ease-out
```

### 7.2 Probability Bars
```
Critical  ████░░░░░░░░░░░░  3%
At-Risk   ████░░░░░░░░░░░░  5%
Stable    ████████░░░░░░░░  20%
Growing   █████████████░░░  65%   ← highlighted (predicted class)
Thriving  ████░░░░░░░░░░░░  7%

Animation: bars grow from width 0% to final width, 600ms staggered
Each bar colored with grade color
```

### 7.3 SHAP Waterfall Chart
```
Feature contributions to prediction "Growing":

Base value:           ─────────────── 0.5
SBA_Appv (high):     ──────────────────────── +0.23  (green)
Term (long):          ───────────── +0.18             (green)
NoEmp (small):       ──────  -0.12                    (red)
LowDoc (yes):        ──  -0.08                        (red)
NewExist (existing): ─────────── +0.11               (green)
...
Final prediction:     ──────────────────────────────── 0.87

Interactive: hover shows tooltip "Loan Term of 84 months pushed 
your grade UP by 18% because longer terms reduce monthly burden"
```

### 7.4 What-If Simulator (MOST IMPORTANT FEATURE)
```
┌─────────────────────────────────────────────────────────┐
│  🎮 What-If Simulator                                    │
│  Drag the sliders and watch your grade update live       │
│                                                          │
│  Loan Term (months)                                      │
│  ├── 12 ─────────────────●──────────── 240 ──┤  84 mo  │
│                                                          │
│  Loan Amount (₹)                                         │
│  ├── 1L ──────────────●───────────── 5Cr ──┤  ₹12.5L  │
│                                                          │
│  Employees                                               │
│  ├── 0 ─────────●─────────────────── 50 ──┤  5       │
│                                                          │
│  SBA Guarantee %                                         │
│  ├── 50% ────────────────●──────── 90% ──┤  75%     │
│                                                          │
│  ┌─────────────────────┐                                 │
│  │   Current: 🔵 Growing  87%    │  (updates live)      │
│  └─────────────────────┘                                 │
└─────────────────────────────────────────────────────────┘

Behavior: debounce 300ms → call /predict → update grade display
Show grade transition: "Stable → Growing" if slider change upgrades
```

### 7.5 Chat Interface
```
┌───────────────────────────────────────────────────┐
│  💬 MSME Loan Coach                               │
│  Powered by Groq Llama 3.3 70B                    │
├───────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │ 👤 User bubble (right-aligned, blue)         │  │
│  │ "I run a grocery shop in Mumbai with        │  │
│  │  5 workers and need ₹10 lakh"              │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │ 🤖 AI bubble (left-aligned, dark glass)     │  │
│  │ "Great! A grocery shop in Mumbai...         │  │
│  │  How long has it been running?"             │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌────┐   Typing indicator (3 animated dots)   │  │
│  │ 🤖 │   ● ● ●                                │  │
│  └────┘                                          │  │
│                                                   │
├───────────────────────────────────────────────────┤
│  🎤  │  Type your message here...          ➤  │  │
└───────────────────────────────────────────────────┘
```

### 7.6 Similar Businesses — Horizontal Cards
```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ ✅ PAID FULL  │ │ ✅ PAID FULL  │ │ ❌ DEFAULTED  │
│              │ │              │ │              │
│ SHARMA FOODS │ │ KUMAR RETAIL │ │ PATEL BAKERY │
│ Maharashtra  │ │ Gujarat      │ │ Rajasthan    │
│              │ │              │ │              │
│ 5 employees  │ │ 8 employees  │ │ 3 employees  │
│ ₹8.3L loan   │ │ ₹11.2L loan  │ │ ₹6.1L loan  │
│              │ │              │ │              │
│ Match: 94%   │ │ Match: 91%   │ │ Match: 89%   │
└──────────────┘ └──────────────┘ └──────────────┘
Scrollable horizontally, hover shows more details
```

---

## 8. LANDING PAGE SPEC

```
┌────────────────────────────────────────────────────────────────────┐
│  NAVBAR: Logo | Chat Coach | Expert Mode | Batch | Analytics | API │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  HERO SECTION (gradient bg, animated)                              │
│                                                                    │
│  🏦 MSME Loan Readiness Coach                                      │
│  AI-powered financial assessment for Indian small businesses       │
│                                                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │  897K+       │ │  92%         │ │  5           │              │
│  │  Businesses  │ │  Accuracy    │ │  Viability   │              │
│  │  Indexed     │ │  XGBoost     │ │  Grades      │              │
│  └──────────────┘ └──────────────┘ └──────────────┘              │
│         (numbers animate/count-up on page load)                    │
│                                                                    │
│  [ 💬 Start Assessment ]  [ 📚 View Research ]                     │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  HOW IT WORKS (3 steps, with icons)                                │
│                                                                    │
│  1️⃣ Chat naturally          2️⃣ AI analyzes          3️⃣ Get grade  │
│  "Tell us about your       "XGBoost + SHAP          "Growing 🔵   │
│   business in Hindi         explains every           + roadmap to  │
│   or English"               decision"                Thriving 🟣"  │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  VIABILITY GRADES EXPLAINER                                        │
│                                                                    │
│  F Critical | D At-Risk | C Stable | B Growing | A Thriving       │
│  ████████████████████████████████████████████████████             │
│  (colored gradient bar from red to purple)                         │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  TECH STACK BADGES (shows engineering depth)                       │
│  XGBoost · LightGBM · SHAP · FastAPI · Groq · Gemini · SQLite     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. WHAT THE API RETURNS (FULL RESPONSE SHAPES)

### `/explain` response:
```json
{
  "predicted_class": 3,
  "predicted_label": "Growing",
  "feature_contributions": {
    "SBA_Appv": 0.23,
    "Term": 0.18,
    "GrAppv": 0.12,
    "NoEmp": -0.12,
    "LowDoc": -0.08,
    "NewExist": 0.11,
    "DisbursementGross": 0.09,
    "CreateJob": 0.06,
    "RetainedJob": 0.04,
    "UrbanRural": -0.03,
    "RevLineCr": -0.02
  },
  "top_positive_features": ["SBA_Appv", "Term", "NewExist"],
  "top_negative_features": ["NoEmp", "LowDoc", "UrbanRural"]
}
```

### `/redflags` response:
```json
{
  "flags": [
    {
      "flag": "Low Documentation Risk",
      "emoji": "⚠️",
      "severity": "high",
      "explanation": "Low-doc loans have 2.3x higher default rates in your loan size range.",
      "suggestion": "Submit full financial documents to improve your profile significantly."
    }
  ],
  "total_flags": 1
}
```

### `/optimize` response:
```json
{
  "original_prediction": { "predicted_label": "Stable", "confidence": 0.72 },
  "optimized_prediction": { "predicted_label": "Growing", "confidence": 0.85 },
  "improvement": 1,
  "changes": [
    { "feature": "Term", "original": 84, "optimized": 120 },
    { "feature": "SBA_Appv", "original": 75000, "optimized": 100000 }
  ],
  "amount_analysis": {
    "requested_amount": 100000,
    "max_safe_amount": 415000,
    "can_take_more": true,
    "should_reduce": false
  },
  "term_analysis": {
    "recommended_term": 240,
    "recommended_term_years": 20.0
  }
}
```

### `/similar` response:
```json
{
  "total_similar": 50,
  "success_count": 38,
  "default_count": 12,
  "success_rate": 0.76,
  "baseline_success_rate": 0.82,
  "risk_vs_baseline": "below_average",
  "similar_businesses": [
    {
      "rank": 1,
      "name": "SHARMA FOODS",
      "state": "CA",
      "outcome": "Paid in Full",
      "outcome_emoji": "✅",
      "similarity_score": 0.94,
      "employees": 5,
      "disbursement": 83000
    }
  ],
  "insight": "🔴 Out of 50 similar businesses, 38 (76%) successfully repaid...",
  "dataset_size": 897164
}
```

### `/chat` request/response:
```json
// Request
{ "messages": [{"role": "user", "content": "I have a bakery in Mumbai..."}] }

// Response
{
  "response": "Great! How long has your bakery been running?",
  "extraction_complete": false,
  "features_extracted": null,
  "validation_warnings": []
}

// When complete:
{
  "response": "Here's what I understand...\n```json\n{...11 features...}\n```",
  "extraction_complete": true,
  "features_extracted": { "Term": 84, "NoEmp": 5, ... },
  "validation_warnings": []
}
```

---

## 10. DEPLOYMENT TARGETS

| Component | Platform | URL Pattern |
|---|---|---|
| FastAPI Backend | Render.com (free) | `https://msme-viability-assessment.onrender.com` |
| React Frontend | Vercel (free) | `https://msme-viability.vercel.app` |

**Important Render constraint:** Free tier has 512MB RAM limit. The backend already handles this with lazy-loading of heavy components.

**CORS:** Backend already has `allow_origins=["*"]` — needs to be tightened to the Vercel domain once deployed.

---

## 11. FILE LOCATIONS IN REPO

```
msme-viability-assessment/        ← repo root
├── app.py                         ← OLD Streamlit app (keep for reference)
├── backend/                       ← FastAPI server (DO NOT TOUCH unless needed)
├── models/                        ← Trained ML models (DO NOT TOUCH)
├── notebooks/msme_viability_analysis.ipynb ← Training pipeline
├── assets/                        ← Screenshots (for README)
├── requirements.txt               ← Production deps
├── requirements-dev.txt           ← Dev deps
├── .env.example                   ← Copy to .env and fill keys
└── README.md                      ← Already updated, professional
```

**Create the new frontend at:** `frontend/` in the repo root.

---

## 12. ENVIRONMENT VARIABLES

```bash
# Backend needs these (already on Render as env vars):
MSME_API_KEY=msme-dev-key-2024
GROQ_API_KEY=<your key>
GEMINI_API_KEY=<your key>

# Frontend needs this:
VITE_API_URL=https://msme-viability-assessment.onrender.com
VITE_API_KEY=msme-dev-key-2024
```

---

## 13. NON-NEGOTIABLES

1. **Dark theme always** — no light mode
2. **All amounts shown in both USD and ₹** — use `83` as conversion rate
3. **Fully responsive** — must work on mobile (375px width)
4. **No placeholder text/images** — every element must have real content or a proper loading state
5. **Graceful degradation** — if the Render API is sleeping (cold start), show a friendly "Server is waking up, 30 seconds..." message with a spinner, NOT an error
6. **SHAP chart must be explained in plain English** — not just numbers, must have tooltips that say "This means..."
7. **Grade card must be the most visually dramatic element** — this is what users remember

---

## 14. SUGGESTED INITIAL PROMPTS FOR CURSOR

### Prompt 1 — Setup & Foundation
```
I am building a premium React + Vite frontend for an AI-powered MSME Loan 
Readiness assessment tool. The backend is a FastAPI server already running at 
https://msme-viability-assessment.onrender.com with 12 REST endpoints.

Read the PROJECT_BRIEF.md file in this repo for full context on the project, 
API responses, design system, and component specifications.

Start by:
1. Scaffolding a new Vite + React project in the `frontend/` directory
2. Setting up the design system in `src/index.css` — use the exact CSS 
   variables defined in the brief (dark theme, purple accent, glassmorphism)
3. Setting up React Router with routes for: /, /chat, /result, /expert, /batch, /analytics
4. Creating the Navbar component
5. Creating the Landing page with the hero section and animated stat counters

DO NOT use Tailwind. Use vanilla CSS with CSS variables. Use Inter font from Google Fonts.
```

### Prompt 2 — Chat Coach
```
Build the Chat Coach page for the MSME Loan Readiness app.

Read PROJECT_BRIEF.md for the exact API spec for `/chat` endpoint.

Requirements:
- Full-screen dark chat interface
- Message bubbles: user messages right-aligned (blue), AI messages left-aligned (glass)
- Animated typing indicator (3 bouncing dots) while API call is in flight
- Each message slides in from bottom with 200ms stagger
- Voice input button (Web Speech API)
- When extraction_complete=true in API response, transition smoothly to /result page 
  passing the extracted features as state

The /chat endpoint expects: POST { "messages": [...history] }
Returns: { "response": string, "extraction_complete": bool, "features_extracted": object|null }
```

### Prompt 3 — Assessment Result Page
```
Build the Assessment Result page — this is the most important page of the app.

Read PROJECT_BRIEF.md for all component specs and API response shapes.

This page receives `features` (the 11-field object) via React Router state.
It needs to call these 6 API endpoints simultaneously on load:
- POST /predict → grade, confidence, probabilities
- POST /explain → SHAP feature contributions  
- POST /optimize → counterfactual improvements
- POST /redflags → risk warnings
- POST /schemes → government scheme matches
- POST /similar → KNN peer businesses

Build these components in order:
1. GradeCard — animated reveal (scale + fade in, 400ms)
2. ProbabilityBars — 5 animated horizontal bars, staggered entrance
3. RadarChart — using Plotly.js, interactive with hover tooltips
4. ShapWaterfallChart — horizontal bar chart, red=negative, green=positive, 
   hover tooltips explaining each feature in plain English
5. WhatIfSimulator — sliders for Term, DisbursementGross, NoEmp, SBA_Appv; 
   calls /predict on slider change (debounced 300ms), live grade update
6. RedFlagsPanel — severity-coded cards
7. SimilarBusinesses — horizontal scrollable card row
8. GovernmentSchemes — rich scheme cards

Show loading skeletons for each section while its API call is in flight.
```

### Prompt 4 — Expert Mode + Batch + Analytics
```
Build three remaining pages for the MSME app:

1. EXPERT MODE page:
   - Left panel: 11-field input form (sliders + number inputs for the 11 API features)
   - Right panel: live assessment result (updates on every input change, debounced 500ms)
   - Shows grade, probability bars, and top 3 SHAP contributors live
   - No submit button — it's always live

2. BATCH UPLOAD page:
   - Drag-and-drop CSV upload (uses POST /predict/batch)
   - Show CSV template download button
   - Process and show results in a sortable table
   - Color-code rows by grade
   - Export results as CSV

3. ANALYTICS DASHBOARD page:
   - Fetches GET /analytics
   - Shows: total predictions counter, avg confidence, grade distribution
   - Plotly pie chart for grade distribution
   - Bar chart for predictions count by grade
   - Recent predictions table

All pages must match the dark glassmorphism design system from PROJECT_BRIEF.md
```

### Prompt 5 — Deployment
```
Deploy the React frontend to Vercel and update the backend CORS settings.

1. In the `frontend/` directory, ensure `vite.config.js` is production-ready
2. Create `frontend/.env.production` with VITE_API_URL pointing to the Render backend
3. Create `vercel.json` at the root of `frontend/` for SPA routing
4. Update `backend/server.py` CORS middleware to allow the Vercel domain in addition to *
5. Update `README.md` with the live Vercel URL and a new demo GIF/screenshot
6. Create a GitHub Actions workflow at `.github/workflows/deploy.yml` that:
   - On push to main, runs `cd frontend && npm run build`
   - Vercel handles auto-deploy from the GitHub connection
```

---

## 15. QUALITY CHECKLIST (VERIFY BEFORE DONE)

- [ ] Landing page hero has animated stat counters
- [ ] Chat has typing indicator animation
- [ ] Grade reveal is animated (not instant)
- [ ] Probability bars animate from 0% to final width
- [ ] SHAP chart is interactive with tooltips
- [ ] Radar chart is interactive Plotly (not matplotlib)
- [ ] What-If simulator updates grade live without page reload
- [ ] Similar businesses shows scrollable horizontal cards
- [ ] Government schemes cards look premium
- [ ] Expert mode shows live results without submit button
- [ ] All amounts shown in ₹ (Indian format)
- [ ] All loading states show skeletons, not spinners
- [ ] Graceful API cold-start handling (friendly 30-second message)
- [ ] Mobile responsive at 375px
- [ ] No console errors
- [ ] Deployed to Vercel with live URL
