<div align="center">

# 🏦 MSME Viability Assessment System

### AI-powered loan risk stratification with conversational intelligence, SHAP explainability, and medical-grade PDF reporting.

[![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.2-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-blueviolet?style=flat-square)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[**Live Demo (Coming Soon)**](#) · [**API Docs (Coming Soon)**](#) · [**Research Notebook**](notebooks/msme_viability_analysis.ipynb)

</div>

---

## 🎯 The Problem

Over **63 million MSMEs** in India face a critical challenge: banks reject ~80% of loan applications not due to lack of creditworthiness, but due to **information asymmetry**. Businesses don't understand *why* they're rejected or *how* to improve their profile before applying.

This system solves that by giving founders an honest, data-driven, and highly actionable assessment of their loan viability—acting as an AI "Loan Coach" before they ever step foot in a bank.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 💬 **Conversational Assessment** | Natural language chat—founders describe their situation in their own words. LLM (Llama 3.3) extracts financial features automatically. |
| 🎯 **Diagnostic Scoring Engine** | XGBoost classifies applications and breaks down exactly *where* the application is weak (Repayment Capacity, Stability, etc). |
| 🔍 **SHAP Explainability** | Feature-level contribution analysis explains *exactly why* the model gave that grade, building trust with the user. |
| 📝 **Actionable Prescriptions** | Generates numbered, prioritized action plans (e.g., "Reduce DTI to <50% by increasing term length"). |
| 📄 **Business Health Report** | Downloads a highly professional 9-page PDF report with visual gauges, financial deep dives, and a pre-filled Draft Loan Application to take to the bank. |
| 🏛️ **Government Scheme Matching** | Automatically surfaces relevant CGTMSE, MUDRA, and SVANidhi schemes based on the profile. |

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Client [Frontend - React + Vite]
        UI[Dashboard UI]
        Chat[Conversational Chat Panel]
        Charts[Radar / SHAP Visualizations]
        UI <--> Chat
        UI <--> Charts
    end

    subgraph API [Backend - FastAPI]
        Router[API Router]
        Agent[LLM Chat Agent]
        Report[PDF Generator]
        
        Router --> Agent
        Router --> Report
    end

    subgraph ML [Machine Learning Core]
        Score[Scoring Engine]
        XGB[XGBoost Model]
        SHAP[SHAP Explainer]
        Sim[KNN Similarity Engine]
        Prescribe[Prescription Engine]
        
        Score --> XGB
        Score --> SHAP
        Score --> Sim
        Score --> Prescribe
    end

    Client <-->|REST / JSON| API
    API <--> ML
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install
```bash
git clone https://github.com/PashinP/msme-viability-assessment.git
cd msme-viability-assessment
```

### 2. Run the Backend (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn server:app --reload --port 8000
```

### 3. Run the Frontend (React + Vite)
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
