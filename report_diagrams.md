# MSME Viability Assessment — Report Diagrams

All diagrams below are in Mermaid format. Use these to generate proper diagrams for the Word document.

---

## Figure 1: System Architecture Overview

```mermaid
graph TB
    subgraph Frontend["Tier 2 — Streamlit Frontend"]
        UI1["Single Assessment Tab"]
        UI2["Batch Processing Tab"]
        UI3["Analytics Dashboard Tab"]
    end

    subgraph Backend["Tier 1 — FastAPI Backend"]
        API["REST API Layer<br/>(API Key Auth + Pydantic Validation)"]
        
        subgraph ML["ML Engine"]
            XGB["XGBoost Classifier"]
            LGB["LightGBM Classifier"]
            Scaler["StandardScaler<br/>(Cached)"]
        end

        subgraph XAI["Explainability Engine"]
            SHAP["TreeSHAP Explainer<br/>(Cached Singleton)"]
        end

        subgraph CF["Counterfactual Engine"]
            CFE["Deterministic Grid-Search<br/>Phase 1: Single-Feature<br/>Phase 2: Pairwise"]
        end

        subgraph DB["Persistence Layer"]
            SQLite["SQLite Database<br/>(SQLAlchemy ORM)"]
            AuditLog["Audit Log<br/>(Timestamp, Input, Output,<br/>SHAP Values, Recommendations)"]
        end
    end

    UI1 -->|"HTTP POST /predict"| API
    UI1 -->|"HTTP POST /explain"| API
    UI1 -->|"HTTP POST /recommend"| API
    UI2 -->|"HTTP POST /predict/batch"| API
    UI3 -->|"HTTP GET /history"| API

    API --> Scaler --> ML
    API --> SHAP
    API --> CFE
    ML --> SQLite
    SHAP --> SQLite
    CFE --> SQLite
    SQLite --> AuditLog
```

---

## Figure 2: Use Case Diagram

```mermaid
graph LR
    LO(("Loan Officer<br/>(Primary User)"))
    BA(("Batch Analyst"))
    SA(("System Admin"))
    EXT(("External System<br/>(Bank CRM / Core)"))

    subgraph System["MSME Viability Assessment System"]
        UC1["Submit Single<br/>Enterprise Assessment"]
        UC2["View Viability<br/>Prediction Result"]
        UC3["View SHAP<br/>Explanation"]
        UC4["View Counterfactual<br/>Recommendations"]
        UC5["Upload Batch<br/>CSV File"]
        UC6["View Portfolio<br/>Viability Distribution"]
        UC7["View Prediction<br/>History / Audit Logs"]
        UC8["API Integration<br/>(REST Endpoints)"]
        UC9["Manage API Keys"]
    end

    LO --- UC1
    LO --- UC2
    LO --- UC3
    LO --- UC4
    LO --- UC7

    BA --- UC5
    BA --- UC6
    BA --- UC7

    SA --- UC9
    SA --- UC7

    EXT --- UC8

    UC1 -.->|"includes"| UC2
    UC2 -.->|"includes"| UC3
    UC2 -.->|"includes"| UC4
    UC5 -.->|"includes"| UC6
```

---

## Figure 3: Activity Diagram — Prediction and Recommendation Workflow

```mermaid
flowchart TD
    Start(["Start"]) --> Open["User Opens Streamlit Dashboard"]
    Open --> Input["User Enters 11 Enterprise Features<br/>(Term, NoEmp, GrAppv, etc.)"]
    Input --> Validate{"Pydantic<br/>Validation<br/>Passed?"}

    Validate -->|"No"| Error["Display Validation<br/>Error Message"]
    Error --> Input

    Validate -->|"Yes"| Scale["Preprocess: StandardScaler<br/>Transforms Input Features"]
    Scale --> Predict["XGBoost / LightGBM<br/>Generates Viability Prediction"]
    Predict --> Class["Output: Predicted Class<br/>(Critical / At-Risk / Stable / Growing / Thriving)<br/>+ Class Probabilities"]

    Class --> SHAP["TreeSHAP Computes<br/>Feature Attributions"]
    SHAP --> SHAPOut["Output: Top 3 Risk-Amplifying<br/>+ Top 3 Risk-Mitigating Features"]

    SHAPOut --> CFCheck{"Current Class<br/>= Thriving?"}
    CFCheck -->|"Yes"| NoCF["No Recommendation Needed<br/>(Already Best Class)"]
    CFCheck -->|"No"| CF["Counterfactual Engine:<br/>Search Minimal Changes to<br/>Improve Class by One Level"]

    CF --> Phase1{"Phase 1:<br/>Single-Feature<br/>Solution Found?"}
    Phase1 -->|"Yes"| CFOut["Output: Recommended<br/>Feature Change + New Class"]
    Phase1 -->|"No"| Phase2["Phase 2: Pairwise<br/>Feature Combinations"]
    Phase2 --> CFOut

    NoCF --> Log["Log Full Record to SQLite<br/>(Input, Output, SHAP, CF)"]
    CFOut --> Log

    Log --> Display["Display Results on Dashboard:<br/>• Viability Class & Confidence<br/>• SHAP Waterfall Chart<br/>• Recommended Actions"]
    Display --> End(["End"])
```

---

## Figure 4: Composite Health Score Construction and Five-Class Discretization

```mermaid
flowchart LR
    subgraph Input["Raw Loan Record"]
        MIS["MIS_Status<br/>(PIF / CHGOFF)"]
        T["Term<br/>(months)"]
        Jobs["CreateJob +<br/>RetainedJob"]
        Loan["GrAppv<br/>(approved $)"]
    end

    subgraph Scoring["Component Scores"]
        S1["S_outcome<br/>= PIF ? 40 : 0<br/>(max 40 pts)"]
        S2["S_term<br/>= min(Term,240)/240 × 20<br/>(max 20 pts)"]
        S3["S_job<br/>= min(Jobs,50)/50 × 20<br/>(max 20 pts)"]
        S4["S_loan<br/>= min(GrAppv,500k)/500k × 20<br/>(max 20 pts)"]
    end

    subgraph Health["Health Score"]
        H["H = S_outcome + S_term<br/>+ S_job + S_loan<br/>(0 to 100)"]
    end

    subgraph Classes["Five-Class Taxonomy"]
        C0["Critical (0)<br/>H ∈ [0, 25)"]
        C1["At-Risk (1)<br/>H ∈ [25, 40)"]
        C2["Stable (2)<br/>H ∈ [40, 60)"]
        C3["Growing (3)<br/>H ∈ [60, 75)"]
        C4["Thriving (4)<br/>H ∈ [75, 100]"]
    end

    MIS --> S1
    T --> S2
    Jobs --> S3
    Loan --> S4

    S1 --> H
    S2 --> H
    S3 --> H
    S4 --> H

    H --> C0
    H --> C1
    H --> C2
    H --> C3
    H --> C4
```

---

## Figure 5: Counterfactual Recommendation Algorithm Flowchart

```mermaid
flowchart TD
    Start(["Input: Enterprise Features X,<br/>Current Class C"]) --> Target["Set Target Class = C + 1"]
    Target --> Check{"C = Thriving<br/>(Class 4)?"}
    Check -->|"Yes"| NoRec["Return: No Recommendation<br/>(Already Best Class)"]
    Check -->|"No"| Mutable["Identify 6 Mutable Features:<br/>Term, DisbursementGross, SBA_Appv,<br/>GrAppv, CreateJob, RetainedJob"]

    Mutable --> P1["PHASE 1: Single-Feature Perturbation"]
    P1 --> Loop1["For each mutable feature f:<br/>Test grid of changes<br/>(e.g., Term: +12, +24, +36, +48, +60)"]
    Loop1 --> Pred1["Run prediction on<br/>modified input X'"]
    Pred1 --> Check1{"Predicted Class<br/>≥ Target?"}
    Check1 -->|"Yes"| Best1["Record: feature f,<br/>change amount,<br/>displacement cost"]
    Check1 -->|"No"| Next1["Try next grid value"]
    Next1 --> Loop1
    Best1 --> MinCost["Select solution with<br/>minimum displacement"]

    MinCost --> Found1{"Any single-feature<br/>solution found?"}
    Found1 -->|"Yes"| Return["Return Best Recommendation:<br/>Feature, Old Value, New Value,<br/>Original Class, New Class"]

    Found1 -->|"No"| P2["PHASE 2: Pairwise Perturbation"]
    P2 --> Loop2["For each pair (f1, f2):<br/>Test reduced grid combinations"]
    Loop2 --> Pred2["Run prediction on X'<br/>with both features changed"]
    Pred2 --> Check2{"Predicted Class<br/>≥ Target?"}
    Check2 -->|"Yes"| Best2["Record: (f1, f2),<br/>changes, total cost"]
    Check2 -->|"No"| Next2["Try next combination"]
    Next2 --> Loop2
    Best2 --> Return

    NoRec --> End(["End"])
    Return --> End
```

---

## Figure 6: SHAP Global Feature Importance — Bar Chart Data

This is a horizontal bar chart. Use the values below to create the chart in Word:

| Rank | Feature | Mean Absolute SHAP Value |
|:---:|:---|:---:|
| 1 | GrAppv (Gross Approved Amount) | 1.5191 |
| 2 | Term (Loan Duration) | 1.4523 |
| 3 | RetainedJob | 0.6152 |
| 4 | SBA_Appv (SBA Guaranteed Portion) | 0.4927 |
| 5 | CreateJob | 0.3778 |
| 6 | DisbursementGross | 0.1547 |
| 7 | NoEmp (Number of Employees) | 0.0611 |
| 8 | UrbanRural | 0.0471 |
| 9 | RevLineCr (Revolving Line of Credit) | 0.0399 |
| 10 | NewExist (New vs Existing Business) | 0.0219 |
| 11 | LowDoc (Low Documentation Program) | 0.0124 |

**Chart Style:** Horizontal bars, sorted descending. GrAppv and Term bars should be prominently longer than all others. Use a blue/teal color gradient. Title: "Global SHAP Feature Importance (Mean |SHAP| on 1,000 Samples)".

---

## Figure 7: Streamlit Dashboard — Single Assessment Output (Description)

The Streamlit dashboard has three tabs at the top: **Single Assessment | Batch Processing | Analytics**

**Single Assessment Tab Layout:**

**Left Column (Input Panel):**
- Header: "Enterprise Features"
- 11 input fields with sliders/number inputs:
  - Term (months): slider 0–480
  - Number of Employees: number input
  - New or Existing Business: dropdown (1=Existing, 2=New)
  - CreateJob: number input
  - RetainedJob: number input
  - Urban/Rural: dropdown (0=Undefined, 1=Urban, 2=Rural)
  - Revolving Line of Credit: toggle (Yes/No)
  - Low Documentation: toggle (Yes/No)
  - Disbursement Gross: currency input
  - SBA Approved: currency input
  - Gross Approved: currency input
- Green "Assess Viability" button at bottom

**Right Column (Results Panel):**
- **Prediction Card:** Large colored badge showing class (e.g., "GROWING" in green), with confidence percentage (e.g., "94.2%")
- **Class Probabilities:** Horizontal stacked bar showing probabilities for all 5 classes
- **SHAP Explanation Section:**
  - Heading: "Why this prediction?"
  - Waterfall chart showing top features pushing prediction up/down
  - Two lists: "Risk Amplifying Factors" (red) and "Risk Mitigating Factors" (green)
- **Recommendations Section:**
  - Heading: "How to improve?"
  - Card showing: "Increase Term from 84 to 144 months → Class improves from Stable to Growing"
  - Arrow icon showing the transition

---

## Additional: Data Flow Diagram (DFD Level 0)

```mermaid
flowchart LR
    User(["Loan Officer /<br/>Analyst"]) -->|"11 Enterprise<br/>Features"| System["MSME Viability<br/>Assessment System"]
    System -->|"Viability Class +<br/>Confidence"| User
    System -->|"SHAP Feature<br/>Attributions"| User
    System -->|"Counterfactual<br/>Recommendations"| User
    System -->|"Audit Record"| DB[("SQLite<br/>Database")]
    DB -->|"Historical<br/>Records"| System
```

---

## Additional: Class Distribution Pie Chart Data

Use these values to create a pie chart titled "Five-Class Viability Distribution (Full Dataset)":

| Class | Percentage | Color Suggestion |
|:---|:---:|:---|
| Critical (Class 0) | 16.21% | Red |
| At-Risk (Class 1) | 1.23% | Orange |
| Stable (Class 2) | 53.31% | Blue |
| Growing (Class 3) | 17.42% | Green |
| Thriving (Class 4) | 11.83% | Dark Green |

---

## Additional: Confusion Matrix Heatmap Data (XGBoost)

Use this 5×5 matrix to create a heatmap. Rows = Actual, Columns = Predicted.

|  | Pred: Critical | Pred: At-Risk | Pred: Stable | Pred: Growing | Pred: Thriving |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Actual: Critical** | 23,127 | 876 | 4,906 | 0 | 0 |
| **Actual: At-Risk** | 549 | 879 | 769 | 0 | 0 |
| **Actual: Stable** | 3,893 | 520 | 90,345 | 342 | 0 |
| **Actual: Growing** | 0 | 0 | 623 | 29,187 | 1,260 |
| **Actual: Thriving** | 0 | 0 | 0 | 211 | 20,898 |

**Note:** The matrix shows that most errors occur at class boundaries: Critical↔Stable and At-Risk↔Critical/Stable. Growing and Thriving are very cleanly separated.
