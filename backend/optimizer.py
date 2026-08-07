"""
Loan Structure Optimizer + Red Flags + Government Schemes
==========================================================
Uses the prediction engine and KNN similarity engine to:
1. Find the optimal loan structure for maximum success probability
2. Identify red flags (risky parameter combinations)
3. Match relevant government schemes to the user's profile
"""
import numpy as np
from copy import deepcopy


# ═══════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════

USD_TO_INR = 83  # Approximate conversion rate


def fmt_inr(usd_amount):
    """Format USD amount as ₹ in Indian numbering (lakhs/crores)."""
    inr = usd_amount * USD_TO_INR
    if inr >= 10000000:
        return f"₹{inr/10000000:.1f} Cr"
    elif inr >= 100000:
        return f"₹{inr/100000:.1f}L"
    elif inr >= 1000:
        return f"₹{inr/1000:.0f}K"
    else:
        return f"₹{inr:,.0f}"


# ═══════════════════════════════════════════
# GOVERNMENT SCHEMES KNOWLEDGE BASE (INDIA-FIRST)
# ═══════════════════════════════════════════

GOVERNMENT_SCHEMES = [
    {
        "name": "PM SVANidhi (Street Vendors)",
        "country": "India",
        "max_amount_inr": 50000,
        "description": "Working capital loan up to ₹50,000 for street vendors. No collateral, 7% interest subsidy.",
        "eligibility": {"max_employees": 3, "business_types": [1, 2]},
        "benefits": ["₹50K without collateral", "7% interest subsidy", "₹1,200/yr digital incentive"],
        "url": "https://pmsvanidhi.mohua.gov.in",
    },
    {
        "name": "MUDRA Yojana — Shishu",
        "country": "India",
        "max_amount_inr": 50000,
        "description": "Micro loans up to ₹50,000 for tiny/micro businesses. Zero collateral, available at all banks.",
        "eligibility": {"max_employees": 5, "business_types": [1, 2]},
        "benefits": ["Zero collateral", "Low interest (7-12%)", "All banks participate"],
        "url": "https://www.mudra.org.in",
    },
    {
        "name": "MUDRA Yojana — Kishor",
        "country": "India",
        "max_amount_inr": 500000,
        "description": "Loans ₹50,000 to ₹5 lakh for growing businesses needing working capital or equipment.",
        "eligibility": {"max_employees": 20, "business_types": [1, 2]},
        "benefits": ["No collateral up to ₹5L", "Flexible repayment", "Equipment + working capital"],
        "url": "https://www.mudra.org.in",
    },
    {
        "name": "MUDRA Yojana — Tarun",
        "country": "India",
        "max_amount_inr": 1000000,
        "description": "Loans ₹5 lakh to ₹10 lakh for established businesses looking to expand operations.",
        "eligibility": {"max_employees": 50, "business_types": [1]},
        "benefits": ["No collateral up to ₹10L", "For expansion", "All banks participate"],
        "url": "https://www.mudra.org.in",
    },
    {
        "name": "PM Vishwakarma Yojana",
        "country": "India",
        "max_amount_inr": 300000,
        "description": "For traditional artisans & craftspeople — carpenters, blacksmiths, potters, tailors. Subsidized loans + free training.",
        "eligibility": {"max_employees": 10, "business_types": [1, 2]},
        "benefits": ["₹3L at just 5% interest", "Free skills training", "₹15K toolkit grant", "Marketing support"],
        "url": "https://pmvishwakarma.gov.in",
    },
    {
        "name": "PMEGP (PM Employment Generation Programme)",
        "country": "India",
        "max_amount_inr": 2500000,
        "description": "Government SUBSIDY of 15-35% on project cost for new enterprises. The subsidy portion is free money — no repayment.",
        "eligibility": {"max_employees": 50, "business_types": [2]},
        "benefits": ["15-35% SUBSIDY (free money!)", "Up to ₹25L for manufacturing", "Up to ₹10L for services"],
        "url": "https://kviconline.gov.in/pmegpeportal/",
    },
    {
        "name": "CGTMSE (Credit Guarantee Scheme)",
        "country": "India",
        "max_amount_inr": 20000000,
        "description": "Government provides collateral-free guarantee to banks — your business gets a loan WITHOUT any security up to ₹2 Crore.",
        "eligibility": {"max_employees": 200, "business_types": [1, 2]},
        "benefits": ["Collateral-free up to ₹2 Cr", "Govt backs your loan", "New & existing businesses"],
        "url": "https://www.cgtmse.in",
    },
    {
        "name": "Stand-Up India",
        "country": "India",
        "max_amount_inr": 10000000,
        "description": "Loans ₹10 lakh to ₹1 crore for SC/ST and women entrepreneurs starting new ventures.",
        "eligibility": {"max_employees": 200, "business_types": [2]},
        "benefits": ["For SC/ST & women entrepreneurs", "₹10L to ₹1Cr", "Greenfield enterprises"],
        "url": "https://www.standupmitra.in",
    },
    {
        "name": "NSIC Schemes (for Govt Tenders)",
        "country": "India",
        "max_amount_inr": 5000000,
        "description": "Register with NSIC to get preference in government tenders, subsidized raw materials, and credit support.",
        "eligibility": {"max_employees": 200, "business_types": [1]},
        "benefits": ["Priority in govt tenders", "Subsidized raw materials", "Bank credit support"],
        "url": "https://www.nsic.co.in",
    },
    {
        "name": "SIDBI MSME Loans",
        "country": "India",
        "max_amount_inr": 50000000,
        "description": "Growth capital for established MSMEs looking to scale operations, adopt technology, or expand.",
        "eligibility": {"max_employees": 500, "business_types": [1]},
        "benefits": ["For scaling businesses", "Flexible terms", "Quick disbursement", "Large ticket size"],
        "url": "https://www.sidbi.in",
    },
]


def match_government_schemes(features: dict) -> list[dict]:
    """Match user's business profile to relevant government schemes."""
    loan_amount_usd = features.get("DisbursementGross", 0)
    loan_amount_inr = loan_amount_usd * USD_TO_INR
    employees = features.get("NoEmp", 0)
    business_type = features.get("NewExist", 1)

    matched = []
    for scheme in GOVERNMENT_SCHEMES:
        elig = scheme["eligibility"]
        if employees > elig["max_employees"]:
            continue
        if business_type not in elig["business_types"]:
            continue

        max_inr = scheme["max_amount_inr"]
        if loan_amount_inr <= max_inr * 1.5:
            relevance = "high" if loan_amount_inr <= max_inr else "medium"
            if "subsidy" in scheme["description"].lower():
                relevance = "high"
            matched.append({**scheme, "relevance": relevance})

    matched.sort(key=lambda x: (0 if x["relevance"] == "high" else 1, x["max_amount_inr"]))
    return matched[:5]


# ═══════════════════════════════════════════
# RED FLAGS DETECTION
# ═══════════════════════════════════════════

def detect_red_flags(features: dict, similar_data: dict = None) -> list[dict]:
    """Identify risky parameter combinations from 897K loan patterns."""
    flags = []

    # Flag 1: New business + large loan
    if features.get("NewExist") == 2 and features.get("DisbursementGross", 0) > 100000:
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": "High-value loan for a new business",
            "explanation": f"New businesses requesting {fmt_inr(features['DisbursementGross'])} have significantly higher default rates — "
                           "28% default vs 15% for established businesses in our dataset.",
            "suggestion": "Start smaller and scale up after 2-3 years of track record."
        })

    # Flag 2: Low-doc + large amount
    if features.get("LowDoc") == 1 and features.get("DisbursementGross", 0) > 150000:
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": "Low documentation on a large loan",
            "explanation": "Low-doc loans above ₹1 Crore have nearly 2x the default rate. Banks and NBFCs view this as high risk.",
            "suggestion": "Prepare full documentation — ITR, GST returns, bank statements. It dramatically improves approval odds."
        })

    # Flag 3: Short term + large amount
    term = features.get("Term", 84)
    amount = features.get("DisbursementGross", 0)
    if term > 0 and amount / term > 5000 and term < 60:
        monthly_inr = (amount / term) * USD_TO_INR
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": "Heavy monthly EMI burden",
            "explanation": f"Your EMI would be approximately ₹{monthly_inr/1000:.0f}K/month — that's severe cash flow pressure.",
            "suggestion": f"Extend the term to {max(84, term * 2)} months to cut your EMI by ~50%."
        })

    # Flag 4: Low guarantee ratio
    sba = features.get("SBA_Appv", 0)
    gr = features.get("GrAppv", 1)
    if gr > 0 and sba / gr < 0.5:
        ratio = sba / gr
        flags.append({
            "severity": "medium", "emoji": "🟡",
            "flag": f"Low guarantee coverage ({ratio*100:.0f}%)",
            "explanation": "Loans with less than 50% coverage default more often. Under CGTMSE, you can get up to 85% govt guarantee.",
            "suggestion": f"Apply for CGTMSE guarantee — increase from {ratio*100:.0f}% to 75-85%."
        })

    # Flag 5: Rural + new business
    if features.get("UrbanRural") == 2 and features.get("NewExist") == 2:
        flags.append({
            "severity": "medium", "emoji": "🟡",
            "flag": "New business in a rural area",
            "explanation": "Rural startups face limited customers, supply chain issues, fewer banking options — 20% higher default rates.",
            "suggestion": "Look into PMEGP — they give 25-35% subsidy for rural enterprises (vs 15-25% urban)."
        })

    # Flag 6: Zero job creation
    if features.get("CreateJob", 0) == 0 and features.get("RetainedJob", 0) == 0:
        flags.append({
            "severity": "low", "emoji": "🟡",
            "flag": "No job creation documented",
            "explanation": "Applications showing employment creation get priority under PMEGP, MUDRA, and most bank schemes.",
            "suggestion": "Document planned hiring — even 1-2 new jobs strengthens your application."
        })

    # Flag 7: Revolving credit + startup
    if features.get("RevLineCr") == 1 and features.get("NewExist") == 2:
        flags.append({
            "severity": "medium", "emoji": "🟡",
            "flag": "Revolving credit for a startup",
            "explanation": "CC/OD facilities are riskier for new businesses with unpredictable cash flow. Term loans with fixed EMIs are safer.",
            "suggestion": "Start with a fixed-term loan. Apply for CC/OD after 2-3 years of banking history."
        })

    # Flag from similarity data
    if similar_data and similar_data.get("success_rate", 1) < 0.6:
        rate = similar_data["success_rate"]
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": f"Below-average peer performance ({rate*100:.0f}% success)",
            "explanation": f"Only {rate*100:.0f}% of similar businesses repaid successfully (vs 82% average).",
            "suggestion": "Reduce loan amount or extend term. Also look into CGTMSE guarantee for safety."
        })

    severity_order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda x: severity_order.get(x["severity"], 3))
    return flags


# ═══════════════════════════════════════════
# LOAN STRUCTURE OPTIMIZER
# ═══════════════════════════════════════════

class LoanOptimizer:
    """Finds optimal loan structure by sweeping parameters."""

    def __init__(self, prediction_engine):
        self.engine = prediction_engine

    def find_optimal_term(self, features: dict) -> dict:
        """Find the safest loan term for this business profile."""
        terms = [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 300, 360]
        results = []
        for term in terms:
            test = deepcopy(features)
            test["Term"] = term
            pred = self.engine.predict(test)
            results.append({
                "term_months": term, "term_years": round(term / 12, 1),
                "predicted_class": pred["predicted_class"],
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"],
            })
        best = max(results, key=lambda x: (x["predicted_class"], x["confidence"]))
        return {
            "all_terms": results,
            "recommended_term": best["term_months"],
            "recommended_term_years": best["term_years"],
            "best_class": best["predicted_label"],
            "best_confidence": best["confidence"],
        }

    def find_max_safe_amount(self, features: dict, target_class: int = 2) -> dict:
        """Find max loan amount where predicted class stays >= target."""
        original_amount = features.get("DisbursementGross", 100000)
        test_multipliers = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        results = []
        max_safe = 0
        for mult in test_multipliers:
            test_amount = int(original_amount * mult)
            test = deepcopy(features)
            test["DisbursementGross"] = test_amount
            test["GrAppv"] = test_amount
            test["SBA_Appv"] = int(test_amount * 0.75)
            pred = self.engine.predict(test)
            is_safe = pred["predicted_class"] >= target_class
            results.append({
                "amount": test_amount, "multiplier": mult,
                "predicted_class": pred["predicted_class"],
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"], "is_safe": is_safe,
            })
            if is_safe:
                max_safe = test_amount
        return {
            "requested_amount": original_amount, "max_safe_amount": max_safe,
            "amount_analysis": results,
            "can_take_more": max_safe > original_amount,
            "should_reduce": max_safe < original_amount,
        }

    def generate_optimal_structure(self, features: dict) -> dict:
        """Generate the single best loan config for this business."""
        optimized = deepcopy(features)

        # Optimal term
        term_data = self.find_optimal_term(features)
        optimized["Term"] = term_data["recommended_term"]

        # Safe amount
        amount_data = self.find_max_safe_amount(features)

        # Fix common issues
        if optimized.get("LowDoc") == 1:
            optimized["LowDoc"] = 0
        optimized["SBA_Appv"] = int(optimized["DisbursementGross"] * 0.80)
        optimized["GrAppv"] = optimized["DisbursementGross"]
        if optimized.get("CreateJob", 0) == 0:
            optimized["CreateJob"] = max(1, features.get("CreateJob", 0))
        if optimized.get("RetainedJob", 0) == 0:
            optimized["RetainedJob"] = max(features.get("NoEmp", 1), features.get("RetainedJob", 0))

        original_pred = self.engine.predict(features)
        optimized_pred = self.engine.predict(optimized)

        changes = []
        for key in features:
            if features[key] != optimized[key]:
                changes.append({"feature": key, "original": features[key], "optimized": optimized[key]})

        return {
            "original_prediction": original_pred,
            "optimized_prediction": optimized_pred,
            "optimized_features": optimized,
            "changes": changes,
            "improvement": optimized_pred["predicted_class"] - original_pred["predicted_class"],
            "term_analysis": term_data,
            "amount_analysis": amount_data,
        }
