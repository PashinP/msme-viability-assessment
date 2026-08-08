"""
Prescription Engine — Actionable Fix Suggestions
==================================================
For each weak section in the readiness assessment, generates specific,
numbered actions the business owner can take to improve their chances.

Think of this as "Grammarly suggestions" for a loan application.
Each suggestion tells you WHAT to do, WHY, and HOW MUCH it helps.
"""


def generate_prescriptions(assessment: dict, features: dict, context: dict) -> list[dict]:
    """
    Generate actionable prescriptions for weak sections.

    Args:
        assessment: Output from scoring_engine.generate_readiness_assessment()
        features: The 11 core ML features
        context: Additional context fields

    Returns:
        List of prescription dicts, one per weak section.
    """
    prescriptions = []

    for section in assessment["sections"]:
        if section["status"] in ("critical", "needs_attention"):
            rx = _prescribe_for_section(section, features, context)
            if rx:
                prescriptions.append(rx)
        elif section["status"] == "moderate":
            # Even moderate sections can have quick wins
            rx = _prescribe_for_section(section, features, context, quick_wins_only=True)
            if rx and rx.get("suggestions"):
                prescriptions.append(rx)

    return prescriptions


def _prescribe_for_section(section: dict, features: dict, context: dict,
                           quick_wins_only: bool = False) -> dict | None:
    """Route to the appropriate prescription generator."""
    name = section["section"]
    generators = {
        "Repayment Capacity": _rx_repayment,
        "Business Stability": _rx_stability,
        "Loan Structure": _rx_loan_structure,
        "Documentation & Compliance": _rx_documentation,
        "Collateral & Security": _rx_collateral,
        "Risk Profile": _rx_risk,
    }

    gen = generators.get(name)
    if not gen:
        return None

    return gen(section, features, context, quick_wins_only)


# ═══════════════════════════════════════════
# Prescription Generators
# ═══════════════════════════════════════════

def _rx_repayment(section, features, context, quick_wins_only=False):
    """Prescriptions for weak repayment capacity."""
    numbers = section.get("key_numbers", {})
    dti = numbers.get("dti_ratio", 0)
    existing_emi = numbers.get("existing_emi", 0)
    new_emi = numbers.get("new_emi_estimate", 0)
    revenue = numbers.get("monthly_revenue", 0)
    profit = numbers.get("monthly_profit", 0)

    suggestions = []

    # Suggestion: Reduce loan amount
    if dti > 50:
        target_emi = profit * 0.45  # Target 45% DTI
        max_affordable_emi = target_emi - existing_emi
        if max_affordable_emi > 0 and new_emi > max_affordable_emi:
            ratio = max_affordable_emi / max(new_emi, 1)
            suggestions.append({
                "action": "Reduce the loan amount",
                "detail": (
                    f"To bring your debt-to-profit ratio below 50%, consider reducing "
                    f"the loan amount to roughly {ratio*100:.0f}% of what you're asking. "
                    f"This would keep your EMI manageable at around ₹{max_affordable_emi:,.0f}/month."
                ),
                "impact": "Brings DTI ratio below the 50% safety threshold",
                "difficulty": "Easy",
                "priority": "high",
            })

    # Suggestion: Extend the term
    if dti > 40 and features.get("Term", 84) < 180:
        current_term = features.get("Term", 84)
        suggestions.append({
            "action": "Extend the repayment period",
            "detail": (
                f"Your current term is {current_term} months. Extending to {min(current_term + 60, 240)} months "
                f"would reduce your monthly EMI significantly, making the application more comfortable."
            ),
            "impact": "Lower monthly EMI, improved cash flow buffer",
            "difficulty": "Easy",
            "priority": "high",
        })

    # Suggestion: Pay off existing debt
    if existing_emi > 0 and dti > 50:
        suggestions.append({
            "action": "Clear existing loans before applying",
            "detail": (
                f"You're currently paying ₹{existing_emi:,.0f}/month in existing EMIs. "
                f"If you can close even one of these, it immediately reduces your debt obligations "
                f"and makes the new loan more viable."
            ),
            "impact": f"Reduces total obligations by ₹{existing_emi:,.0f}/month",
            "difficulty": "Medium",
            "priority": "medium",
        })

    # Suggestion: Show additional income
    if dti > 50 and not quick_wins_only:
        suggestions.append({
            "action": "Document additional income sources",
            "detail": (
                "If you have rental income, freelance earnings, spouse's income, or income "
                "from a second business — include it in the application. This increases "
                "your demonstrated repayment capacity."
            ),
            "impact": "Reduces effective DTI ratio",
            "difficulty": "Easy",
            "priority": "medium",
        })

    if not suggestions:
        return None

    return {
        "section": "Repayment Capacity",
        "problem": section["diagnosis"],
        "suggestions": suggestions,
    }


def _rx_stability(section, features, context, quick_wins_only=False):
    """Prescriptions for business stability concerns."""
    numbers = section.get("key_numbers", {})
    years = numbers.get("years_in_operation")
    registration = numbers.get("registration", "Unknown")
    is_new = numbers.get("is_new_business", False)

    suggestions = []

    if is_new or (years is not None and years < 2):
        suggestions.append({
            "action": "Build a track record first",
            "detail": (
                "Banks strongly prefer businesses with 2-3+ years of history. If possible, "
                "wait until you have at least 12-18 months of operations and revenue before applying. "
                "In the meantime, maintain meticulous financial records."
            ),
            "impact": "Dramatically improves approval chances",
            "difficulty": "Requires time",
            "priority": "high",
        })

    if registration and registration.lower() in ("unregistered", "unknown"):
        suggestions.append({
            "action": "Register your business formally",
            "detail": (
                "Get Udyam MSME registration (free, 10 min online at udyamregistration.gov.in), "
                "and consider Sole Proprietorship or Partnership registration. This moves your "
                "business from 'informal' to 'formal' in the bank's eyes."
            ),
            "impact": "Unlocks priority sector lending and builds credibility",
            "difficulty": "Easy",
            "priority": "high",
        })

    if not quick_wins_only and (years is not None and years < 5):
        suggestions.append({
            "action": "Prepare a business plan",
            "detail": (
                "A simple 2-3 page document showing your business model, revenue trajectory, "
                "market opportunity, and expansion plan can significantly sway a loan officer. "
                "It shows you're serious and have thought about growth."
            ),
            "impact": "Adds credibility for younger businesses",
            "difficulty": "Medium",
            "priority": "medium",
        })

    if not suggestions:
        return None

    return {
        "section": "Business Stability",
        "problem": section["diagnosis"],
        "suggestions": suggestions,
    }


def _rx_loan_structure(section, features, context, quick_wins_only=False):
    """Prescriptions for loan structure issues."""
    numbers = section.get("key_numbers", {})
    loan_inr = numbers.get("loan_amount_inr", 0)
    revenue = context.get("monthly_revenue", 0)

    suggestions = []

    if revenue and loan_inr > revenue * 12:
        target_amount = revenue * 10  # 10 months of revenue
        suggestions.append({
            "action": "Reduce the loan amount to match your revenue",
            "detail": (
                f"Banks rarely approve loans exceeding annual revenue. Consider starting "
                f"with a smaller amount (around ₹{target_amount/100000:.0f}L) and applying for "
                f"a top-up after 12-18 months of successful repayment."
            ),
            "impact": "Much higher approval probability",
            "difficulty": "Easy",
            "priority": "high",
        })

    sba_pct = numbers.get("sba_guarantee_pct", 0)
    if sba_pct < 50 and not quick_wins_only:
        suggestions.append({
            "action": "Apply through CGTMSE or MUDRA for government guarantee",
            "detail": (
                "Government-backed guarantee schemes reduce the bank's risk significantly. "
                "Ask your bank if the loan qualifies for CGTMSE coverage or MUDRA scheme."
            ),
            "impact": "Higher approval with lower interest rates",
            "difficulty": "Easy",
            "priority": "medium",
        })

    if not suggestions:
        return None

    return {
        "section": "Loan Structure",
        "problem": section["diagnosis"],
        "suggestions": suggestions,
    }


def _rx_documentation(section, features, context, quick_wins_only=False):
    """Prescriptions for documentation gaps."""
    numbers = section.get("key_numbers", {})

    suggestions = []

    if numbers.get("has_gst") is False:
        suggestions.append({
            "action": "Get GST registration",
            "detail": (
                "Apply for GST at gst.gov.in. Even if your turnover is below the threshold, "
                "voluntary GST registration provides verifiable revenue proof that banks accept. "
                "Processing takes 3-7 working days."
            ),
            "impact": "Major — banks can now verify your revenue claims",
            "difficulty": "Easy (3-7 days)",
            "priority": "high",
        })

    if numbers.get("has_udyam") is False:
        suggestions.append({
            "action": "Register on Udyam Portal",
            "detail": (
                "Free registration at udyamregistration.gov.in (10 minutes with Aadhaar). "
                "This gives you an MSME certificate that qualifies you for priority sector lending, "
                "lower interest rates, and government scheme benefits."
            ),
            "impact": "Unlocks lower rates and government scheme eligibility",
            "difficulty": "Very Easy (10 minutes)",
            "priority": "high",
        })

    tax_years = numbers.get("tax_filing_years")
    if tax_years is not None and tax_years < 3:
        if tax_years == 0:
            suggestions.append({
                "action": "File ITR immediately",
                "detail": (
                    "File at least the last 2 years of ITR through a CA. Banks consider "
                    "ITR the most reliable proof of income. Without it, you're essentially asking "
                    "the bank to trust your word — which they won't."
                ),
                "impact": "Critical — most banks won't process without ITR",
                "difficulty": "Medium (needs CA, costs ₹2-5K)",
                "priority": "high",
            })
        else:
            suggestions.append({
                "action": f"File ITR for one more year to reach the 3-year mark",
                "detail": (
                    f"You have {tax_years} year(s) of ITR. Banks prefer 3 years. "
                    f"File the remaining year(s) before applying to strengthen your case."
                ),
                "impact": "Improves income verification credibility",
                "difficulty": "Easy",
                "priority": "medium",
            })

    if not suggestions:
        return None

    return {
        "section": "Documentation & Compliance",
        "problem": section["diagnosis"],
        "suggestions": suggestions,
    }


def _rx_collateral(section, features, context, quick_wins_only=False):
    """Prescriptions for collateral gaps."""
    numbers = section.get("key_numbers", {})
    collateral = numbers.get("collateral_value", 0)
    loan_inr = numbers.get("loan_amount_inr", 0)

    suggestions = []

    if collateral == 0 and loan_inr > 1_000_000:
        suggestions.append({
            "action": "Apply under CGTMSE (Credit Guarantee Scheme)",
            "detail": (
                "CGTMSE provides government-backed collateral-free guarantee for loans up to ₹2 Crore. "
                "Ask your bank to process the loan under CGTMSE — they are mandated to offer this. "
                "Visit cgtmse.in for details."
            ),
            "impact": "Eliminates the collateral requirement entirely",
            "difficulty": "Easy — ask the bank",
            "priority": "high",
        })

    if collateral > 0 and loan_inr > 0:
        coverage = collateral / max(loan_inr, 1) * 100
        if coverage < 100:
            gap = loan_inr - collateral
            suggestions.append({
                "action": "Bridge the collateral gap",
                "detail": (
                    f"You need approximately ₹{gap/100000:.1f}L more in collateral to reach 100% coverage. "
                    f"Options: (a) add a guarantor with property, (b) pledge business equipment/inventory, "
                    f"(c) use fixed deposits as additional security."
                ),
                "impact": f"Closes the {100-coverage:.0f}% collateral gap",
                "difficulty": "Medium",
                "priority": "high",
            })

    if loan_inr <= 1_000_000 and collateral == 0:
        suggestions.append({
            "action": "Apply under MUDRA Yojana",
            "detail": (
                "For loans up to ₹10 lakh, MUDRA provides collateral-free loans through all banks. "
                "No security needed. Ask specifically for MUDRA at your bank."
            ),
            "impact": "No collateral needed for loans up to ₹10L",
            "difficulty": "Very Easy",
            "priority": "high",
        })

    if not suggestions:
        return None

    return {
        "section": "Collateral & Security",
        "problem": section["diagnosis"],
        "suggestions": suggestions,
    }


def _rx_risk(section, features, context, quick_wins_only=False):
    """Prescriptions for risk profile issues."""
    numbers = section.get("key_numbers", {})
    history = numbers.get("previous_loan_history", "")
    ml_label = numbers.get("ml_prediction", "")

    suggestions = []

    if history == "defaulted":
        suggestions.append({
            "action": "Settle the defaulted loan first",
            "detail": (
                "A previous default is the biggest barrier to new credit. Contact the original "
                "lender and negotiate a one-time settlement (OTS). Once settled, get a No Dues "
                "certificate. After 12 months of clean credit history, your chances improve."
            ),
            "impact": "Removes the single biggest red flag",
            "difficulty": "Hard (requires negotiation)",
            "priority": "high",
        })

    if ml_label in ("Critical", "At-Risk") and not quick_wins_only:
        suggestions.append({
            "action": "Restructure your loan parameters",
            "detail": (
                "Our model identified risk patterns similar to historically defaulted loans. "
                "Use the 'Manual' mode to experiment with different loan amounts, terms, and "
                "SBA guarantee levels to find a combination that moves your profile toward 'Stable'."
            ),
            "impact": "Could shift ML assessment to a safer category",
            "difficulty": "Medium — requires experimentation",
            "priority": "high",
        })

    if not suggestions:
        return None

    return {
        "section": "Risk Profile",
        "problem": section["diagnosis"],
        "suggestions": suggestions,
    }
