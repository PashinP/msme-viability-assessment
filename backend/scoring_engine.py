"""
Loan Readiness Scoring Engine
==============================
Breaks a single "viability grade" into a section-by-section diagnostic
that tells the business owner exactly what a bank would think about
each aspect of their application — and WHY.

This is NOT about giving a score. It's about explaining what's wrong
and what's strong, in language a chai-shop owner can understand.
"""
from copy import deepcopy

USD_TO_INR = 83


def _estimate_monthly_emi(loan_amount_usd, term_months, annual_rate=0.10):
    """Estimate monthly EMI for the requested loan (in INR)."""
    loan_inr = loan_amount_usd * USD_TO_INR
    monthly_rate = annual_rate / 12
    if monthly_rate == 0 or term_months == 0:
        return loan_inr / max(term_months, 1)
    emi = loan_inr * monthly_rate * (1 + monthly_rate) ** term_months / \
          ((1 + monthly_rate) ** term_months - 1)
    return round(emi)


def _fmt_inr(amount):
    """Format INR amount in lakhs/crores for readability."""
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.1f} Cr"
    elif amount >= 100_000:
        return f"₹{amount / 100_000:.1f}L"
    elif amount >= 1_000:
        return f"₹{amount / 1_000:.0f}K"
    else:
        return f"₹{amount:,.0f}"


# ═══════════════════════════════════════════
# Individual Section Analyzers
# ═══════════════════════════════════════════

def _analyze_repayment_capacity(features, context):
    """Can they actually afford the new EMI on top of existing obligations?"""
    revenue = context.get("monthly_revenue")
    expenses = context.get("monthly_expenses")
    existing_emi = context.get("existing_debt_emi", 0)
    loan_amount = features.get("DisbursementGross", 0)
    term = features.get("Term", 84)

    # If we don't have revenue/expense data, we can't assess this deeply
    if not revenue or not expenses:
        return {
            "section": "Repayment Capacity",
            "status": "unknown",
            "score": None,
            "what_bank_sees": "We couldn't assess your repayment capacity because monthly revenue and expense details weren't provided.",
            "diagnosis": "A bank will definitely ask for your revenue and expense details. Having these numbers ready strengthens your application significantly.",
            "key_numbers": {},
        }

    monthly_profit = revenue - expenses
    new_emi = _estimate_monthly_emi(loan_amount, term)
    total_obligations = existing_emi + new_emi
    disposable = monthly_profit - total_obligations

    # Debt-to-income ratio (using profit, not revenue)
    dti = total_obligations / max(monthly_profit, 1) * 100

    if dti > 80:
        status = "critical"
        diagnosis = (
            f"After paying your expenses ({_fmt_inr(expenses)}/month) and existing EMIs "
            f"({_fmt_inr(existing_emi)}/month), adding this new loan EMI of {_fmt_inr(new_emi)}/month "
            f"would consume {dti:.0f}% of your profit. Banks typically reject applications where "
            f"obligations exceed 60% of profit. You would only have {_fmt_inr(max(0, disposable))}/month "
            f"left for personal expenses and emergencies."
        )
        score = 2
    elif dti > 60:
        status = "needs_attention"
        diagnosis = (
            f"Your total monthly obligations (existing EMIs + new loan EMI of {_fmt_inr(new_emi)}/month) "
            f"would be {dti:.0f}% of your monthly profit of {_fmt_inr(monthly_profit)}. "
            f"This is on the edge — banks prefer this to be below 50-60%. "
            f"You'd have {_fmt_inr(disposable)}/month remaining."
        )
        score = 5
    elif dti > 40:
        status = "moderate"
        diagnosis = (
            f"Your monthly profit of {_fmt_inr(monthly_profit)} can comfortably absorb the new EMI "
            f"of {_fmt_inr(new_emi)}/month. Total obligations would be {dti:.0f}% of profit, "
            f"leaving you {_fmt_inr(disposable)}/month — a healthy cushion."
        )
        score = 7
    else:
        status = "strong"
        diagnosis = (
            f"Excellent repayment capacity. Your monthly profit of {_fmt_inr(monthly_profit)} "
            f"easily covers the new EMI of {_fmt_inr(new_emi)}/month. Only {dti:.0f}% of your profit "
            f"goes to loan obligations, leaving {_fmt_inr(disposable)}/month as buffer."
        )
        score = 9

    return {
        "section": "Repayment Capacity",
        "status": status,
        "score": score,
        "what_bank_sees": (
            f"Monthly revenue: {_fmt_inr(revenue)} | Expenses: {_fmt_inr(expenses)} | "
            f"Profit: {_fmt_inr(monthly_profit)} | New EMI: {_fmt_inr(new_emi)} | "
            f"Debt-to-profit ratio: {dti:.0f}%"
        ),
        "diagnosis": diagnosis,
        "key_numbers": {
            "monthly_revenue": revenue,
            "monthly_expenses": expenses,
            "monthly_profit": monthly_profit,
            "new_emi_estimate": new_emi,
            "existing_emi": existing_emi,
            "total_obligations": total_obligations,
            "disposable_income": max(0, disposable),
            "dti_ratio": round(dti, 1),
        },
    }


def _analyze_business_stability(features, context):
    """How mature and stable is the business?"""
    years = context.get("years_in_operation")
    new_exist = features.get("NewExist", 1)
    employees = features.get("NoEmp", 0)
    registration = context.get("business_registration", "Unknown")

    is_new = new_exist == 2
    score = 5  # baseline

    findings = []

    if years is not None:
        if years >= 5:
            findings.append(f"Your business has been running for {years} years — this is a strong signal of stability. Banks love established businesses.")
            score += 3
        elif years >= 3:
            findings.append(f"With {years} years of operation, your business has proven it can survive the critical first 3 years. Good foundation.")
            score += 2
        elif years >= 1:
            findings.append(f"At {years} year(s), your business is still young. Banks see higher risk in businesses under 3 years old.")
            score -= 1
        else:
            findings.append("A brand new business carries the highest risk for lenders. Less than 1 year of track record makes approval harder.")
            score -= 2
    elif is_new:
        findings.append("New startups face tougher scrutiny from banks. Having even 6-12 months of revenue history helps significantly.")
        score -= 2

    if employees >= 10:
        findings.append(f"Having {employees} employees signals a well-established operation with proven demand.")
        score += 1
    elif employees >= 3:
        findings.append(f"Your team of {employees} shows the business is beyond the solo-founder stage.")
    elif employees <= 1:
        findings.append("Being a one-person operation isn't a dealbreaker, but banks see more risk in businesses without a team.")
        score -= 1

    if registration and registration.lower() in ("pvt ltd", "llp", "partnership"):
        findings.append(f"Being registered as a {registration} adds credibility. It shows formal business structure.")
        score += 1
    elif registration and registration.lower() in ("unregistered",):
        findings.append("An unregistered business raises red flags with lenders. Getting even basic registration (like Udyam) helps a lot.")
        score -= 2

    score = max(1, min(10, score))

    if score >= 8:
        status = "strong"
    elif score >= 5:
        status = "moderate"
    elif score >= 3:
        status = "needs_attention"
    else:
        status = "critical"

    return {
        "section": "Business Stability",
        "status": status,
        "score": score,
        "what_bank_sees": f"Business age: {years or 'Unknown'} years | Employees: {employees} | Type: {'New' if is_new else 'Existing'} | Registration: {registration}",
        "diagnosis": " ".join(findings),
        "key_numbers": {
            "years_in_operation": years,
            "employees": employees,
            "is_new_business": is_new,
            "registration": registration,
        },
    }


def _analyze_loan_structure(features, context):
    """Is the loan amount reasonable for their business size?"""
    loan_usd = features.get("DisbursementGross", 0)
    loan_inr = loan_usd * USD_TO_INR
    sba_ratio = features.get("SBA_Appv", 0) / max(loan_usd, 1)
    term = features.get("Term", 84)
    revenue = context.get("monthly_revenue")
    loan_purpose = context.get("loan_purpose", "Not specified")

    findings = []
    score = 6  # baseline

    # Loan-to-annual-revenue ratio
    if revenue:
        annual_revenue = revenue * 12
        loan_revenue_ratio = loan_inr / max(annual_revenue, 1)

        if loan_revenue_ratio > 2:
            findings.append(
                f"You're asking for {_fmt_inr(loan_inr)} which is {loan_revenue_ratio:.1f}x your annual revenue "
                f"of {_fmt_inr(annual_revenue)}. Banks rarely approve loans exceeding 1-2x annual revenue."
            )
            score -= 3
        elif loan_revenue_ratio > 1:
            findings.append(
                f"The loan amount ({_fmt_inr(loan_inr)}) is {loan_revenue_ratio:.1f}x your annual revenue. "
                f"This is on the higher side — banks are more comfortable when the loan is below 1x annual revenue."
            )
            score -= 1
        else:
            findings.append(
                f"The loan amount ({_fmt_inr(loan_inr)}) is {loan_revenue_ratio:.1f}x your annual revenue "
                f"of {_fmt_inr(annual_revenue)} — a reasonable ask that banks can justify."
            )
            score += 2

    # SBA guarantee ratio
    if sba_ratio >= 0.80:
        findings.append(f"High SBA guarantee ({sba_ratio*100:.0f}%) means the government backs most of the risk. This is good for approval.")
        score += 1
    elif sba_ratio >= 0.50:
        findings.append(f"The SBA guarantee covers {sba_ratio*100:.0f}% of the loan — moderate government backing.")
    elif sba_ratio > 0:
        findings.append(f"Low SBA guarantee ({sba_ratio*100:.0f}%). The bank is taking on most of the risk, which makes them more cautious.")
        score -= 1

    # Term appropriateness
    if term > 300:
        findings.append(f"A {term}-month ({term//12}-year) term is unusually long. This increases total interest paid and bank risk.")
        score -= 1
    elif term >= 60:
        findings.append(f"A {term}-month ({term//12}-year) repayment period is reasonable and keeps monthly EMIs manageable.")
        score += 1
    elif term < 24:
        findings.append(f"A very short {term}-month term means high monthly EMIs. Make sure your cash flow can handle it.")

    # Loan purpose
    if loan_purpose and loan_purpose.lower() not in ("not specified", ""):
        findings.append(f"Loan purpose: {loan_purpose}. Having a clear purpose strengthens the application.")

    score = max(1, min(10, score))
    status = "strong" if score >= 8 else "moderate" if score >= 5 else "needs_attention" if score >= 3 else "critical"

    return {
        "section": "Loan Structure",
        "status": status,
        "score": score,
        "what_bank_sees": f"Loan amount: {_fmt_inr(loan_inr)} | Term: {term} months | SBA guarantee: {sba_ratio*100:.0f}% | Purpose: {loan_purpose}",
        "diagnosis": " ".join(findings),
        "key_numbers": {
            "loan_amount_inr": loan_inr,
            "term_months": term,
            "sba_guarantee_pct": round(sba_ratio * 100, 1),
            "loan_purpose": loan_purpose,
        },
    }


def _analyze_documentation(features, context):
    """How well-documented is the business?"""
    has_gst = context.get("has_gst", None)
    has_udyam = context.get("has_udyam", None)
    tax_years = context.get("tax_filing_years", None)
    low_doc = features.get("LowDoc", 0)
    registration = context.get("business_registration", "Unknown")

    findings = []
    score = 5  # baseline

    if has_gst is True:
        findings.append("GST registration is active — this shows the business is in the formal economy and has verifiable revenue.")
        score += 2
    elif has_gst is False:
        findings.append("No GST registration. This is a major gap — without GST records, banks can't verify your revenue claims. Getting GST registered should be a top priority.")
        score -= 2

    if has_udyam is True:
        findings.append("Udyam MSME registration is active — this unlocks priority sector lending and lower interest rates.")
        score += 1
    elif has_udyam is False:
        findings.append("No Udyam registration. This is free and takes 10 minutes online. It qualifies you for priority sector lending benefits and lower rates.")
        score -= 1

    if tax_years is not None:
        if tax_years >= 3:
            findings.append(f"ITR filed for {tax_years} years — excellent. Banks strongly prefer 3+ years of tax history.")
            score += 2
        elif tax_years >= 1:
            findings.append(f"ITR filed for {tax_years} year(s). Banks prefer 3+ years, but having even 1-2 years helps.")
            score += 1
        else:
            findings.append("No ITR filed. This is a serious red flag — banks have no way to verify your income without tax returns.")
            score -= 3

    if low_doc == 1:
        findings.append("This is marked as a low-documentation loan. While faster, these carry higher interest rates and lower approval limits.")
        score -= 1

    if not findings:
        findings.append("We don't have enough information about your documentation status. A bank will definitely ask about GST, ITR, and business registration.")

    score = max(1, min(10, score))
    status = "strong" if score >= 8 else "moderate" if score >= 5 else "needs_attention" if score >= 3 else "critical"

    return {
        "section": "Documentation & Compliance",
        "status": status,
        "score": score,
        "what_bank_sees": f"GST: {'Yes' if has_gst else 'No' if has_gst is False else 'Unknown'} | Udyam: {'Yes' if has_udyam else 'No' if has_udyam is False else 'Unknown'} | ITR years: {tax_years if tax_years is not None else 'Unknown'} | Registration: {registration}",
        "diagnosis": " ".join(findings),
        "key_numbers": {
            "has_gst": has_gst,
            "has_udyam": has_udyam,
            "tax_filing_years": tax_years,
            "is_low_doc": low_doc == 1,
        },
    }


def _analyze_collateral(features, context):
    """Do they have security to back the loan?"""
    collateral_value = context.get("collateral_value", 0)
    loan_usd = features.get("DisbursementGross", 0)
    loan_inr = loan_usd * USD_TO_INR

    if collateral_value and collateral_value > 0:
        coverage = collateral_value / max(loan_inr, 1) * 100

        if coverage >= 150:
            status = "strong"
            score = 9
            diagnosis = (
                f"Your collateral ({_fmt_inr(collateral_value)}) covers {coverage:.0f}% of the loan amount. "
                f"This is well above the typical 100-125% requirement. Banks will be very comfortable with this."
            )
        elif coverage >= 100:
            status = "moderate"
            score = 7
            diagnosis = (
                f"Your collateral ({_fmt_inr(collateral_value)}) covers {coverage:.0f}% of the loan amount. "
                f"This meets the minimum requirement, but banks prefer 125%+ coverage for a safety margin."
            )
        elif coverage >= 50:
            status = "needs_attention"
            score = 4
            diagnosis = (
                f"Your collateral ({_fmt_inr(collateral_value)}) only covers {coverage:.0f}% of the loan. "
                f"You may need additional security, or consider applying under CGTMSE (collateral-free guarantee scheme)."
            )
        else:
            status = "needs_attention"
            score = 3
            diagnosis = (
                f"Your collateral ({_fmt_inr(collateral_value)}) covers only {coverage:.0f}% of the loan. "
                f"Consider CGTMSE or MUDRA schemes which offer collateral-free loans up to certain limits."
            )
    elif collateral_value == 0 or collateral_value is None:
        coverage = 0
        # No collateral isn't necessarily bad — MUDRA/CGTMSE exist
        if loan_inr <= 1_000_000:
            status = "moderate"
            score = 6
            diagnosis = (
                f"No collateral, but your loan amount ({_fmt_inr(loan_inr)}) is within the range of "
                f"collateral-free schemes like MUDRA (up to ₹10L) and CGTMSE (up to ₹2Cr). "
                f"You can get approved without security."
            )
        else:
            status = "needs_attention"
            score = 3
            diagnosis = (
                f"No collateral for a {_fmt_inr(loan_inr)} loan. For larger amounts, banks usually require "
                f"some form of security. Consider CGTMSE (government guarantee up to ₹2Cr without collateral) "
                f"or identifying any assets you could pledge."
            )

    return {
        "section": "Collateral & Security",
        "status": status,
        "score": score,
        "what_bank_sees": f"Collateral: {_fmt_inr(collateral_value) if collateral_value else 'None'} | Loan: {_fmt_inr(loan_inr)} | Coverage: {coverage:.0f}%",
        "diagnosis": diagnosis,
        "key_numbers": {
            "collateral_value": collateral_value or 0,
            "loan_amount_inr": loan_inr,
            "coverage_pct": round(coverage, 1),
        },
    }


def _analyze_risk_factors(features, context, prediction=None):
    """Broader risk signals from the ML model and business profile."""
    findings = []
    score = 6  # baseline

    # Use ML prediction if available
    if prediction:
        label = prediction.get("predicted_label", "")
        confidence = prediction.get("confidence", 0)

        if label in ("Critical", "At-Risk"):
            findings.append(
                f"Our ML model (trained on 900,000+ historical SBA loans) flagged this profile as '{label}' "
                f"with {confidence*100:.0f}% confidence. This means businesses with similar profiles have historically "
                f"had higher default rates."
            )
            score -= 3
        elif label == "Stable":
            findings.append(
                f"Our ML model rates this profile as '{label}' — historically, similar businesses have "
                f"had moderate success with loan repayment."
            )
        elif label in ("Growing", "Thriving"):
            findings.append(
                f"Our ML model rates this profile as '{label}' with {confidence*100:.0f}% confidence — "
                f"businesses with similar profiles have historically performed very well on loan repayment."
            )
            score += 2

    # Check for common risk patterns
    previous_history = context.get("previous_loan_history", "")
    if previous_history == "defaulted":
        findings.append("Previous loan default is the single biggest red flag for banks. This makes new approval very difficult without significant changes.")
        score -= 4
    elif previous_history == "repaid":
        findings.append("Successfully repaid a previous loan — this is a strong positive signal. Banks reward good credit history.")
        score += 2
    elif previous_history == "ongoing":
        findings.append("You have an ongoing loan, which is fine as long as your repayment capacity can handle multiple obligations.")

    # Industry context
    industry = context.get("industry_sector", "")
    if industry:
        findings.append(f"Industry: {industry}. Banks assess sector-specific risks — some industries (like food/retail) have higher default rates than manufacturing or services.")

    if not findings:
        findings.append("We don't have enough context to assess broader risk factors. The ML model needs your full profile for an accurate assessment.")

    score = max(1, min(10, score))
    status = "strong" if score >= 8 else "moderate" if score >= 5 else "needs_attention" if score >= 3 else "critical"

    return {
        "section": "Risk Profile",
        "status": status,
        "score": score,
        "what_bank_sees": f"ML Assessment: {prediction.get('predicted_label', 'N/A') if prediction else 'N/A'} | Previous loans: {previous_history or 'Unknown'} | Industry: {industry or 'Unknown'}",
        "diagnosis": " ".join(findings),
        "key_numbers": {
            "ml_prediction": prediction.get("predicted_label", "") if prediction else "",
            "ml_confidence": prediction.get("confidence", 0) if prediction else 0,
            "previous_loan_history": previous_history,
            "industry_sector": industry,
        },
    }


# ═══════════════════════════════════════════
# Main Scoring Function
# ═══════════════════════════════════════════

def generate_readiness_assessment(features: dict, context: dict, prediction: dict = None) -> dict:
    """
    Generate a complete loan readiness assessment.

    Args:
        features: The 11 core ML features (Term, NoEmp, etc.)
        context: Additional context fields (monthly_revenue, has_gst, etc.)
        prediction: Optional ML prediction result from engine.predict()

    Returns:
        {
            "sections": [...],  # list of section analyses
            "overall_status": "...",
            "overall_score": int,
            "summary": "...",  # human-readable overall assessment
            "strengths": [...],
            "weaknesses": [...],
        }
    """
    sections = [
        _analyze_repayment_capacity(features, context),
        _analyze_business_stability(features, context),
        _analyze_loan_structure(features, context),
        _analyze_documentation(features, context),
        _analyze_collateral(features, context),
        _analyze_risk_factors(features, context, prediction),
    ]

    # Calculate overall score from sections that have scores
    scored = [s for s in sections if s["score"] is not None]
    if scored:
        overall_score = round(sum(s["score"] for s in scored) / len(scored) * 10)
    else:
        overall_score = 50

    overall_score = max(0, min(100, overall_score))

    # Determine overall status
    if overall_score >= 75:
        overall_status = "strong"
    elif overall_score >= 50:
        overall_status = "moderate"
    elif overall_score >= 30:
        overall_status = "needs_work"
    else:
        overall_status = "critical"

    # Extract strengths and weaknesses
    strengths = [s["section"] for s in sections if s["status"] == "strong"]
    weaknesses = [s["section"] for s in sections if s["status"] in ("critical", "needs_attention")]

    # Generate human-readable summary
    summary = _generate_summary(sections, overall_status, overall_score)

    return {
        "sections": sections,
        "overall_status": overall_status,
        "overall_score": overall_score,
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


def _generate_summary(sections, status, score):
    """Generate a paragraph-level summary of the assessment."""
    weak_sections = [s for s in sections if s["status"] in ("critical", "needs_attention")]
    strong_sections = [s for s in sections if s["status"] == "strong"]

    parts = []

    if status == "strong":
        parts.append("Your loan application looks strong overall.")
    elif status == "moderate":
        parts.append("Your application has a reasonable foundation, but there are areas that need improvement before approaching a bank.")
    elif status == "needs_work":
        parts.append("There are significant gaps in your application that would likely lead to rejection in its current form.")
    else:
        parts.append("Your application faces serious challenges. Major changes are needed before a bank would consider approval.")

    if strong_sections:
        names = ", ".join(s["section"] for s in strong_sections)
        parts.append(f"Your strongest areas are: {names}.")

    if weak_sections:
        names = ", ".join(s["section"] for s in weak_sections)
        parts.append(f"The areas needing immediate attention are: {names}.")
        parts.append("See the detailed prescriptions below for specific actions you can take.")

    return " ".join(parts)
