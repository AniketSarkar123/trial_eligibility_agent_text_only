"""
Generate human-readable why-not eligibility reports.
Purely deterministic - no LLM.
"""

from src.matcher import EligibilityResult, calculate_indeterminate_risk_breakdown


def generate_why_not_report(result: EligibilityResult) -> str:
    """Generate a plain-text why-not report."""

    lines = []
    lines.append(f"Trial: {result.nct_id}")
    lines.append(f"Eligible: {'YES' if result.eligible else 'NO'}")
    
    # We retrieve the triage_status if it exists in the result object
    triage_status = getattr(result, "triage_status", "N/A")
    lines.append(f"Triage Status: {triage_status}")
    # Format and print the metastatic status
    is_met = getattr(result, "is_metastatic", None)
    if is_met is True:
        lines.append("THE PATIENT IS METASTATIC!")

    lines.append("")

    if result.failing_criteria:
        lines.append("FAILING CRITERIA:")
        for i, r in enumerate(result.failing_criteria, 1):
            rule_type = "Inclusion" if r.criterion.is_inclusion else "Exclusion"
            lines.append(
                f"  {i}. [{r.criterion.category.value}] [{rule_type}] {r.criterion.description}"
            )
            
            op_str = r.criterion.operator.value if r.criterion.operator else "N/A"
            val_str = r.criterion.value if r.criterion.value is not None else "N/A"
            
            if r.criterion.field == "unmapped_rule":
                lines.append("     Field: unmapped_rule")
                lines.append(f"     → {r.reason}")
            else:
                lines.append(f"     Patient value: {r.patient_value}")
                lines.append(f"     Required: {op_str} {val_str}")
                lines.append(f"     → {r.reason}")
            
            lines.append("")

    if result.indeterminate_criteria:
        # Calculate hard vs soft indeterminates
        hard_indet = sum(1 for r in result.indeterminate_criteria if r.criterion.category.value != "administrative")
        soft_indet = sum(1 for r in result.indeterminate_criteria if r.criterion.category.value == "administrative")
        
        lines.append(f"INDETERMINATE (missing patient data): [Hard: {hard_indet} | Soft: {soft_indet}]")
        for i, r in enumerate(result.indeterminate_criteria, 1):
            rule_type = "Inclusion" if r.criterion.is_inclusion else "Exclusion"
            lines.append(
                f"  {i}. [{r.criterion.category.value}] [{rule_type}] {r.criterion.description}"
            )
            
            if r.criterion.field == "unmapped_rule":
                lines.append("     Field: unmapped_rule")
                # Indeterminate unmapped rules will have their fallback reason stored
                reason_text = r.reason if "LLM Fallback:" in r.reason else "Data not available in patient record"
                lines.append(f"     → {reason_text}")
            else:
                lines.append(f"     Field: {r.criterion.field}")
                lines.append("     → Data not available in patient record")
            lines.append("")

    passing = [r for r in result.results if r.status == "pass"]
    lines.append(f"PASSING CRITERIA: {len(passing)} of {len(result.results)}")
    lines.append("")
    
    # Generate the Risk Score Breakdown Table
    if result.indeterminate_criteria:
        lines.append("RISK SCORE CALCULATION BREAKDOWN:")
        lines.append("-" * 90)
        lines.append(f"{'CRITERION ID':<25} | {'CATEGORY':<20} | {'BASE PENALTY':<15} | {'MULTIPLIER':<10} | {'SCORE':<10}")
        lines.append("-" * 90)
        
        _, breakdown = calculate_indeterminate_risk_breakdown(result.indeterminate_criteria)
        
        for item in breakdown:
            lines.append(
                f"{item['criterion_id']:<25} | {item['category']:<20} | "
                f"{item['base_penalty']:<15.1f} | {item['multiplier']:<10.1f} | {item['calculated_score']:<10.1f}"
            )
            
        lines.append("-" * 90)
        lines.append(f"{'TOTAL RISK SCORE:':<81} {result.risk_score:.1f}")

    return "\n".join(lines)


def generate_why_not_json(result: EligibilityResult) -> dict:
    """Generate structured why-not report (for programmatic use)."""

    return {
        "nct_id": result.nct_id,
        "eligible": result.eligible,
        "triage_status": getattr(result, "triage_status", "N/A"),
        "summary": {
            "total_criteria": len(result.results),
            "passing": len([r for r in result.results if r.status == "pass"]),
            "failing": len(result.failing_criteria),
            "indeterminate": len(result.indeterminate_criteria),
        },
        "failures": [
            {
                "criterion_id": r.criterion.criterion_id,
                "category": r.criterion.category.value,
                "rule_type": "inclusion" if r.criterion.is_inclusion else "exclusion",
                "description": r.criterion.description,
                "patient_value": r.patient_value,
                "required_operator": r.criterion.operator.value if r.criterion.operator else None,
                "required_value": r.criterion.value,
            }
            for r in result.failing_criteria
        ],
        "missing_data": [
            {
                "criterion_id": r.criterion.criterion_id,
                "category": r.criterion.category.value,
                "rule_type": "inclusion" if r.criterion.is_inclusion else "exclusion",
                "field": r.criterion.field,
                "description": r.criterion.description,
            }
            for r in result.indeterminate_criteria
        ],
    }
