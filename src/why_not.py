"""
Generate human-readable why-not eligibility reports.
Purely deterministic - no LLM.
"""

from src.matcher import EligibilityResult


def generate_why_not_report(result: EligibilityResult) -> str:
    """Generate a plain-text why-not report."""

    lines = []
    lines.append(f"Trial: {result.nct_id}")
    lines.append(f"Eligible: {'YES' if result.eligible else 'NO'}")
    lines.append("")

    if result.failing_criteria:
        lines.append("FAILING CRITERIA:")
        for i, r in enumerate(result.failing_criteria, 1):
            lines.append(
                f"  {i}. [{r.criterion.category.value}] {r.criterion.description}"
            )
            
            # --- CRITICAL FIX: Safe extraction of operator and value ---
            op_str = r.criterion.operator.value if r.criterion.operator else "N/A"
            val_str = r.criterion.value if r.criterion.value is not None else "N/A"
            
            # If it's an unmapped rule, it doesn't make sense to print "Patient value: [LLM Evaluated]" 
            # and "Required: N/A". We just print the fallback reason directly.
            if r.criterion.field == "unmapped_rule":
                lines.append("     Field: unmapped_rule")
                lines.append(f"     → {r.reason}")
            else:
                lines.append(f"     Patient value: {r.patient_value}")
                lines.append(f"     Required: {op_str} {val_str}")
                lines.append(f"     → {r.reason}")
            
            lines.append("")

    if result.indeterminate_criteria:
        lines.append("INDETERMINATE (missing patient data):")
        for i, r in enumerate(result.indeterminate_criteria, 1):
            lines.append(
                f"  {i}. [{r.criterion.category.value}] {r.criterion.description}"
            )
            lines.append(f"     Field: {r.criterion.field}")
            lines.append("     → Data not available in patient record")
            lines.append("")

    passing = [r for r in result.results if r.status == "pass"]
    lines.append(f"PASSING CRITERIA: {len(passing)} of {len(result.results)}")

    return "\n".join(lines)


def generate_why_not_json(result: EligibilityResult) -> dict:
    """Generate structured why-not report (for programmatic use)."""

    return {
        "nct_id": result.nct_id,
        "eligible": result.eligible,
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
                "description": r.criterion.description,
                "patient_value": r.patient_value,
                # --- CRITICAL FIX: Safely access the operator value ---
                "required_operator": r.criterion.operator.value if r.criterion.operator else None,
                "required_value": r.criterion.value,
            }
            for r in result.failing_criteria
        ],
        "missing_data": [
            {
                "criterion_id": r.criterion.criterion_id,
                "field": r.criterion.field,
                "description": r.criterion.description,
            }
            for r in result.indeterminate_criteria
        ],
    }
