# ============================================================
# Deterministic Matching Engine
# File: src/matcher.py
# ============================================================

"""
Deterministic matching: Patient JSON vs Trial criteria.
NO LLM involved. Pure Python logic.
"""

from pydantic import BaseModel

from schemas.patient import PatientProfile
from src.trial_parser import TrialCriterion, Operator


class MatchResult(BaseModel):
    """Result of evaluating one criterion against a patient."""

    criterion: TrialCriterion
    status: str  # "pass", "fail", "indeterminate"
    patient_value: str | int | float | bool | None
    reason: str


class EligibilityResult(BaseModel):
    """Full eligibility result for a patient-trial pair."""

    nct_id: str
    eligible: bool  # True only if ALL criteria pass
    has_indeterminate: bool
    results: list[MatchResult]
    failing_criteria: list[MatchResult]
    indeterminate_criteria: list[MatchResult]


def get_patient_value(patient: PatientProfile, field: str):
    """
    Resolve a criterion field name to the patient's value.
    Returns (value, found) tuple.
    """

    simple_fields = {
        "age", "sex", "menopausal_status", "cancer_stage", "is_metastatic",
        "er_status", "pr_status", "her2_status", "brca_status",
        "ecog_score", "ki67_percent", "pdl1_status", "brain_metastases",
        "lines_of_therapy", "prior_radiation", "prior_surgery",
        "adequate_liver_function", "adequate_renal_function",
        "adequate_bone_marrow",
    }

    if field in simple_fields:
        value = getattr(patient, field, None)
        return value, value is not None

    # Lab values
    if field.startswith("lab:"):
        test_name = field.split(":", 1)[1]
        for lab in patient.lab_values:
            if lab.test_name.lower() == test_name.lower():
                return lab.value, True
        return None, False

    # Prior drug
    if field.startswith("prior_drug:"):
        drug = field.split(":", 1)[1]
        for therapy in patient.prior_therapies:
            if therapy.drug_name.lower() == drug.lower():
                return True, True
        return False, True

    # Prior drug class
    if field.startswith("prior_drug_class:"):
        drug_class = field.split(":", 1)[1]
        for therapy in patient.prior_therapies:
            if therapy.drug_class and therapy.drug_class.lower() == drug_class.lower():
                return True, True
        return False, True

    return None, False


def evaluate_criterion(
    patient: PatientProfile,
    criterion: TrialCriterion,
) -> MatchResult:
    """Evaluate a single criterion against a patient."""

    value, found = get_patient_value(patient, criterion.field)

    if not found or value is None:
        return MatchResult(
            criterion=criterion,
            status="indeterminate",
            patient_value=None,
            reason=f"Patient data missing for '{criterion.field}'",
        )

    op = criterion.operator
    target = criterion.value
    passed = False

    try:
        if op == Operator.EQ:
            passed = _normalize(value) == _normalize(target)
        elif op == Operator.NEQ:
            passed = _normalize(value) != _normalize(target)
        elif op == Operator.GTE:
            passed = float(value) >= float(target)
        elif op == Operator.LTE:
            passed = float(value) <= float(target)
        elif op == Operator.GT:
            passed = float(value) > float(target)
        elif op == Operator.LT:
            passed = float(value) < float(target)
        elif op == Operator.IN:
            passed = _normalize(value) in [_normalize(v) for v in target]
        elif op == Operator.NOT_IN:
            passed = _normalize(value) not in [_normalize(v) for v in target]
        elif op == Operator.EXISTS:
            passed = found
        elif op == Operator.NOT_EXISTS:
            passed = not found
    except (ValueError, TypeError) as e:
        return MatchResult(
            criterion=criterion,
            status="indeterminate",
            patient_value=value,
            reason=f"Comparison error: {e}",
        )

    # Invert logic for exclusion criteria
    if not criterion.is_inclusion:
        passed = not passed

    return MatchResult(
        criterion=criterion,
        status="pass" if passed else "fail",
        patient_value=value,
        reason=_build_reason(criterion, value, passed),
    )


def _normalize(val) -> str:
    """Normalize values for comparison."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, bool):
        return str(val).lower()
    return str(val)


def _build_reason(criterion: TrialCriterion, patient_value, passed: bool) -> str:
    """Build human-readable reason string."""
    if passed:
        return (
            f"PASS: {criterion.field} = {patient_value} "
            f"(required: {criterion.operator.value} {criterion.value})"
        )
    return (
        f"FAIL: {criterion.field} = {patient_value} "
        f"(required: {criterion.operator.value} {criterion.value})"
    )


def evaluate_eligibility(
    patient: PatientProfile,
    criteria: list[TrialCriterion],
    nct_id: str,
) -> EligibilityResult:
    """Evaluate all trial criteria against a patient."""

    results = [evaluate_criterion(patient, c) for c in criteria]

    failing = [r for r in results if r.status == "fail"]
    indeterminate = [r for r in results if r.status == "indeterminate"]

    return EligibilityResult(
        nct_id=nct_id,
        eligible=len(failing) == 0 and len(indeterminate) == 0,
        has_indeterminate=len(indeterminate) > 0,
        results=results,
        failing_criteria=failing,
        indeterminate_criteria=indeterminate,
    )

