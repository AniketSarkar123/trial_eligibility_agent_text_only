"""
Hybrid matching: Patient JSON vs Trial criteria.
Uses pure Python logic for structured data, and LLM fallback for unmapped rules.
"""

import json
from typing import Literal
from pydantic import BaseModel

from schemas.patient import PatientProfile
from src.trial_parser import TrialCriterion, Operator
from src.extractor import get_client  # Ensure this points to your client setup


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


# --- LLM Fallback Models and Functions ---
class RuleEvaluation(BaseModel):
    """How the LLM responds when evaluating an unmapped rule."""
    status: Literal["pass", "fail", "indeterminate"]
    reason: str


def evaluate_unmapped_rule_with_llm(
    patient_json: str, 
    criterion_desc: str, 
    is_inclusion: bool, 
    model: str = "llama3"
) -> RuleEvaluation:
    """Uses the LLM to read the description and evaluate it against the patient."""
    client, _ = get_client(model=model)
    
    rule_type = "Inclusion Criterion (Patient MUST meet this)" if is_inclusion else "Exclusion Criterion (Patient MUST NOT meet this)"
    
    prompt = f"""You are an expert oncologist. Determine if the patient meets this complex clinical trial rule.
    
    PATIENT PROFILE:
    {patient_json}
    
    TRIAL RULE:
    {criterion_desc}
    Rule Type: {rule_type}
    
    INSTRUCTIONS:
    1. If the patient clearly passes the rule based on the profile, return "pass".
    2. If the patient clearly fails the rule based on the profile, return "fail".
    3. If the patient profile lacks the information needed to evaluate the rule, return "indeterminate".
    Provide a brief, 1-sentence reason referencing specific patient facts.
    """
    
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=RuleEvaluation,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_retries=3
        )
        return result
    except Exception as e:
        return RuleEvaluation(status="indeterminate", reason=f"LLM fallback failed: {e}")


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

    # --- NEW: Intercept unmapped rules for LLM evaluation ---
    if criterion.field == "unmapped_rule":
        patient_data_str = patient.model_dump_json(exclude_none=True)
        llm_eval = evaluate_unmapped_rule_with_llm(
            patient_json=patient_data_str,
            criterion_desc=criterion.description,
            is_inclusion=criterion.is_inclusion
        )
        
        return MatchResult(
            criterion=criterion,
            status=llm_eval.status,
            patient_value="[LLM Evaluated]",  # Special marker for the why-not report
            reason=f"LLM Fallback: {llm_eval.reason}"
        )

    # --- Standard Deterministic Evaluation ---
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
