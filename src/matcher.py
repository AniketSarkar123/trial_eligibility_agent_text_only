"""
Hybrid matching: Patient JSON vs Trial criteria.
Uses pure Python logic for structured data, and LLM fallback for unmapped rules.
"""

import json
from typing import Literal
from pydantic import BaseModel

from schemas.patient import PatientProfile
from src.trial_parser import TrialCriterion, Operator
from src.extractor import get_client


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


class RuleEvaluation(BaseModel):
    """How the LLM responds when evaluating an unmapped rule."""
    status: Literal["pass", "fail", "indeterminate"]
    reason: str


def evaluate_unmapped_rule_with_llm(
    patient_json: str, 
    raw_patient_text: str, 
    criterion_desc: str, 
    is_inclusion: bool, 
    model: str = "qwen2.5:14b"
) -> RuleEvaluation:
    """Uses the LLM to read the description and evaluate it against the patient."""
    client, _ = get_client(model=model)
    
    rule_type = "Inclusion Criterion (Patient MUST meet this)" if is_inclusion else "Exclusion Criterion (Patient MUST NOT meet this)"
    
    prompt = f"""You are an expert oncologist. Determine if the patient meets this complex clinical trial rule.
    
    RAW CLINICAL NARRATIVE:
    {raw_patient_text}
    
    STRUCTURED PATIENT PROFILE:
    {patient_json}
    
    TRIAL RULE:
    {criterion_desc}
    Rule Type: {rule_type}
    
    INSTRUCTIONS:
    1. If the patient clearly passes the rule based on the narrative or profile, return "pass".
    2. If the patient clearly fails the rule based on the narrative or profile, return "fail".
    3. If the information is completely missing from both, return "indeterminate".
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
    
    # Alias Resolution Map 
    alias_map = {
        "gender": "sex",
        "tumor_type": "primary_diagnosis",
        "pathologic_stage": "cancer_stage",
        "metastasis_status": "is_metastatic",
        "surgical_status": "prior_surgery",
        "performance_status": "ecog_score",
        "neoadjuvant_therapy": "received_neoadjuvant_therapy",
        "neoadjuvant_systemic_therapy": "received_neoadjuvant_therapy",
        "adjuvant_therapy": "received_adjuvant_therapy",
        "stil_score": "stil_score_percent",
        "recurrence_status": "has_recurrence",
        "lvi": "lymphovascular_invasion",
        "tumor_focality": "disease_focality"
    }
    
    resolved_field = field.lower().strip()
    resolved_field = alias_map.get(resolved_field, resolved_field)

    simple_fields = {
        "age", "sex", "menopausal_status", "primary_diagnosis", "cancer_stage", 
        "is_metastatic", "histology", "er_status", "pr_status", "her2_status", 
        "brca_status", "ki67_percent", "pdl1_status", "ecog_score", "brain_metastases",
        "lines_of_therapy", "prior_radiation", "prior_surgery",
        "adequate_liver_function", "adequate_renal_function", "adequate_bone_marrow",
        "tumor_size_cm", "nodal_status", "tumor_grade", "lymphovascular_invasion", 
        "disease_focality", "stil_score_percent", "pik3ca_mutation", "esr1_mutation",
        "received_neoadjuvant_therapy", "received_adjuvant_therapy", 
        "disease_free_interval_months", "has_recurrence"
    }

    if resolved_field in simple_fields:
        value = getattr(patient, resolved_field, None)
        return value, value is not None

    if resolved_field.startswith("lab:"):
        test_name = resolved_field.split(":", 1)[1].strip()
        for lab in patient.lab_values:
            if lab.test_name.lower() == test_name.lower():
                return lab.value, True
        return None, False

    if resolved_field.startswith("prior_drug:"):
        drug = resolved_field.split(":", 1)[1].strip()
        for therapy in patient.prior_therapies:
            if therapy.drug_name.lower() == drug.lower():
                return True, True
        return False, True

    if resolved_field.startswith("prior_drug_class:"):
        drug_class = resolved_field.split(":", 1)[1].strip()
        for therapy in patient.prior_therapies:
            if therapy.drug_class and therapy.drug_class.lower() == drug_class.lower():
                return True, True
        return False, True

    return None, False


def evaluate_criterion(
    patient: PatientProfile, 
    criterion: TrialCriterion, 
    raw_patient_text: str, 
    model: str
) -> MatchResult:
    """Evaluate a single criterion against a patient."""

    if criterion.field == "unmapped_rule":
        patient_data_str = patient.model_dump_json(exclude_none=True)
        llm_eval = evaluate_unmapped_rule_with_llm(
            patient_json=patient_data_str,
            raw_patient_text=raw_patient_text,
            criterion_desc=criterion.description,
            is_inclusion=criterion.is_inclusion,
            model=model
        )
        return MatchResult(
            criterion=criterion, status=llm_eval.status,
            patient_value="[LLM Evaluated]", reason=f"LLM Fallback: {llm_eval.reason}"
        )

    value, found = get_patient_value(patient, criterion.field)

    if not found or value is None:
        return MatchResult(
            criterion=criterion, 
            status="indeterminate", 
            patient_value=None, 
            reason=f"Patient data missing for '{criterion.field}'"
        )

    op = criterion.operator
    target = criterion.value
    passed = False

    # TYPE COERCION FIX FOR BOOLEAN MISMATCHES
    if isinstance(value, bool) and isinstance(target, str):
        if target.lower() in ["true", "yes", "positive"]:
            target = True
        elif target.lower() in ["false", "no", "negative"]:
            target = False
        elif op == Operator.EQ and value is True:
            passed = True

    # TRAP 1 FIX: THE LAZY PARSER SAFETY GUARD
    if op is None and not passed:
        if target is None and found:
            # If the LLM just mapped the field (e.g. field="sex") but gave no target,
            # and the patient HAS a value for that field, default to Pass.
            passed = True
        else:
            return MatchResult(
                criterion=criterion, 
                status="indeterminate", 
                patient_value=value, 
                reason="Comparison error: Missing mathematical operator in trial criterion extraction."
            )

    try:
        if not passed and op is not None:
            if op == Operator.EQ: passed = _normalize(value) == _normalize(target)
            elif op == Operator.NEQ: passed = _normalize(value) != _normalize(target)
            elif op == Operator.GTE: passed = float(value) >= float(target)
            elif op == Operator.LTE: passed = float(value) <= float(target)
            elif op == Operator.GT: passed = float(value) > float(target)
            elif op == Operator.LT: passed = float(value) < float(target)
            elif op == Operator.IN: passed = _normalize(value) in [_normalize(v) for v in target]
            elif op == Operator.NOT_IN: passed = _normalize(value) not in [_normalize(v) for v in target]
            elif op == Operator.EXISTS: passed = found
            elif op == Operator.NOT_EXISTS: passed = not found
    except (ValueError, TypeError) as e:
        return MatchResult(
            criterion=criterion, 
            status="indeterminate", 
            patient_value=value, 
            reason=f"Comparison error: {e}"
        )

    if not criterion.is_inclusion:
        passed = not passed

    return MatchResult(
        criterion=criterion, 
        status="pass" if passed else "fail", 
        patient_value=value, 
        reason=_build_reason(criterion, value, passed)
    )


def _normalize(val) -> str:
    if isinstance(val, str): return val.strip().lower()
    if isinstance(val, bool): return str(val).lower()
    return str(val)


def _build_reason(criterion: TrialCriterion, patient_value, passed: bool) -> str:
    status_str = "PASS" if passed else "FAIL"
    
    # TRAP 1 FIX: Safely extract operator and value without crashing
    op_str = criterion.operator.value if criterion.operator else "N/A"
    val_str = criterion.value if criterion.value is not None else "N/A"
    
    return f"{status_str}: {criterion.field} = {patient_value} (required: {op_str} {val_str})"


def evaluate_eligibility(
    patient: PatientProfile, 
    criteria: list[TrialCriterion], 
    nct_id: str, 
    raw_patient_text: str, 
    model: str
) -> EligibilityResult:
    """Evaluate all trial criteria against a patient."""
    
    results = [evaluate_criterion(patient, c, raw_patient_text, model) for c in criteria]
    
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
