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
    model: str = "openai/gpt-oss-20b"
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
    CRITICAL: You MUST output your response using the provided JSON tool schema. Do NOT output plain text.
    1. If the patient clearly passes the rule based on the narrative or profile, set the `status` field to "pass".
    2. If the patient clearly fails the rule based on the narrative or profile, set the `status` field to "fail".
    3. If the information is completely missing from both, set the `status` field to "indeterminate".

    CRITICAL SAFETY GUARDS, MISSING DATA & SOURCE OF TRUTH:
    - TUNNEL VISION (CRITICAL): You must evaluate ONLY the specific TRIAL RULE provided. Ignore any statements in the narrative about the patient's OVERALL eligibility for the trial. Do NOT fail a rule about "Cancer Stage" just because the text says they are ineligible due to "Prior Therapy". Evaluate the rule in total isolation.
    - IGNORE META-COMMENTARY: The clinical narrative may contain "spoilers" written by a human grader (e.g., "this renders the patient ineligible", "she does not fulfill criteria"). You MUST completely ignore these statements. Treat the text as a raw medical record. Do not use the word "ineligible" from the text to fail administrative rules like "consent" or "researcher discretion."
    - THE NARRATIVE IS KING: If the STRUCTURED PATIENT PROFILE contradicts the RAW CLINICAL NARRATIVE (e.g., the JSON says "Stage IV" but the narrative says "Stage II"), you MUST trust the RAW CLINICAL NARRATIVE. The JSON was auto-extracted and may contain hallucinations.
    - MISSING = INDETERMINATE: If the data required to answer the rule is missing, you MUST return "indeterminate". Do NOT return "fail" just because data is missing. 
    - MEDICAL EXCLUSIONS: If the rule excludes patients with specific prior diseases (e.g., HIV, Hepatitis, heart failure, other cancers) and the text is SILENT on these conditions, you MUST return 'indeterminate'. Do NOT assume the patient does not have them just because they aren't mentioned.
    - ADMINISTRATIVE/SUBJECTIVE RULES: If the rule requires "signed consent," "willingness to comply," or a specific "life expectancy," and the text does not explicitly address it, return 'indeterminate'. Do NOT hallucinate clinical data (like tumor shrinkage) to justify passing an administrative rule.
    - Do NOT estimate or guess quantitative numbers from vague qualitative words.
    - PREGNANCY: If a rule excludes pregnant/lactating women, and the text does NOT explicitly state "not pregnant", you MUST return 'indeterminate'. Do NOT assume they are not pregnant just because they are consenting to therapy. If stated "not pregnant", they pass the non-pregnant rule.
    - CONTRACEPTION: If a rule requires contraception, and the text does NOT explicitly state they are using it or plan to use it, you MUST return 'indeterminate'. 
    - Do NOT substitute qualitative symptoms for formal clinical tests. If a rule requires a specific assessment score (e.g., TICS-M, MoCA, ECOG) and that exact test score is missing from the patient profile, you MUST return 'indeterminate', regardless of the patient's symptoms.
    - TIMEFRAME RANGES (CRITICAL): If a rule provides a range of time (e.g., "3 to 36 months", "between 14 and 28 days"), you MUST map it to `unmapped_rule` and set `timeframe_days` to null. Do NOT try to calculate a single number for a range.

    For the `reason` field in the JSON, provide a brief, 1-sentence reason referencing specific patient facts.
    """
    
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=RuleEvaluation,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
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
        "age", "sex", "menopausal_status", "pregnancy_status", "primary_diagnosis", "cancer_stage", 
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
        test_name = resolved_field.split(":", 1)[1].strip().lower()
        
        # Clinical synonym mapping for labs
        lab_aliases = {
            "plt": "platelets",
            "hgb": "hemoglobin",
            "hb": "hemoglobin",
            "anc": "absolute neutrophil count",
            "wbc": "white blood cells",
            "ast": "aspartate aminotransferase",
            "alt": "alanine aminotransferase",
            "cr": "creatinine"
        }
        
        # Normalize the target test name
        target_name = lab_aliases.get(test_name, test_name)
        
        for lab in patient.lab_values:
            # Normalize the extracted test name
            extracted_name = lab_aliases.get(lab.test_name.lower(), lab.test_name.lower())
            
            if extracted_name == target_name:
                return lab.value, True
        return None, False

    if resolved_field.startswith("prior_drug:"):
        drug = resolved_field.split(":", 1)[1].strip().lower()
        
        for therapy in patient.prior_therapies:
            if therapy.drug_name.lower() == drug:
                # WASHOUT LOGIC:
                # If the trial has a washout (e.g. 28 days) and the patient had the drug...
                if hasattr(criterion, 'timeframe_days') and criterion.timeframe_days:
                    if therapy.days_since_last_dose is not None:
                        if therapy.days_since_last_dose <= criterion.timeframe_days:
                            return True, True # VIOLATION: Had drug within the restricted window
                        else:
                            return False, True # SAFE: Had drug, but it was outside the washout period
                    else:
                        return None, False # We know they had it, but don't know WHEN -> Indeterminate
                return True, True # No washout period specified, standard check
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

    # --- CRITICAL FIX: THE BULLETPROOF INTERCEPTOR ---
    is_stubborn_parser_error = (
        criterion.field == "primary_diagnosis" and 
        criterion.value is not None and 
        "stage" in str(criterion.value).lower()
    )

    needs_llm_fallback = (
        criterion.field == "unmapped_rule" or 
        (criterion.operator is None and criterion.value is None) or
        is_stubborn_parser_error # Forces the reroute
    )

    if needs_llm_fallback:
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

    # --- STANDARD DETERMINISTIC MATCHING ---
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

    # Safety Guard for partial missing operators (e.g., op is None but target exists)
    if op is None and not passed:
        return MatchResult(
            criterion=criterion, 
            status="indeterminate", 
            patient_value=value, 
            reason="Comparison error: Missing mathematical operator in trial criterion extraction."
        )

    try:
        if not passed and op is not None:
            # --- UNIT MISMATCH GUARD ---
            patient_unit = None
            resolved_field = criterion.field.lower().strip()
            if resolved_field.startswith("lab:"):
                test_name = resolved_field.split(":", 1)[1].strip().lower()
                lab_aliases = {
                    "plt": "platelets", "hgb": "hemoglobin", "hb": "hemoglobin",
                    "anc": "absolute neutrophil count", "wbc": "white blood cells",
                    "ast": "aspartate aminotransferase", "alt": "alanine aminotransferase",
                    "cr": "creatinine"
                }
                target_name = lab_aliases.get(test_name, test_name)
                for lab in patient.lab_values:
                    extracted_name = lab_aliases.get(lab.test_name.lower(), lab.test_name.lower())
                    if extracted_name == target_name:
                        patient_unit = lab.unit
                        break
                        
            if patient_unit and criterion.unit and patient_unit.lower().strip() != criterion.unit.lower().strip():
                # Reroute to the LLM to handle the complex mathematical conversion
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
                    patient_value=f"{value} {patient_unit}", 
                    reason=f"Unit Mismatch ({patient_unit} vs {criterion.unit}) → LLM Fallback: {llm_eval.reason}"
                )

            # --- NUMERIC & STRING COMPARISONS ---
            if op == Operator.EQ: passed = _normalize(value) == _normalize(target)
            elif op == Operator.NEQ: passed = _normalize(value) != _normalize(target)
            elif op == Operator.GTE: passed = float(value) >= float(target)
            elif op == Operator.LTE: passed = float(value) <= float(target)
            elif op == Operator.GT: passed = float(value) > float(target)
            elif op == Operator.LT: passed = float(value) < float(target)
            elif op == Operator.IN: 
                # SMART INTERCEPTOR: Allow hierarchical staging bypass (e.g. "IIIA" passes an in ["II", "III"] check)
                if criterion.field == "cancer_stage" and isinstance(value, str) and isinstance(target, list):
                    passed = any(value.upper().startswith(str(t).upper()) for t in target)
                else:
                    passed = _normalize(value) in [_normalize(v) for v in target]
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
    model: str,
    exhaustive: bool = True  # <-- ADDED TOGGLE (Defaults to True for detailed study)
) -> EligibilityResult:
    
    # PASS 1: Purely deterministic "Simple" rules (Python)
    simple_criteria = [c for c in criteria if c.field != "unmapped_rule"]
    # PASS 2: Complex rules (LLM)
    complex_criteria = [c for c in criteria if c.field == "unmapped_rule"]
    
    results = []
    has_failed_simple = False
    
    # Evaluate Pass 1 (Deterministic)
    for c in simple_criteria:
        result = evaluate_criterion(patient, c, raw_patient_text, model)
        results.append(result)
        
        if result.status == "fail":
            has_failed_simple = True
            # If we are NOT doing an exhaustive search, abort immediately
            if not exhaustive:
                return EligibilityResult(
                    nct_id=nct_id, eligible=False, has_indeterminate=False, 
                    results=results, failing_criteria=[result], indeterminate_criteria=[]
                )
            
    # Evaluate Pass 2 (LLM Fallbacks)
    # If not exhaustive AND we already failed a simple rule, skip these expensive calls!
    if not (not exhaustive and has_failed_simple):
        for c in complex_criteria:
            result = evaluate_criterion(patient, c, raw_patient_text, model)
            results.append(result)
            
            if result.status == "fail" and not exhaustive:
                # Fail-fast triggered during complex rules
                failing = [r for r in results if r.status == "fail"]
                indeterminate = [r for r in results if r.status == "indeterminate"]
                return EligibilityResult(
                    nct_id=nct_id, eligible=False, has_indeterminate=len(indeterminate) > 0, 
                    results=results, failing_criteria=failing, indeterminate_criteria=indeterminate
                )

    # --- BOOLEAN GROUPING LOGIC (OR) ---
    grouped_results = {}
    for r in results:
        if hasattr(r.criterion, 'group_id') and r.criterion.group_id:
            if r.criterion.group_id not in grouped_results:
                grouped_results[r.criterion.group_id] = []
            grouped_results[r.criterion.group_id].append(r)

    for group_id, group_items in grouped_results.items():
        operator = group_items[0].criterion.group_operator
        if operator == "OR":
            if any(item.status == "pass" for item in group_items):
                for item in group_items:
                    item.status = "pass"
                    item.reason = f"Group {group_id} (OR) condition met by another criterion."

    # Final Compilation
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
