import json
from pathlib import Path
from src.extractor import extract_patient
from src.matcher import evaluate_eligibility, EligibilityResult
from src.trial_parser import ParsedTrial

def screen_patient_against_golden(patient_text: str, golden_trial_path: str, model: str) -> EligibilityResult:
    # 1. Load the human-verified Golden Trial
    with open(golden_trial_path, "r") as f:
        golden_data = json.load(f)
    trial = ParsedTrial(**golden_data)
    
    # 2. Extract Patient Features (LLM call)
    patient, metadata = extract_patient(patient_text, model=model)
    
    # 3. Deterministic Matching using verified criteria
    result = evaluate_eligibility(
        patient=patient, 
        criteria=trial.criteria, 
        nct_id=trial.nct_id, 
        raw_patient_text=patient_text, 
        model=model
    )
    
    return result

def rank_cohort(batch_results: list[dict]) -> dict:
    """
    Takes a list of dictionaries with {"patient_id": str, "result": EligibilityResult}
    and sorts them into actionable clinical priority tiers based on risk scores.
    """
    # Tier 1: Perfect Passes
    eligible = [r for r in batch_results if r["result"].eligible]
    
    # Tier 2: The Indeterminates (0 Fails), sorted by lowest risk score first
    indeterminates = [r for r in batch_results if not r["result"].eligible and len(r["result"].failing_criteria) == 0]
    indeterminates.sort(key=lambda x: x["result"].risk_score)
    
    # Tier 3: Hard Fails
    ineligible = [r for r in batch_results if len(r["result"].failing_criteria) > 0]
    
    return {
        "Priority 1 (Ready to Screen)": eligible,
        "Priority 2 (Chart Review Needed - Sorted by Lowest Risk)": indeterminates,
        "Priority 3 (Rejected)": ineligible
    }