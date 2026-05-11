import json
from pathlib import Path
from src.extractor import extract_patient
from src.matcher import evaluate_eligibility, EligibilityResult
from src.trial_parser import ParsedTrial

def screen_patient_against_golden(patient_text: str, golden_trial_path: str, model: str) -> dict:
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
    
    # Return the composite payload including the patient details
    return {
        "patient": patient.model_dump(),
        "result": result
    }

def rank_cohort(batch_results: list[dict]) -> dict:
    """
    Takes a list of dictionaries with {"patient_id": str, "result": EligibilityResult}
    and sorts them into actionable clinical priority tiers based on triage status and risk scores.
    """
    # Tier 1: Ready to Screen (Fully Eligible OR purely missing Administrative data)
    eligible = [
        r for r in batch_results 
        if r["result"].triage_status in ["FULLY ELIGIBLE", "ELIGIBLE (Pending Administrative Verification)"]
    ]
    
    # Tier 2: The Indeterminates (Missing clinical data), sorted by lowest clinical risk score first
    indeterminates = [
        r for r in batch_results 
        if r["result"].triage_status == "POTENTIALLY ELIGIBLE (Needs Chart Review)"
    ]
    indeterminates.sort(key=lambda x: x["result"].risk_score)
    
    # Tier 3: Hard Fails (Has at least one clinical exclusion/failure)
    ineligible = [
        r for r in batch_results 
        if r["result"].triage_status == "INELIGIBLE"
    ]
    
    return {
        "Priority 1 (Ready to Screen)": eligible,
        "Priority 2 (Chart Review Needed - Sorted by Lowest Risk)": indeterminates,
        "Priority 3 (Rejected)": ineligible
    }
