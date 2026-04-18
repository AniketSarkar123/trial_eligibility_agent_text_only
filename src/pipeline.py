from __future__ import annotations

# Import the actual functions defined in your modules
from src.extractor import extract_patient
from src.trial_parser import parse_structured_fields, parse_free_text_criteria, ParsedTrial
from src.matcher import evaluate_eligibility
from src.why_not import generate_why_not_report

def run_pipeline(clinical_text: str, trial_json: dict, model: str = "qwen2.5:14b") -> dict:
    """
    Core pipeline logic: maps patient text and trial JSON to an eligibility result.
    No file I/O happens here.
    """
    
    # 1. Extract Patient Features using the LLM
    patient, metadata = extract_patient(clinical_text, model=model)
    
    # 2. Parse Trial Criteria
    nct_id = trial_json.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "UNKNOWN")
    title = trial_json.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", "UNKNOWN")
    
    conditions_module = trial_json.get("protocolSection", {}).get("conditionsModule", {})
    conditions = conditions_module.get("conditions", []) if conditions_module else []
    
    eligibility_text = trial_json.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")
    
    # Extract both structured and free-text criteria from the trial JSON
    criteria = parse_structured_fields(trial_json)
    criteria.extend(parse_free_text_criteria(eligibility_text, nct_id))
    
    parsed_trial = ParsedTrial(
        nct_id=nct_id, 
        title=title, 
        conditions=conditions, 
        criteria=criteria
    )
    
    # 3. Evaluate Eligibility
    result = evaluate_eligibility(patient, parsed_trial.criteria, parsed_trial.nct_id, clinical_text, model=model)
    
    # 4. Generate Plain-text Report
    report_text = generate_why_not_report(result)
    
    return {
        "patient": patient.model_dump(),
        "trial": parsed_trial.model_dump(),
        "result": result.model_dump(),
        "report_text": report_text, 
    }
