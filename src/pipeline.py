from __future__ import annotations

from src.extractor import extract_patient
from src.trial_parser import parse_structured_fields, parse_free_text_criteria, ParsedTrial
from src.matcher import evaluate_eligibility
from src.why_not import generate_why_not_report

def run_pipeline(clinical_text: str, trial_json: dict, model: str = "google/gemma-4-31b-it:free") -> dict:
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
    
    # Extract structured criteria
    structured_criteria = parse_structured_fields(trial_json)
    structured_fields = {c.field for c in structured_criteria}
    
    # Extract free-text criteria
    free_text_criteria = parse_free_text_criteria(eligibility_text, nct_id, model=model)
    
    # Deduplicate basic demographic criteria (age, sex) extracted by LLM
    # if parse_structured_fields already captured them.
    filtered_free_text = []
    for c in free_text_criteria:
        if c.field in ["age", "sex"] and c.field in structured_fields:
            continue  # Skip LLM-generated age/sex rule to prevent redundancy
        filtered_free_text.append(c)
        
    criteria = structured_criteria + filtered_free_text
    
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