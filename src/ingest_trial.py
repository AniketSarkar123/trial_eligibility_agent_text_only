import json
from pathlib import Path
from src.trial_fetcher import fetch_trial
from src.trial_parser import parse_structured_fields, parse_free_text_criteria, ParsedTrial

def create_golden_trial(nct_id: str, output_dir: str = "data/golden_trials", model: str = "qwen/qwen3-32b"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching {nct_id}...")
    raw_trial = fetch_trial(nct_id)
    
    title = raw_trial.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", "UNKNOWN")
    conditions = raw_trial.get("protocolSection", {}).get("conditionsModule", {}).get("conditions", [])
    eligibility_text = raw_trial.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")
    
    print("Extracting structured criteria...")
    structured = parse_structured_fields(raw_trial)
    structured_fields = {c.field for c in structured}
    
    print("Using LLM to extract free-text criteria...")
    unstructured = parse_free_text_criteria(eligibility_text, nct_id, model=model)
    
    # Deduplicate demographic criteria if structured already caught them
    filtered_unstructured = [
        c for c in unstructured 
        if c.field not in ["age", "sex"] or c.field not in structured_fields
    ]
    
    parsed_trial = ParsedTrial(
        nct_id=nct_id,
        title=title,
        conditions=conditions,
        criteria=structured + filtered_unstructured
    )
    
    output_path = Path(output_dir) / f"{nct_id}_golden.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_trial.model_dump(), f, indent=2)
        
    print(f"Golden trial saved to {output_path}. Ready for human review!")
