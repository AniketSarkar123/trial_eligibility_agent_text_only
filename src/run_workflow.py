import json
from pathlib import Path

# Import both the pipeline AND your fetcher functions
from src.pipeline import run_pipeline
from src.trial_fetcher import fetch_trial, save_trial

def run_automated_workflow(patient_filepath: str | Path, nct_id: str, model: str = "llama3"):
    """
    Runs the eligibility pipeline for a given patient file and trial,
    saving both JSON and text reports dynamically.
    """
    patient_path = Path(patient_filepath)
    
    # Dynamically extract the patient ID (e.g., "patient_001" from "patient_001.txt")
    patient_id = patient_path.stem 
    patient_text = patient_path.read_text(encoding="utf-8")
    
    print(f"Fetching trial {nct_id} from ClinicalTrials.gov API...")
    trial_dict = fetch_trial(nct_id)
    
    # Save the fetched trial to your local folder
    save_trial(trial_dict, Path("data/trials"))

    print("Running pipeline...")
    
    pipeline_output = run_pipeline(
        clinical_text=patient_text,
        trial_json=trial_dict,
        model=model 
    )

    # Create a results directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save the structured JSON report dynamically
    json_output_file = output_dir / f"{patient_id}_{nct_id}_report.json"
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, indent=2)
    print(f"Success! JSON Report saved to: {json_output_file}")
    
    # 2. Save the plain-text why-not report dynamically
    text_output_file = output_dir / f"{patient_id}_{nct_id}_whynot.txt"
    with open(text_output_file, "w", encoding="utf-8") as f:
        f.write(pipeline_output["report_text"])
    print(f"Success! Why-Not Text Report saved to: {text_output_file}")

    # Optionally keep printing to the console
    print("\n--- Why-Not Report ---")
    print(pipeline_output["report_text"])


def main():
    # Now you can easily loop through directories or pass different files
    # For a single run:
    run_automated_workflow(
        patient_filepath="data/narratives/patient_001.txt",
        nct_id="NCT04698252",
        model="llama3" 
    )

if __name__ == "__main__":
    main()
