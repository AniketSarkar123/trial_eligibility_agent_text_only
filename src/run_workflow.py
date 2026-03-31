import json
from pathlib import Path

# Import both the pipeline AND your fetcher functions
from src.pipeline import run_pipeline
from src.trial_fetcher import fetch_trial, save_trial

def main():
    # 1. Load the patient text
    patient_text = Path("data/narratives/patient_001.txt").read_text(encoding="utf-8")
    
    # 2. FETCH THE TRIAL DYNAMICALLY
    nct_id = "NCT06203314"  # You can change this to any valid NCT ID!
    print(f"Fetching trial {nct_id} from ClinicalTrials.gov API...")
    trial_dict = fetch_trial(nct_id)
    
    # Optional: Save the fetched trial to your local folder so you don't have to fetch it next time
    save_trial(trial_dict, Path("data/trials"))

    print("Running pipeline...")
    
    # 3. Call the run_pipeline function exactly as before
    pipeline_output = run_pipeline(
        clinical_text=patient_text,
        trial_json=trial_dict,
        model="llama3" 
    )

    # 4. Create a results directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save the report dynamically using the NCT ID in the filename
    output_file = output_dir / f"patient_001_{nct_id}_report.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, indent=2)

    print(f"Success! Report saved to: {output_file}")
    
    # Print the plain-text why-not report to the console
    print("\n--- Why-Not Report ---")
    print(pipeline_output["report_text"])

if __name__ == "__main__":
    main()
