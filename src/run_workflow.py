import json
from pathlib import Path

# Import the pipeline function
from src.pipeline import run_pipeline

def main():
    # 1. Load the input parameters (patient text and trial JSON)
    patient_text = Path("data/narratives/patient_001.txt").read_text(encoding="utf-8")
    
    with open("data/trials/NCT00000000.json", "r", encoding="utf-8") as f:
        trial_dict = json.load(f)

    print("Running pipeline...")
    
    # 2. Call the run_pipeline function explicitly
    # Make sure you have Ollama running with the phi4-mini model
    pipeline_output = run_pipeline(
        clinical_text=patient_text,
        trial_json=trial_dict,
        model="llama3" 
    )

    # 3. Create a results directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Dump the report and save it to a file
    output_file = output_dir / "patient_001_NCT00000000_report.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        # json.dump saves the Python dictionary as formatted JSON
        json.dump(pipeline_output, f, indent=2)

    print(f"Success! Report saved to: {output_file}")
    
    # Optional: Print the plain-text why-not report to the console
    print("\n--- Why-Not Report ---")
    print(pipeline_output["report_text"])

if __name__ == "__main__":
    main()
