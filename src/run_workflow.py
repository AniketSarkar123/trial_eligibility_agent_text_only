import json
from pathlib import Path

from src.ingest_trial import create_golden_trial
from src.batch_screener import screen_patient_against_golden, rank_cohort
from src.why_not import generate_why_not_report

# =====================================================================
# ⚙️ CONFIGURATION - CHANGE THESE VALUES TO CONTROL THE PIPELINE
# =====================================================================

# MODE can be either "INGEST" (parse a new trial) or "SCREEN" (test patients)
MODE = "SCREEN" 

# Which trial are we working with?
NCT_ID = "NCT04698252"

# Which model should we use?
MODEL = "qwen/qwen3-32b"

# --- SCREENING SETTINGS (Only used if MODE = "SCREEN") ---
# To run JUST ONE patient, put the full path to the file:
# TARGET_PATH = "data/narratives/patient_001.txt"
#
# To run EVERY patient in the folder, just put the folder path:
TARGET_PATH = "data/narratives/patient_001.txt" 

# =====================================================================
# 🚀 EXECUTION LOGIC (Do not change below this line)
# =====================================================================

def main():
    print(f"--- Running in {MODE} mode ---")
    
    if MODE == "INGEST":
        print(f"Ingesting Trial {NCT_ID} using {MODEL}...")
        create_golden_trial(NCT_ID, output_dir="data/golden_trials", model=MODEL)
        print(f"\n✅ Success! Please manually review 'data/golden_trials/{NCT_ID}_golden.json'")
        
    elif MODE == "SCREEN":
        golden_trial_path = Path(f"data/golden_trials/{NCT_ID}_golden.json")
        
        if not golden_trial_path.exists():
            print(f"❌ Error: Golden trial file not found at {golden_trial_path}")
            print(f"Please run INGEST mode for {NCT_ID} first!")
            return

        target = Path(TARGET_PATH)
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)

        if target.is_file() and target.suffix == ".txt":
            patient_files = [target]
        elif target.is_dir():
            patient_files = list(target.glob("*.txt"))
        else:
            print("❌ Error: TARGET_PATH must be a .txt file or a folder.")
            return

        batch_results = []
        print(f"Found {len(patient_files)} patients to screen against {NCT_ID}...")

        for patient_file in patient_files:
            patient_id = patient_file.stem
            print(f"Processing {patient_id}...")
            
            result = screen_patient_against_golden(
                patient_text=patient_file.read_text(encoding="utf-8"),
                golden_trial_path=str(golden_trial_path),
                model=MODEL
            )
            
            batch_results.append({"patient_id": patient_id, "result": result})
            
            # Save JSON report
            with open(out_dir / f"{patient_id}_{NCT_ID}_report.json", "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2)
                
            # Save Text report
            with open(out_dir / f"{patient_id}_{NCT_ID}_whynot.txt", "w", encoding="utf-8") as f:
                f.write(generate_why_not_report(result))

        print("\nRanking cohort based on clinical risk factors...")
        ranked_cohorts = rank_cohort(batch_results)
        
        print("\n======================================")
        print("      BATCH SCREENING SUMMARY")
        print("======================================")
        for priority, patients in ranked_cohorts.items():
            print(f"\n{priority}: {len(patients)} patients")
            for p in patients:
                res = p["result"]
                score_text = f" | Risk Score: {res.risk_score:.1f}" if not res.eligible and len(res.failing_criteria) == 0 else ""
                print(f"  - Patient ID: {p['patient_id']}{score_text}")

if __name__ == "__main__":
    main()
