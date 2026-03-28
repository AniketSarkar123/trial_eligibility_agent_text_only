from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extractor import extract_patient
from src.trial_fetcher import load_trial_json
from src.trial_parser import parse_trial_json
from src.matcher import evaluate
from src.why_not import generate

def run(patient_path: str, trial_path: str) -> dict:
    patient_text = Path(patient_path).read_text(encoding="utf-8")
    trial_raw = load_trial_json(trial_path)
    patient = extract_patient(patient_text)
    trial = parse_trial_json(trial_raw)
    result = evaluate(patient, trial)
    report = generate(patient, trial, result)
    return {
        "patient": patient.model_dump(),
        "trial": trial.model_dump(),
        "result": result.model_dump(),
        "report": report,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    output = run(args.patient, args.trial)
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
