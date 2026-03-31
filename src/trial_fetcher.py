"""
Fetch clinical trials from ClinicalTrials.gov API v2.
Returns structured JSON with eligibility criteria already partially parsed.
"""

import requests
import json
from pathlib import Path

API_BASE = "https://clinicaltrials.gov/api/v2/studies"

def fetch_trial(nct_id: str) -> dict:
    """Fetch a single full trial JSON by NCT ID."""
    url = f"{API_BASE}/{nct_id}"
    
    # We just need format=json to get the full raw data
    response = requests.get(url, params={"format": "json"})
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch trial {nct_id}: Status {response.status_code}")


def search_trials(
    condition: str = "breast cancer",
    phase: str = "PHASE2|PHASE3",
    status: str = "RECRUITING",
    max_results: int = 20,
) -> list[dict]:
    """
    Search for trials matching criteria.
    Returns list of trial JSON objects.
    """
    params = {
        "query.cond": condition,
        "filter.phase": phase,
        "filter.overallStatus": status,
        "pageSize": max_results,
        "format": "json",
    }

    response = requests.get(API_BASE, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("studies", [])
    else:
         raise Exception(f"Failed to search trials: Status {response.status_code}")


def save_trial(trial: dict, output_dir: Path) -> Path:
    """Save trial JSON to data/trials/."""
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the NCT ID from the JSON to use as the filename
    nct_id = trial["protocolSection"]["identificationModule"]["nctId"]
    path = output_dir / f"{nct_id}.json"

    # Save the file beautifully formatted
    path.write_text(json.dumps(trial, indent=2))
    return path
