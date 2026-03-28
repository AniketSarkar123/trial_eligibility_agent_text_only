"""
Fetch clinical trials from ClinicalTrials.gov API v2.
Returns structured JSON with eligibility criteria already partially parsed.
"""

import httpx
import json
from pathlib import Path

API_BASE = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trial(nct_id: str) -> dict:
    """Fetch a single trial by NCT ID."""
    url = f"{API_BASE}/{nct_id}"
    resp = httpx.get(url, params={"format": "json"})
    resp.raise_for_status()
    return resp.json()


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

    resp = httpx.get(API_BASE, params=params)
    resp.raise_for_status()
    data = resp.json()

    return data.get("studies", [])


def save_trial(trial: dict, output_dir: Path) -> Path:
    """Save trial JSON to data/trials/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    nct_id = trial["protocolSection"]["identificationModule"]["nctId"]
    path = output_dir / f"{nct_id}.json"

    path.write_text(json.dumps(trial, indent=2))
    return path


