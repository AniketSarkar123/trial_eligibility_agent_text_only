"""
All evaluation metrics for the trial matching system.

Five evaluation layers:
L1: Extraction quality (LLM output vs gold JSON)
L2: Matching logic (gold JSON vs trial criteria)
L3: Why-not report quality
L4: End-to-end (raw text → eligibility decision)
L5: Ablations (cross-model, cross-prompt, with/without Instructor)
"""

import json
from typing import Any

from schemas.patient import PatientProfile


# ============================================================
# LAYER 1: EXTRACTION METRICS
# ============================================================

def field_exact_match(predicted: Any, gold: Any) -> bool:
    """Check if a single field value matches exactly."""
    if predicted is None and gold is None:
        return True
    if predicted is None or gold is None:
        return False

    # Normalize strings
    if isinstance(predicted, str) and isinstance(gold, str):
        return predicted.strip().lower() == gold.strip().lower()

    return predicted == gold


def compute_extraction_metrics(
    predicted: PatientProfile,
    gold: PatientProfile,
) -> dict:
    """
    Compare predicted patient JSON against gold standard.
    Returns per-field results and aggregate metrics.
    """

    SCALAR_FIELDS = {
        "demographic": ["age", "sex", "menopausal_status"],
        "diagnosis": [
            "primary_diagnosis",
            "cancer_stage",
            "is_metastatic",
            "histology",
        ],
        "biomarker": [
            "er_status",
            "pr_status",
            "her2_status",
            "brca_status",
            "ki67_percent",
            "pdl1_status",
        ],
        "clinical": ["ecog_score", "brain_metastases"],
        "treatment": ["lines_of_therapy", "prior_radiation", "prior_surgery"],
        "organ_function": [
            "adequate_liver_function",
            "adequate_renal_function",
            "adequate_bone_marrow",
        ],
    }

    results = {}

    for category, fields in SCALAR_FIELDS.items():
        for field in fields:
            pred_val = getattr(predicted, field, None)
            gold_val = getattr(gold, field, None)

            gold_is_null = gold_val is None
            pred_is_null = pred_val is None

            results[field] = {
                "category": category,
                "predicted": pred_val,
                "gold": gold_val,
                "exact_match": field_exact_match(pred_val, gold_val),

                # Null-handling classification
                "correct_null": gold_is_null and pred_is_null,
                "false_null": (not gold_is_null) and pred_is_null,
                "hallucinated": gold_is_null and (not pred_is_null),
                "both_present": (not gold_is_null) and (not pred_is_null),
            }

    # Aggregate metrics
    total = len(results)
    correct = sum(1 for r in results.values() if r["exact_match"])
    hallucinated = sum(1 for r in results.values() if r["hallucinated"])
    false_null = sum(1 for r in results.values() if r["false_null"])

    # Per-category metrics
    category_metrics = {}
    for cat in SCALAR_FIELDS:
        cat_fields = [r for r in results.values() if r["category"] == cat]
        cat_correct = sum(1 for r in cat_fields if r["exact_match"])

        category_metrics[cat] = {
            "accuracy": cat_correct / len(cat_fields) if cat_fields else 0,
            "total": len(cat_fields),
            "correct": cat_correct,
        }

    return {
        "per_field": results,
        "per_category": category_metrics,
        "aggregate": {
            "total_fields": total,
            "exact_match_accuracy": correct / total if total else 0,
            "hallucination_rate": hallucinated / total if total else 0,
            "false_null_rate": false_null / total if total else 0,
        },
    }


def compute_prior_therapy_metrics(
    predicted: list[dict],
    gold: list[dict],
) -> dict:
    """
    Evaluate prior therapy extraction.
    """

    gold_drugs = {t["drug_name"].lower(): t for t in gold}
    pred_drugs = {t["drug_name"].lower(): t for t in predicted}

    matched = set(gold_drugs) & set(pred_drugs)
    missed = set(gold_drugs) - set(pred_drugs)
    extra = set(pred_drugs) - set(gold_drugs)

    drug_recall = len(matched) / len(gold_drugs) if gold_drugs else 1.0
    drug_precision = len(matched) / len(pred_drugs) if pred_drugs else 1.0

    subfield_results = []

    for drug in matched:
        g = gold_drugs[drug]
        p = pred_drugs[drug]

        for field in [
            "drug_class",
            "is_current",
            "discontinued_reason",
            "months_since_last_dose",
        ]:
            subfield_results.append({
                "drug": drug,
                "field": field,
                "match": field_exact_match(p.get(field), g.get(field)),
            })

    return {
        "drug_recall": drug_recall,
        "drug_precision": drug_precision,
        "drugs_missed": list(missed),
        "drugs_hallucinated": list(extra),
        "subfield_accuracy": (
            sum(1 for s in subfield_results if s["match"]) /
            len(subfield_results)
            if subfield_results else 0
        ),
        "subfield_details": subfield_results,
    }


# ============================================================
# LAYER 2: MATCHING LOGIC METRICS
# ============================================================

def compute_matching_metrics(
    predicted_decisions: list[dict],
    gold_decisions: list[dict],
) -> dict:
    """Compare system decisions vs expert ground truth."""

    tp = fp = tn = fn = 0

    for pred, gold in zip(predicted_decisions, gold_decisions):
        assert pred["nct_id"] == gold["nct_id"]

        p = pred["eligible"]
        g = gold["eligible"]

        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and not g:
            tn += 1
        elif not p and g:
            fn += 1

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


# ============================================================
# LAYER 3: WHY-NOT REPORT METRICS
# ============================================================

def compute_whynot_metrics(
    predicted_failures: list[str],
    gold_failures: list[str],
) -> dict:
    """Evaluate why-not report quality."""

    pred_set = set(predicted_failures)
    gold_set = set(gold_failures)

    correct = pred_set & gold_set
    missed = gold_set - pred_set
    spurious = pred_set - gold_set

    completeness = len(correct) / len(gold_set) if gold_set else 1.0
    precision = len(correct) / len(pred_set) if pred_set else 1.0

    return {
        "completeness": completeness,
        "precision": precision,
        "correct_failures": list(correct),
        "missed_failures": list(missed),
        "spurious_failures": list(spurious),
    }


# ============================================================
# LAYER 4: END-TO-END METRICS
# ============================================================

def compute_e2e_metrics(
    e2e_decisions: list[dict],
    gold_decisions: list[dict],
    gold_input_decisions: list[dict],
) -> dict:
    """End-to-end evaluation + extraction error cost."""

    e2e = compute_matching_metrics(e2e_decisions, gold_decisions)
    gold_input = compute_matching_metrics(
        gold_input_decisions, gold_decisions
    )

    return {
        "e2e_accuracy": e2e["accuracy"],
        "e2e_sensitivity": e2e["sensitivity"],
        "e2e_specificity": e2e["specificity"],
        "gold_input_accuracy": gold_input["accuracy"],
        "gold_input_sensitivity": gold_input["sensitivity"],
        "extraction_error_cost": {
            "accuracy_drop": gold_input["accuracy"] - e2e["accuracy"],
            "sensitivity_drop": (
                gold_input["sensitivity"] - e2e["sensitivity"]
            ),
        },
        "e2e_confusion": e2e["confusion"],
    }