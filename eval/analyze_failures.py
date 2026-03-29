"""
Categorize extraction failures into a taxonomy.

This analysis is what pushes the paper from 'benchmark table' to 'insight'.

FAILURE CATEGORIES:
1. Negation errors - "no prior CDK4/6i" extracted as prior CDK4/6i present
2. Temporal errors - Timeline wrong (months, sequence)
3. Abbreviation errors - Failed to expand clinical abbreviations
4. Hallucination - Value invented when text is silent
5. Omission - Value present in text but extracted as null
6. Type confusion - e.g., BRCA VUS classified as mutated
7. Boundary errors - Numeric value off by small amount
8. Composite errors - Multiple issues in one field
9. Schema mismatch - Correct info but wrong field/format

For each model, count failures by category.
This tells readers WHERE each model struggles, not just how much.
"""

from enum import Enum


class FailureType(str, Enum):
    NEGATION = "negation"
    TEMPORAL = "temporal"
    ABBREVIATION = "abbreviation"
    HALLUCINATION = "hallucination"
    OMISSION = "omission"
    TYPE_CONFUSION = "type_confusion"
    BOUNDARY = "boundary"
    COMPOSITE = "composite"
    SCHEMA_MISMATCH = "schema_mismatch"


def classify_failure(
    field: str,
    predicted_value,
    gold_value,
    narrative_text: str,
) -> FailureType:
    """
    Classify a single extraction failure.

    NOTE:
    - Some cases can be auto-classified (hallucination, omission, boundary).
    - Others require manual review (negation, temporal, etc.).

    Recommendation:
    Auto-classify obvious cases and flag ambiguous ones for human annotation.
    """

    # Auto-classifiable cases
    if gold_value is None and predicted_value is not None:
        return FailureType.HALLUCINATION

    if gold_value is not None and predicted_value is None:
        return FailureType.OMISSION

    # Numeric boundary check (within 10%)
    try:
        if abs(float(predicted_value) - float(gold_value)) < abs(float(gold_value)) * 0.1:
            return FailureType.BOUNDARY
    except (ValueError, TypeError):
        pass

    # Default: requires manual classification
    return FailureType.COMPOSITE


def generate_failure_report(all_results: list[dict]) -> dict:
    """
    Generate per-model failure taxonomy report.

    OUTPUT:
    {
        "phi4-mini": {
            "negation": 12,
            "temporal": 8,
            "hallucination": 3,
            ...
        },
        "phi4-mini-reasoning": { ... },
        "deepseek-r1:7b": { ... },
        "qwen3:8b": { ... },
        ...
    }

    This becomes a stacked bar chart or heatmap in the paper.

    KEY ANALYSIS FOR REASONING MODELS:
    - Compare failure distributions between reasoning vs non-reasoning models
    - Hypothesis:
        * Reasoning models → fewer negation & temporal errors
        * Tradeoff → slower, more verbose

    - For deepseek-r1:
        Extract <think>...</think> traces and analyze correctness alignment.
    """

    # TODO: Implement aggregation logic
    pass