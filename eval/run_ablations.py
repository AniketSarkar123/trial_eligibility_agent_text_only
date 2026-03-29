"""
Layer 5: Ablation studies.

Four ablation axes:
1. Model size: 3B → 9B + cloud ceiling
2. Reasoning mode: non-reasoning vs CoT reasoning vs thinking-mode
3. Schema enforcement: Instructor (retry) vs grammar-constrained (GBNF) vs raw JSON prompt
4. Prompt complexity: minimal vs detailed vs few-shot

This script runs the full matrix and produces comparison tables.
"""

import json
from pathlib import Path
from itertools import product


MODELS = [
    "smollm3:3b",
    "phi4-mini",
    "phi4-mini-reasoning",
    "qwen3:4b",
    "deepseek-r1:7b",
    "mistral:7b",
    "qwen3:8b",
    "gemma2:9b",
]

PROMPTS = [
    "minimal",
    "detailed",
    "few_shot",
]

SCHEMA_MODES = [
    "instructor_retry",
    "instructor_no_retry",
    "grammar_constrained",
    "raw_json_prompt",
]


def run_ablation_matrix():
    """
    Run extraction for every (model, prompt, schema_mode) combination.

    For 'raw_json_prompt':
        Ask LLM to output JSON → manually parse → measure raw capability.

    For 'instructor_no_retry':
        Instructor with max_retries=0 → isolates retry benefit.

    For 'grammar_constrained':
        Use GBNF (llama.cpp) → guarantees valid JSON structure but not correctness.

    REASONING MODE ABLATION:
        Compare reasoning ON vs OFF for:
        - phi4-mini-reasoning
        - deepseek-r1
        - qwen3 (thinking mode)

        Also analyze <think> traces vs correctness.

    EXPECTED OUTPUT:
        - Table 1: Model size vs accuracy
        - Table 2: Reasoning vs non-reasoning
        - Table 3: Schema enforcement vs format compliance
        - Table 4: Prompt vs accuracy
        - Table 5: Full matrix
    """

    results = []

    for model, prompt, schema_mode in product(MODELS, PROMPTS, SCHEMA_MODES):
        # TODO: Run extraction over all narratives for this config
        # (Reuse logic from run_extraction_eval)
        pass

    # Save results
    output = Path("results/ablation_matrix.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))


def compute_format_compliance(raw_outputs: list[str]) -> float:
    """
    For raw_json_prompt mode:
    Fraction of outputs that are valid JSON matching schema.

    This is a KEY metric for Instructor ablation.
    """

    valid = 0

    for output in raw_outputs:
        try:
            from schemas.patient import PatientProfile

            data = json.loads(output)
            PatientProfile(**data)
            valid += 1
        except Exception:
            pass

    return valid / len(raw_outputs) if raw_outputs else 0