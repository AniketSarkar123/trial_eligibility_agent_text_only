"""
Layer 1: Run extraction evaluation across all narratives and models.

Usage:
python -m eval.run_extraction_eval --model phi4-mini --prompt detailed
python -m eval.run_extraction_eval --model all --prompt all
"""

import json
from pathlib import Path
from itertools import groupby

import typer
from rich.console import Console
from rich.table import Table

from schemas.patient import PatientProfile
from src.extractor import extract_patient, get_client
from eval.metrics import (
    compute_extraction_metrics,
    compute_prior_therapy_metrics,
)

console = Console()

MODELS = [
    "smollm3:3b",           # 3B - smallest baseline
    "phi4-mini",            # 3.8B - primary reasoning model
    "phi4-mini-reasoning",  # 3.8B - explicit CoT reasoning
    "qwen3:4b",             # 4B - compact multilingual reasoning
    "deepseek-r1:7b",       # 7B - visible chain-of-thought
    "mistral:7b",           # 7B - general baseline (no reasoning)
    "qwen3:8b",             # 8B - strongest small reasoning
    "gemma2:9b",            # 9B - largest local model
]

PROMPTS = [
    "minimal",
    "detailed",
    "few_shot",
]


def run_single_extraction(
    narrative_path: Path,
    gold_path: Path,
    model: str,
    prompt_key: str,
) -> dict:
    """Run extraction on one narrative, compare to gold."""

    text = narrative_path.read_text()
    gold_data = json.loads(gold_path.read_text())
    gold = PatientProfile(**gold_data)

    # TODO: Load prompt template based on prompt_key
    predicted, metadata = extract_patient(text, model=model)

    extraction_metrics = compute_extraction_metrics(predicted, gold)

    therapy_metrics = compute_prior_therapy_metrics(
        [t.model_dump() for t in predicted.prior_therapies],
        gold_data.get("prior_therapies", []),
    )

    return {
        "narrative_id": narrative_path.stem,
        "model": model,
        "prompt": prompt_key,
        "extraction": extraction_metrics,
        "therapy": therapy_metrics,
        "metadata": metadata,
    }


def run_full_eval(model: str = "all", prompt: str = "all"):
    """Run extraction eval across all combinations."""

    narratives_dir = Path("data/narratives")
    gold_dir = Path("data/gold_labels/extractions")

    models = MODELS if model == "all" else [model]
    prompts = PROMPTS if prompt == "all" else [prompt]

    all_results = []

    for m in models:
        for p in prompts:
            console.print(f"\n[bold]Running: {m} / {p}[/bold]")

            for narrative_path in sorted(narratives_dir.glob("*.txt")):
                gold_path = gold_dir / f"{narrative_path.stem}.json"

                if not gold_path.exists():
                    console.print(
                        f" [yellow]Skipping {narrative_path.stem}: no gold label[/yellow]"
                    )
                    continue

                result = run_single_extraction(
                    narrative_path, gold_path, m, p
                )
                all_results.append(result)

                acc = result["extraction"]["aggregate"]["exact_match_accuracy"]
                hall = result["extraction"]["aggregate"]["hallucination_rate"]

                console.print(
                    f" {narrative_path.stem}: acc={acc:.2f}, halluc={hall:.2f}"
                )

    # Save results
    output = Path("results/extraction_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2, default=str))

    console.print(f"\n[green]Results saved to {output}[/green]")

    # Print summary table
    _print_summary_table(all_results)


def _print_summary_table(results: list[dict]):
    """Print a model x category accuracy table."""

    table = Table(title="Extraction Accuracy by Model and Category")

    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Overall")
    table.add_column("Demographic")
    table.add_column("Biomarker")
    table.add_column("Diagnosis")
    table.add_column("Treatment")
    table.add_column("Halluc. Rate")

    # Group results by (model, prompt)
    grouped = {}

    for r in results:
        key = (r["model"], r["prompt"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    for (model, prompt), group in grouped.items():
        def avg(cat):
            return sum(
                r["extraction"]["per_category"].get(cat, {}).get("accuracy", 0)
                for r in group
            ) / len(group)

        overall = sum(
            r["extraction"]["aggregate"]["exact_match_accuracy"]
            for r in group
        ) / len(group)

        halluc = sum(
            r["extraction"]["aggregate"]["hallucination_rate"]
            for r in group
        ) / len(group)

        table.add_row(
            model,
            prompt,
            f"{overall:.2f}",
            f"{avg('demographic'):.2f}",
            f"{avg('biomarker'):.2f}",
            f"{avg('diagnosis'):.2f}",
            f"{avg('treatment'):.2f}",
            f"{halluc:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    typer.run(run_full_eval)