"""
Patient feature extraction using Instructor + Ollama (primary) or
llama-cpp-python (alternative).

This is the ONLY file that touches the LLM.
"""

import re
import time
import json

import instructor
from openai import OpenAI

from schemas.patient import PatientProfile

# ============================================================
# CLIENT SETUP
# ============================================================


def get_client(
    model: str = "phi4-reasoning:plus",
    base_url: str = "http://localhost:11434/v1",
) -> tuple[instructor.Instructor, str]:
    """
    Create an Instructor-wrapped client for an Ollama-hosted model.
    """
    client = instructor.from_openai(
        OpenAI(base_url=base_url, api_key="ollama"),  # Ollama ignores API key
        mode=instructor.Mode.JSON,  # JSON mode for compatibility
    )
    return client, model


def get_cloud_client(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> tuple[instructor.Instructor, str]:
    """
    Create client for cloud model (ceiling reference).
    """
    import anthropic

    client = instructor.from_anthropic(anthropic.Anthropic())
    return client, model


# ============================================================
# REASONING MODEL HANDLING
# ============================================================

REASONING_MODELS = {"phi4-mini-reasoning", "deepseek-r1:7b", "deepseek-r1:14b"}
THINKING_MODE_MODELS = {"qwen3:4b", "qwen3:8b"}


def strip_reasoning_tokens(raw_output: str) -> tuple[str, str | None]:
    """
    Strip chain-of-thought reasoning from model output before JSON parsing.
    """
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    reasoning_trace = think_match.group(1).strip() if think_match else None

    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    return cleaned, reasoning_trace


# ============================================================
# MAIN EXTRACTION
# ============================================================

SYSTEM_PROMPT = """You are a clinical data extraction system. Your job is to read
a clinical narrative about a cancer patient and extract structured data.

RULES:
1. CAREFULLY read the text and extract ALL explicitly mentioned data. 
2. You MUST extract standard demographics, stage, and biomarkers (ER/PR/HER2).
3. NEW BREAST CANCER RULES: You MUST look for and extract:
   - sTIL scores (stromal tumor-infiltrating lymphocytes) as a percentage.
   - Neoadjuvant vs Adjuvant therapy sequences.
   - Recurrence status (has the cancer returned?)
   - Lymphovascular invasion (LVI).
4. If a field is completely unmentioned, set it to null. Do NOT infer or guess.
5. For drug names, use generic names in lowercase.
"""

REASONING_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Think step-by-step about what information is present in the text before
producing the JSON output. After reasoning, output ONLY the JSON object.
"""


def extract_patient(
    clinical_text: str,
    model: str = "phi4-reasoning:plus",
    client: instructor.Instructor | None = None,
    temperature: float = 0.0,
) -> tuple[PatientProfile, dict]:
    """
    Extract patient features from clinical narrative.
    """
    if client is None:
        client, model = get_client(model)

    is_reasoning = model in REASONING_MODELS
    system_prompt = REASONING_SYSTEM_PROMPT if is_reasoning else SYSTEM_PROMPT

    start = time.time()

    profile = client.chat.completions.create(
        model=model,
        response_model=PatientProfile,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Extract patient data from this clinical text:\n\n{clinical_text}",
            },
        ],
        temperature=temperature,
        max_retries=3,
    )

    elapsed = time.time() - start

    metadata = {
        "model": model,
        "is_reasoning_model": is_reasoning,
        "extraction_time_seconds": elapsed,
        "temperature": temperature,
    }

    return profile, metadata


# ============================================================
# ALTERNATIVE: llama-cpp-python with GBNF grammar
# ============================================================


def extract_patient_grammar(
    clinical_text: str,
    model_path: str = "./models/phi-4-mini-Q4_K_M.gguf",
    grammar_path: str = "./schemas/patient.gbnf",
) -> tuple[PatientProfile, dict]:
    """
    Extract using llama-cpp-python with GBNF grammar constraint.
    """
    from llama_cpp import Llama, LlamaGrammar

    start = time.time()

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )

    grammar = LlamaGrammar.from_file(grammar_path)

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extract patient data:\n\n{clinical_text}",
            },
        ],
        grammar=grammar,
        temperature=0.0,
    )

    elapsed = time.time() - start

    raw_json = output["choices"][0]["message"]["content"]
    profile = PatientProfile(**json.loads(raw_json))

    metadata = {
        "model": model_path,
        "backend": "llama-cpp-python",
        "schema_enforcement": "grammar_constrained",
        "extraction_time_seconds": elapsed,
        "retries": 0,
    }

    return profile, metadata
