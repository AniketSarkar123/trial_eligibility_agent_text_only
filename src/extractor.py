"""
Patient feature extraction using Instructor + Groq (primary) or
llama-cpp-python (alternative).

This is the ONLY file that touches the LLM.
"""

import os
import re
import time
import json

import instructor
from groq import Groq

from schemas.patient import PatientProfile

# ============================================================
# CLIENT SETUP
# ============================================================

def get_client(
    model: str = "qwen3-32b",
) -> tuple[instructor.Instructor, str]:
    """
    Create an Instructor-wrapped client for Groq.
    """
    api_key = "GROQ_API_KEY"
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your Colab secrets.")

    client = instructor.from_groq(
        Groq(api_key=api_key), 
        mode=instructor.Mode.TOOLS,  
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

SYSTEM_PROMPT = """You are a highly precise clinical data extraction agent. Your job is to read an unstructured clinical narrative about a cancer patient and extract structured data perfectly matching the provided JSON schema.

CRITICAL WORKFLOW:
1. PRESERVE THE SCHEMA: You MUST output EVERY key defined in the expected JSON schema. If information is missing, set the value to `null` or `false`, but DO NOT drop the key entirely.

CRITICAL TOOL CALLING RULE: The tool name is exactly PatientProfile. Do NOT prefix the tool name with functions.

CLINICAL EXTRACTION RULES:
1. STRICT SILENCE (NO GUESSING): If a field is completely unmentioned in the text, set it to `null`. Do NOT infer, guess, or assume based on standard medical practices (e.g., do not guess pregnancy status based on age).
2. TEMPORAL BOUNDARIES: Never extract planned, future, or consented treatments as prior history. If a text says 'planned mastectomy', 'consents to chemotherapy', or 'candidate for neoadjuvant', they have NOT received it yet.
3. THE "ZERO" RULE: If the text explicitly states the patient has received "no systemic treatment", "no prior therapy", or "never received treatment", you MUST set `lines_of_therapy` to `0`. Do not set it to `null`.
4. EXHAUSTIVE DRUG CAPTURE: You must extract EVERY specific drug name mentioned and add it as an object to the `prior_therapies` array (e.g., [{"drug_name": "letrozole"}]), even if they are current treatments. Use generic names in lowercase.
5. PREGNANCY EXTRACTION: If the text explicitly says 'not pregnant', use 'not_pregnant'. If the text is SILENT on pregnancy, you MUST output `null`. 
6. BREAST CANCER SPECIFICS: Look specifically for and extract sTIL scores (%), Neoadjuvant vs Adjuvant therapy sequences, Recurrence status, and Lymphovascular invasion (LVI).
"""

REASONING_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Think step-by-step about what information is present in the text before
producing the JSON output. After reasoning, output ONLY the JSON object.
"""


def extract_patient(
    clinical_text: str,
    model: str = "qwen3-32b",
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
        max_tokens=16384,
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
