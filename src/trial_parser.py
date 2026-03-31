import re
from enum import Enum
from pydantic import BaseModel
import instructor
from openai import OpenAI

# Import the Ollama client setup from your extractor
from src.extractor import get_client

class CriterionType(str, Enum):
    DEMOGRAPHIC = "demographic"
    BIOMARKER = "biomarker"
    CLINICAL = "clinical"
    LAB_VALUE = "lab_value"
    PRIOR_THERAPY = "prior_therapy"
    COMORBIDITY = "comorbidity"

class Operator(str, Enum):
    EQ = "eq"  # equals
    NEQ = "neq"  # not equals
    GTE = "gte"  # >=
    LTE = "lte"  # <=
    GT = "gt"
    LT = "lt"
    IN = "in"  # value in list
    NOT_IN = "not_in"  # value not in list
    EXISTS = "exists"  # field must be present
    NOT_EXISTS = "not_exists"

class TrialCriterion(BaseModel):
    """A single matchable criterion from a trial."""
    criterion_id: str  # e.g., "NCT00001234_INC_03"
    category: CriterionType
    description: str  # human-readable original text
    field: str  # maps to patient schema field name
    operator: Operator
    value: str | int | float | list | bool | None
    unit: str | None = None
    is_inclusion: bool = True  # True=inclusion, False=exclusion

class CriteriaList(BaseModel):
    """Wrapper class for Instructor to return a list of criteria."""
    criteria: list[TrialCriterion]

class ParsedTrial(BaseModel):
    """A trial with all criteria structured."""
    nct_id: str
    title: str
    conditions: list[str]
    criteria: list[TrialCriterion]

def parse_structured_fields(trial_json: dict) -> list[TrialCriterion]:
    """
    Extract criteria from the already-structured parts of the trial JSON.
    These are the "easy" ones - age, sex, healthy volunteers.
    """
    elig = trial_json["protocolSection"]["eligibilityModule"]
    criteria = []

    # Age range
    if min_age := elig.get("minimumAge"):
        age_val = int(min_age.split()[0])  # "18 Years" → 18
        criteria.append(
            TrialCriterion(
                criterion_id="STRUCT_AGE_MIN",
                category=CriterionType.DEMOGRAPHIC,
                description=f"Minimum age: {min_age}",
                field="age",
                operator=Operator.GTE,
                value=age_val,
                unit="years",
            )
        )

    if max_age := elig.get("maximumAge"):
        age_val = int(max_age.split()[0])
        criteria.append(
            TrialCriterion(
                criterion_id="STRUCT_AGE_MAX",
                category=CriterionType.DEMOGRAPHIC,
                description=f"Maximum age: {max_age}",
                field="age",
                operator=Operator.LTE,
                value=age_val,
                unit="years",
            )
        )

    # Sex
    sex = elig.get("sex", "ALL")
    if sex != "ALL":
        criteria.append(
            TrialCriterion(
                criterion_id="STRUCT_SEX",
                category=CriterionType.DEMOGRAPHIC,
                description=f"Sex: {sex}",
                field="sex",
                operator=Operator.EQ,
                value=sex.lower(),
            )
        )

    return criteria

def parse_free_text_criteria(
    eligibility_text: str,
    nct_id: str,
    model: str = "llama3"
) -> list[TrialCriterion]:
    """
    Parse the free-text eligibility block into structured criteria using an LLM.
    """
    if not eligibility_text or not eligibility_text.strip():
        return []

    # Reuse your Ollama client setup
    client, _ = get_client(model=model)

    # The System Prompt acts as the new "Regex"
   system_prompt = """You are an expert clinical trial parser. Extract all criteria into a SINGLE list named `criteria`. 
    CRITICAL: DO NOT create separate "inclusion_criteria" and "exclusion_criteria" lists. Put everything into the one `criteria` list.

    CRITICAL RULES FOR 'category':
    - You MUST use exactly one of these: 'demographic', 'biomarker', 'clinical', 'lab_value', 'prior_therapy', 'comorbidity'.
    - NEVER use "inclusion" or "exclusion" as a category.

    CRITICAL RULES FOR THE 'field' NAME:
    - For lab tests, prefix with 'lab: ' (e.g., 'lab: hemoglobin').
    - For prior drug classes, use 'prior_drug_class:<class>'.
    - For specific prior drugs, use 'prior_drug:<drug>'.
    - For standard oncology fields, use EXACTLY: 'menopausal_status', 'her2_status', 'ecog_score', 'brain_metastases', 'is_metastatic'.
    - If a criterion is about complex diseases, administrative rules, or things that don't fit the standard fields, set the field name to 'unmapped_rule'.

    CRITICAL RULES FOR 'is_inclusion':
    - Set to `true` if it is an Inclusion Criterion.
    - Set to `false` if it is an Exclusion Criterion.
    """

    try:
        # Ask the LLM to extract the criteria matching your Pydantic schema
        result = client.chat.completions.create(
            model=model,
            response_model=CriteriaList,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the criteria from this text:\n\n{eligibility_text}"}
            ],
            temperature=0.0, # Keep temperature 0 for strict extraction
        )
        
        extracted_criteria = result.criteria
        
        # Post-process: Inject the NCT ID into the criterion_id sequentially
        for i, c in enumerate(extracted_criteria):
            c.criterion_id = f"{nct_id}_LLM_{i+1}"
            
        return extracted_criteria
        
    except Exception as e:
        print(f"Error parsing criteria with LLM: {e}")
        return []
