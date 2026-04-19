import re
from enum import Enum
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from src.extractor import get_client

class CriterionType(str, Enum):
    DEMOGRAPHIC = "demographic"
    BIOMARKER = "biomarker"
    CLINICAL = "clinical"
    LAB_VALUE = "lab_value"
    PRIOR_THERAPY = "prior_therapy"
    COMORBIDITY = "comorbidity"

class Operator(str, Enum):
    EQ = "eq"  
    NEQ = "neq"  
    GTE = "gte"  
    LTE = "lte"  
    GT = "gt"
    LT = "lt"
    IN = "in"  
    NOT_IN = "not_in"  
    EXISTS = "exists"  
    NOT_EXISTS = "not_exists"

ApprovedFields = Literal[
    "age", "sex", "menopausal_status", "primary_diagnosis", "cancer_stage", 
    "is_metastatic", "histology", "er_status", "pr_status", "her2_status", 
    "brca_status", "ki67_percent", "pdl1_status", "ecog_score", "lines_of_therapy", 
    "prior_radiation", "prior_surgery", "adequate_liver_function", 
    "adequate_renal_function", "adequate_bone_marrow", "brain_metastases",
    "tumor_size_cm", "nodal_status", "tumor_grade", "lymphovascular_invasion", 
    "disease_focality", "stil_score_percent", "pik3ca_mutation", "esr1_mutation",
    "received_neoadjuvant_therapy", "received_adjuvant_therapy", 
    "disease_free_interval_months", "has_recurrence", "unmapped_rule"
]

class TrialCriterion(BaseModel):
    criterion_id: str  
    category: CriterionType
    description: str  
    field: ApprovedFields | str = Field(description="MUST be an exact schema match or prefixed with 'lab:', 'prior_drug:', 'prior_drug_class:'")
    operator: Optional[Operator] = None
    
    # CRITICAL FIX: Add a strict description telling the LLM to NEVER use dictionaries here
    value: Optional[str | int | float | bool | list[str] | list[int] | list[float]] = Field(
        None, description="MUST be a primitive type (string, number, boolean, list). NEVER a dictionary/object."
    )
    
    unit: Optional[str] = None
    is_inclusion: bool = True  

class CriteriaList(BaseModel):
    criteria: list[TrialCriterion]

class ParsedTrial(BaseModel):
    nct_id: str
    title: str
    conditions: list[str]
    criteria: list[TrialCriterion]

def parse_structured_fields(trial_json: dict) -> list[TrialCriterion]:
    # ... (Keep your existing parse_structured_fields logic here) ...
    elig = trial_json["protocolSection"]["eligibilityModule"]
    criteria = []

    if min_age := elig.get("minimumAge"):
        age_val = int(min_age.split()[0]) 
        criteria.append(TrialCriterion(criterion_id="STRUCT_AGE_MIN", category=CriterionType.DEMOGRAPHIC, description=f"Minimum age: {min_age}", field="age", operator=Operator.GTE, value=age_val, unit="years"))

    if max_age := elig.get("maximumAge"):
        age_val = int(max_age.split()[0])
        criteria.append(TrialCriterion(criterion_id="STRUCT_AGE_MAX", category=CriterionType.DEMOGRAPHIC, description=f"Maximum age: {max_age}", field="age", operator=Operator.LTE, value=age_val, unit="years"))

    sex = elig.get("sex", "ALL")
    if sex != "ALL":
        criteria.append(TrialCriterion(criterion_id="STRUCT_SEX", category=CriterionType.DEMOGRAPHIC, description=f"Sex: {sex}", field="sex", operator=Operator.EQ, value=sex.lower()))

    return criteria

def parse_free_text_criteria(eligibility_text: str, nct_id: str, model: str = "qwen2.5:14b") -> list[TrialCriterion]:
    if not eligibility_text or not eligibility_text.strip(): return []
    client, _ = get_client(model=model)

    # CRITICAL FIX: Fortified system prompt with explicit unmapped_rule instructions
    system_prompt = """You are an expert clinical trial parser. Extract all criteria into a SINGLE list named `criteria`. 
    CRITICAL: DO NOT create separate "inclusion" and "exclusion" lists.

    CRITICAL RULES FOR THE 'field' NAME:
    You MUST map the criterion to EXACTLY ONE of these approved schema fields:
    [age, sex, menopausal_status, primary_diagnosis, cancer_stage, is_metastatic, histology, er_status, pr_status, her2_status, brca_status, ki67_percent, pdl1_status, ecog_score, lines_of_therapy, prior_radiation, prior_surgery, adequate_liver_function, adequate_renal_function, adequate_bone_marrow, brain_metastases, tumor_size_cm, nodal_status, tumor_grade, lymphovascular_invasion, disease_focality, stil_score_percent, pik3ca_mutation, esr1_mutation, received_neoadjuvant_therapy, received_adjuvant_therapy, disease_free_interval_months, has_recurrence, unmapped_rule]

    - For field mappings other than unmapped_rule, you MUST provide an `operator` and `value`.
    - The `value` field MUST be a primitive type (string, number, boolean, or list). NEVER use a nested JSON object/dictionary.
    
    WHEN TO USE 'unmapped_rule' (CRITICAL):
    1. COMPOSITE DISEASE RULES: If a rule requires a combination of factors (e.g. "Triple Negative Breast Cancer" which requires ER-, PR-, HER2-), you MUST use 'unmapped_rule'. 
    2. CONDITIONAL/STRATIFIED THRESHOLDS: If a requirement changes based on another variable (e.g., "sTILs >= 50% if age > 40, but >= 75% if age < 40"), you MUST use 'unmapped_rule'. Do NOT map it to 'stil_score_percent' because the standard schema cannot handle IF/THEN logic.
    3. COMPLEX RULES: Willingness to comply, multifocal disease, timing/intervals, or multiple conditions.
    
    - FOR unmapped_rule: You MUST set both `operator` and `value` to null. Do NOT try to encode logic into the value field.

    EXAMPLE OUTPUT FORMAT:
    {
      "criteria": [
        {
          "criterion_id": "EXAMPLE_1",
          "category": "clinical",
          "description": "Patient must have stage I or II cancer",
          "field": "cancer_stage",
          "operator": "in",
          "value": ["I", "II"],
          "is_inclusion": true
        },
        {
          "criterion_id": "EXAMPLE_2",
          "category": "clinical",
          "description": "Triple Negative Breast Cancer (ER-, PR-, HER2-)",
          "field": "unmapped_rule",
          "operator": null,
          "value": null,
          "is_inclusion": true
        }
      ]
    }
    """

    try:
        result = client.chat.completions.create(
            model=model,
            response_model=CriteriaList,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the criteria from this text:\n\n{eligibility_text}"}
            ],
            temperature=0.0, 
        )
        for i, c in enumerate(result.criteria):
            c.criterion_id = f"{nct_id}_LLM_{i+1}"
        return result.criteria
    except Exception as e:
        raise e
