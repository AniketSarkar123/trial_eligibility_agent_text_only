import re
from enum import Enum
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator
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
    "age", "sex", "menopausal_status", "pregnancy_status", "primary_diagnosis", "cancer_stage", 
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
    # Give it a default so the LLM doesn't have to generate it
    criterion_id: str = Field(default="TBD") 
    
    category: CriterionType = Field(
        description="CRITICAL: MUST be exactly one of: 'demographic', 'biomarker', 'clinical', 'lab_value', 'prior_therapy', 'comorbidity'. DO NOT use specific field names here (like 'primary_diagnosis')."
    )
    
    # Enforce the description field natively in the schema
    description: str = Field(
        ..., 
        description="CRITICAL: You MUST include the exact text snippet of the trial rule here. DO NOT OMIT THIS FIELD."
    )  
    field: ApprovedFields | str = Field(description="MUST be an exact schema match (e.g., 'primary_diagnosis', 'prior_surgery', 'age') or prefixed with 'lab:', 'prior_drug:', 'prior_drug_class:'")
    operator: Optional[Operator] = None
    
    value: Optional[str | int | float | bool | list[str] | list[int] | list[float]] = Field(
        None, description="MUST be a primitive type (string, number, boolean, list). NEVER a dictionary/object."
    )
    
    unit: Optional[str] = None

    timeframe_days: Optional[int] = Field(
        None, description="If a rule specifies a washout period (e.g., 'within 28 days', 'in the last 6 months'), convert it to days. Otherwise null."
    )
    group_id: Optional[str] = Field(None, description="Assign a shared ID (e.g., 'BRCA_GROUP') if this criterion is part of an OR statement.")
    group_operator: Optional[Literal["AND", "OR"]] = Field(None, description="Use 'OR' if passing ANY criterion in this group is sufficient.")

    is_inclusion: bool = True 

    # --- ADD THIS VALIDATOR ---
    @field_validator("category", mode="before")
    @classmethod
    def map_invalid_categories(cls, v):
        """Silently intercepts and corrects LLM hallucinations for the category field."""
        valid_categories = ["demographic", "biomarker", "clinical", "lab_value", "prior_therapy", "comorbidity"]
        
        # If it's already an enum instance, get its value
        v_str = str(getattr(v, 'value', v)).lower().strip()
        
        if v_str in valid_categories:
            return v_str
            
        # If the LLM hallucinates a specific 'field' name into the 'category' slot, map it correctly
        if v_str in ["primary_diagnosis", "cancer_stage", "histology", "is_metastatic", "ecog_score"]:
            return "clinical"
        if v_str in ["prior_surgery", "prior_radiation", "lines_of_therapy", "received_neoadjuvant_therapy"]:
            return "prior_therapy"
        if v_str in ["er_status", "pr_status", "her2_status", "brca_status", "pdl1_status"]:
            return "biomarker"
        if v_str in ["age", "sex", "menopausal_status"]:
            return "demographic"
            
        # Ultimate fallback to ensure Pydantic never crashes
        return "clinical" 

class CriteriaList(BaseModel):
    analysis: Optional[str] = Field(
        None, 
        description="Optional space for you to think or analyze prior errors before outputting the criteria."
    )
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

def parse_free_text_criteria(eligibility_text: str, nct_id: str, model: str = "openai/gpt-oss-20b") -> list[TrialCriterion]:
    if not eligibility_text or not eligibility_text.strip(): return []
    client, _ = get_client(model=model)

    system_prompt = """You are an expert clinical trial parser. Extract all criteria into a SINGLE list named criteria. 
    CRITICAL: DO NOT create separate "inclusion" and "exclusion" lists.
    CRITICAL: DO NOT include an "analysis", "thought", or reasoning field in your JSON output. Output ONLY the criteria array.
    
    CRITICAL TOOL CALLING RULE: The tool name is exactly CriteriaList. Do NOT prefix the tool name with functions.

    CRITICAL RULE FOR REQUIRED FIELDS:
    You MUST output the description field for EVERY criterion, containing the exact text of the rule. NEVER drop or omit the description field to save space.

    CRITICAL RULES FOR THE 'category' NAME:
    You MUST map the criterion to EXACTLY ONE of these approved categories:
    ["demographic", "biomarker", "clinical", "lab_value", "prior_therapy", "comorbidity"]

    CRITICAL RULES FOR THE 'field' NAME:
    You MUST map the criterion to EXACTLY ONE of these approved schema fields:
    [age, sex, menopausal_status, primary_diagnosis, cancer_stage, is_metastatic, histology, er_status, pr_status, her2_status, brca_status, ki67_percent, pdl1_status, ecog_score, lines_of_therapy, prior_radiation, prior_surgery, adequate_liver_function, adequate_renal_function, adequate_bone_marrow, brain_metastases, tumor_size_cm, nodal_status, tumor_grade, lymphovascular_invasion, disease_focality, stil_score_percent, pik3ca_mutation, esr1_mutation, received_neoadjuvant_therapy, received_adjuvant_therapy, disease_free_interval_months, has_recurrence, unmapped_rule]

    CRITICAL RULES FOR THE 'operator' FIELD:
    If you provide an operator, it MUST be EXACTLY one of these strings:
    ["eq", "neq", "gte", "lte", "gt", "lt", "in", "not_in", "exists", "not_exists"]
    DO NOT use abbreviations like "ge" (you must use "gte") or "le" (you must use "lte").

    - For field mappings other than unmapped_rule, you MUST provide an `operator` and `value`.
    - The `value` field MUST be a primitive type (string, number, boolean, or list). NEVER use a nested JSON object/dictionary.

    BIOMARKERS VS UNMAPPED RULE:
    - For simple, single biomarker checks (e.g., "Estrogen receptor positive" or "HER2 negative"), you MUST map them to the specific fields (`er_status`, `pr_status`, `her2_status`) with the operator `eq`.
    - COMPOSITE DISEASE RULES: If a rule requires a combination of factors in a single sentence (e.g., "Triple Negative Breast Cancer" meaning ER-, PR-, HER2-) you MUST use 'unmapped_rule'.

    BOOLEAN GROUPING (OR LOGIC):
    - If a rule contains an "OR" condition (e.g., "Requires BRCA1 OR BRCA2 mutation"), split them into separate criteria, give them the same group_id (e.g., 'GRP_1'), and set group_operator to 'OR'.
    
    WASHOUT PERIODS & TIMEFRAMES:
    - If a criterion specifies a restricted time window, washout period, or expiration (e.g., "within the last 6 months", "prior to 28 days", "in the past year"), you MUST calculate the approximate number of days and map it to the `timeframe_days` field.
    - Conversion Examples: "6 months" = 180, "4 weeks" = 28, "1 year" = 365, "12 weeks" = 84.
    - If no timeframe is explicitly mentioned in the rule, you MUST set `timeframe_days` to null. Do not guess.

    WHEN TO USE 'unmapped_rule' (CRITICAL):
    1. COMPOSITE DISEASE RULES: If a rule requires a combination of factors (e.g. "Triple Negative Breast Cancer" which requires ER-, PR-, HER2-), you MUST use 'unmapped_rule'. 
       -> CRITICAL: If a rule combines a disease diagnosis with a specific stage requirement (e.g., "Stage I-III breast cancer" or "Pathological stage I TNBC"), you MUST map it to 'unmapped_rule'.
    2. CONDITIONAL/STRATIFIED THRESHOLDS: If a requirement changes based on another variable (e.g., "sTILs >= 50% if age > 40, but >= 75% if age < 40"), you MUST use 'unmapped_rule'. Do NOT map it to 'stil_score_percent' because the standard schema cannot handle IF/THEN logic.
    3. COMPLEX RULES: Willingness to comply, multifocal disease, timing/intervals, or multiple conditions.
    4. PREGNANCY & CONTRACEPTION: If a rule relates to pregnancy, lactation, or contraception, you MUST map it to `unmapped_rule`. Do NOT force it into menopausal_status.
    5. COMPLEX STAGING: If the staging requirement involves complex TNM staging (e.g., T1N1-3) or sub-stages that might fail simple exact matches, use `unmapped_rule`.
       
    - FOR unmapped_rule: You MUST set both `operator` and `value` to null. Do NOT try to encode logic into the value field.
    CRITICAL RULES FOR 'is_inclusion':
    - If a rule is listed under an "Inclusion Criteria" section, you MUST set `is_inclusion` to `true`.
    - If a rule is listed under an "Exclusion Criteria" section, or explicitly states the patient must NOT have a condition, you MUST set `is_inclusion` to `false`.
    

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
          "category": "comorbidity",
          "description": "History of severe diseases or HIV",
          "field": "unmapped_rule",
          "operator": null,
          "value": null,
          "is_inclusion": false
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
            max_tokens=16384,
        )
        for i, c in enumerate(result.criteria):
            c.criterion_id = f"{nct_id}_LLM_{i+1}"
        return result.criteria
    except Exception as e:
        raise e
