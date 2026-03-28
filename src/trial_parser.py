# # These paths into the trial JSON are what you'll parse.
# # Study this structure carefully - it's your "free" structured side.
# trial["protocolSection"]["eligibilityModule"]
# # Contains:
# # "eligibilityCriteria": str - free text (inclusion/exclusion block)
# # "sex": "ALL" | "FEMALE" | "MALE"
# # "minimumAge": "18 Years"
# # "maximumAge": "75 Years"
# # "stdAges": ["ADULT", "OLDER_ADULT"]
# # "healthyVolunteers": False
# trial["protocolSection"]["conditionsModule"]
# # Contains:
# # "conditions": ["Breast Cancer", "HER2-negative Breast Cancer"]
# # "keywords": [...]
# trial["protocolSection"]["armsInterventionsModule"]
# # Contains:
# # "interventions": [{"name": "Talazoparib", "type": "DRUG", ...}]


"""
Parse structured trial JSON into a list of matchable criteria.
Some criteria come pre-structured (age, sex).
The free-text eligibilityCriteria block needs further parsing.
"""

from enum import Enum

from pydantic import BaseModel


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
) -> list[TrialCriterion]:
    """
    Parse the free-text eligibility block into structured criteria.

    APPROACH OPTIONS (choose one):
    Option A - Use an LLM with Instructor to parse. This means BOTH sides
    use an LLM and your eval must account for that.

    Option B - Rule-based / regex parsing for common patterns.
    More limited but fully deterministic.

    Option C - Hybrid: rules for common patterns, LLM for complex ones.

    RECOMMENDATION for the paper: Start with Option B for common criteria
    (lab values, ECOG, menopausal status). Document which criteria types
    you can and cannot parse. This keeps the "LLM only does patient
    extraction" story clean.

    If you go with Option A, you MUST evaluate this separately and report
    it as a potential confound.
    """
    criteria = []
    if not eligibility_text:
        return criteria

    # State tracker to determine if current line is inclusion or exclusion
    is_inclusion_block = True
    criterion_idx = 1
    
    # Helper to map text operators to the Enum
    op_map = {
        ">=": Operator.GTE,
        ">": Operator.GT,
        "<=": Operator.LTE,
        "<": Operator.LT,
        "=": Operator.EQ,
        "==": Operator.EQ
    }

    lines = eligibility_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 1. Block State Management
        if "Inclusion Criteria" in line:
            is_inclusion_block = True
            continue
        elif "Exclusion Criteria" in line:
            is_inclusion_block = False
            continue
            
        # 2. Pattern Match: ECOG Performance Status
        # e.g., "ECOG performance status <= 1"
        ecog_match = re.search(r'ECOG.*?([<>=]+)\s*(\d+)', line, re.IGNORECASE)
        if ecog_match:
            op_str, val_str = ecog_match.groups()
            criteria.append(TrialCriterion(
                criterion_id=f"{nct_id}_FREE_{criterion_idx}",
                category=CriterionType.CLINICAL,
                description=line,
                field="ecog_score",
                operator=op_map.get(op_str, Operator.LTE),
                value=int(val_str),
                is_inclusion=is_inclusion_block
            ))
            criterion_idx += 1
            continue

        # 3. Pattern Match: Lab Values
        # Maps common lab names to their corresponding patient schema fields
        lab_patterns = [
            (r'Hemoglobin', 'lab: hemoglobin', 'g/dL'),
            (r'ANC|Absolute Neutrophil Count', 'lab: ANC', 'cells/uL'),
            (r'Platelets', 'lab: platelets', '/uL'),
            (r'Creatinine', 'lab: creatinine', 'mg/dL'),
            (r'Bilirubin', 'lab: total bilirubin', 'mg/dL'),
        ]
        
        matched_lab = False
        for lab_name, field_name, default_unit in lab_patterns:
            # Matches text like "Hemoglobin >= 10 g/dL"
            lab_match = re.search(rf'{lab_name}.*?([<>=]+)\s*([\d.]+)\s*([a-zA-Z/%0-9]+)?', line, re.IGNORECASE)
            if lab_match:
                op_str, val_str, unit_str = lab_match.groups()
                criteria.append(TrialCriterion(
                    criterion_id=f"{nct_id}_FREE_{criterion_idx}",
                    category=CriterionType.LAB_VALUE,
                    description=line,
                    field=field_name,
                    operator=op_map.get(op_str, Operator.GTE),
                    value=float(val_str),
                    unit=unit_str if unit_str else default_unit,
                    is_inclusion=is_inclusion_block
                ))
                criterion_idx += 1
                matched_lab = True
                break
        if matched_lab:
            continue
            
        # 4. Pattern Match: Menopausal Status
        if re.search(r'postmenopausal', line, re.IGNORECASE):
            criteria.append(TrialCriterion(
                criterion_id=f"{nct_id}_FREE_{criterion_idx}",
                category=CriterionType.DEMOGRAPHIC,
                description=line,
                field="menopausal_status",
                operator=Operator.EQ,
                value="postmenopausal",
                is_inclusion=is_inclusion_block
            ))
            criterion_idx += 1
            continue
            
        # 5. Pattern Match: HER2 Status
        if re.search(r'HER2.?(negative|-)', line, re.IGNORECASE):
            criteria.append(TrialCriterion(
                criterion_id=f"{nct_id}_FREE_{criterion_idx}",
                category=CriterionType.BIOMARKER,
                description=line,
                field="her2_status",
                operator=Operator.EQ,
                value="negative",
                is_inclusion=is_inclusion_block
            ))
            criterion_idx += 1
            continue
            
        # 6. Pattern Match: Brain Metastases
        if re.search(r'brain metastas(es|is)', line, re.IGNORECASE):
            # For exclusions, having anything other than "none" is a fail condition
            criteria.append(TrialCriterion(
                criterion_id=f"{nct_id}_FREE_{criterion_idx}",
                category=CriterionType.CLINICAL,
                description=line,
                field="brain_metastases",
                operator=Operator.NEQ,  # Triggers if patient has active/stable mets
                value="none",
                is_inclusion=is_inclusion_block 
            ))
            criterion_idx += 1
            continue

        # 7. Pattern Match: Prior Therapies (e.g. CDK4/6 inhibitors)
        prior_class_match = re.search(r'(prior|previous).*?(CDK4/6|aromatase|taxane)', line, re.IGNORECASE)
        if prior_class_match:
            drug_class = prior_class_match.group(2).strip().lower()
            
            # Check if the rule is forbidding the drug ("No prior...")
            if "no " in line.lower() or not is_inclusion_block:
                criteria.append(TrialCriterion(
                    criterion_id=f"{nct_id}_FREE_{criterion_idx}",
                    category=CriterionType.PRIOR_THERAPY,
                    description=line,
                    field=f"prior_drug_class:{drug_class} inhibitor", 
                    operator=Operator.EQ,
                    value=False,  
                    is_inclusion=True # Treat "No prior X" as an inclusion requirement that Prior=False
                ))
                criterion_idx += 1

    return criteria
