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
    # TODO: Implement chosen approach
    # For now, return empty - you'll supplement with manual annotation
    # of trial criteria for your evaluation set
    return []