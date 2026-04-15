"""
Patient feature schema.

This is the Pydantic model that the LLM must populate from clinical text.

DESIGN RULES:
1. Every field must correspond to something trials actually filter on.
2. Use Optional for fields that may not be mentioned in text.
3. Use Literal types for controlled vocabularies.
4. Add field descriptions - Instructor passes these to the LLM.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# --- Sub-models ---


class PriorTherapy(BaseModel):
    """A single prior treatment the patient received."""

    drug_name: str = Field(description="Generic drug name, lowercase")

    drug_class: str | None = Field(
        None,
        description=(
            "Drug class, e.g., 'CDK4/6 inhibitor', 'aromatase inhibitor', "
            "'taxane'. Null if unknown."
        ),
    )

    is_current: bool = Field(
        description="True if patient is currently on this therapy"
    )

    discontinued_reason: str | None = Field(
        None,
        description=(
            "Reason stopped: 'progression', 'toxicity', 'completed', "
            "'patient choice', or null"
        ),
    )

    months_since_last_dose: float | None = Field(
        None,
        description=(
            "Months since last dose. Null if not mentioned or if currently "
            "on therapy."
        ),
    )


class LabValue(BaseModel):
    """A single laboratory result."""

    test_name: str = Field(
        description="Standardized test name, e.g., 'hemoglobin', 'creatinine', 'ANC'"
    )

    value: float = Field(description="Numeric result")

    unit: str = Field(
        description="Unit of measurement, e.g., 'g/dL', 'mg/dL', 'cells/uL'"
    )


class Comorbidity(BaseModel):
    """A comorbid condition."""

    condition: str = Field(
        description="Condition name, e.g., 'type 2 diabetes', 'hypertension'"
    )

    active: bool = Field(
        description="True if currently active/being treated"
    )


# --- Main Patient Schema ---


class PatientProfile(BaseModel):
    """
    Structured patient profile extracted from clinical narrative.

    CRITICAL INSTRUCTION FOR LLM:
    - If information is NOT mentioned in the text, set the field to null.
    - Do NOT guess or infer values not explicitly stated.
    - For biomarkers, only report what is explicitly stated in the text.
    """

    # Demographics
    age: int | None = Field(None, description="Patient age in years")

    sex: Literal["male", "female"] | None = Field(
        None, description="Biological sex"
    )

    menopausal_status: Literal[
        "premenopausal", "perimenopausal", "postmenopausal"
    ] | None = Field(
        None,
        description="Menopausal status. Null if not mentioned or not applicable.",
    )

    # Diagnosis
    primary_diagnosis: str | None = Field(
        None,
        description=(
            "Primary cancer diagnosis, e.g., 'invasive ductal carcinoma of the breast'"
        ),
    )

    cancer_stage: Literal[
        "I", "IA", "IB", "II", "IIA", "IIB",
        "III", "IIIA", "IIIB", "IIIC", "IV"
    ] | None = Field(
        None, description="AJCC cancer stage. Null if not mentioned."
    )

    is_metastatic: bool | None = Field(
        None, description="True if metastatic disease. Null if not stated."
    )

    histology: str | None = Field(
        None, description="Histological type, e.g., 'ductal', 'lobular'"
    )

    # Biomarkers (breast cancer focused)
    er_status: Literal["positive", "negative"] | None = Field(
        None, description="Estrogen receptor status. Null if not mentioned."
    )

    pr_status: Literal["positive", "negative"] | None = Field(
        None, description="Progesterone receptor status. Null if not mentioned."
    )

    her2_status: Literal["positive", "negative", "equivocal"] | None = Field(
        None, description="HER2 status. Null if not mentioned."
    )

    brca_status: Literal[
        "BRCA1_mutated", "BRCA2_mutated", "BRCA_wildtype"
    ] | None = Field(
        None, description="BRCA mutation status. Null if not mentioned."
    )

    ki67_percent: float | None = Field(
        None,
        description="Ki-67 proliferation index as percentage. Null if not mentioned.",
    )

    pdl1_status: Literal["positive", "negative"] | None = Field(
        None, description="PD-L1 expression status. Null if not mentioned."
    )

    # Clinical status
    ecog_score: Literal[0, 1, 2, 3, 4] | None = Field(
        None, description="ECOG performance status (0-4). Null if not mentioned."
    )

    # Treatment history
    prior_therapies: list[PriorTherapy] = Field(
        default_factory=list,
        description="List of prior cancer treatments. Empty list if none mentioned.",
    )

    lines_of_therapy: int | None = Field(
        None,
        description="Number of prior lines of therapy for current cancer. Null if not mentioned.",
    )

    prior_radiation: bool | None = Field(
        None,
        description="Whether patient received prior radiation therapy. Null if not mentioned.",
    )

    prior_surgery: bool | None = Field(
        None,
        description="Whether patient had prior cancer surgery. Null if not mentioned.",
    )

    # Lab values
    lab_values: list[LabValue] = Field(
        default_factory=list,
        description="Recent laboratory results mentioned in text. Empty list if none mentioned.",
    )

    # Comorbidities
    comorbidities: list[Comorbidity] = Field(
        default_factory=list,
        description="Comorbid conditions. Empty list if none mentioned.",
    )

    # Organ function
    adequate_liver_function: bool | None = Field(
        None, description="Whether liver function is adequate. Null if not assessed."
    )

    adequate_renal_function: bool | None = Field(
        None, description="Whether renal function is adequate. Null if not assessed."
    )

    adequate_bone_marrow: bool | None = Field(
        None, description="Whether bone marrow function is adequate. Null if not assessed."
    )

    # CNS involvement
    brain_metastases: Literal[
        "none", "stable_treated", "active_untreated"
    ] | None = Field(
        None, description="Brain metastasis status. Null if not mentioned."
    )

    @model_validator(mode='after')
    def check_not_empty(self) -> 'PatientProfile':
        # Check a few core fields that should almost always be present in a valid patient narrative
        core_fields = [self.age, self.sex, self.primary_diagnosis, self.er_status, self.ecog_score]
        
        # If absolutely everything is None, reject it so Instructor retries
        if all(field is None for field in core_fields) and len(self.prior_therapies) == 0:
            raise ValueError(
                "Extracted profile is entirely empty. You missed crucial information. "
                "Please read the text again and populate fields like age, sex, ER/PR status, and ECOG."
            )
        return self
