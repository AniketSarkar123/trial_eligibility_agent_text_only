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
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator, AliasChoices


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

    # --- MAKE THIS FIELD OPTIONAL WITH A DEFAULT ---
    is_current: bool = Field(
        default=False, 
        description="True if the patient is currently taking this drug. Default is false."
    )

    days_since_last_dose: Optional[int] = Field(
        None, description="Convert '2 years ago' to 730, '4 weeks ago' to 28. Null if not specified."
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

    value: float = Field( description="Numeric result or null")

    unit: str = Field(
        description="Unit of measurement, e.g., 'g/dL', 'mg/dL', 'cells/uL', if provided, or null."
    )


class Comorbidity(BaseModel):
    """A comorbid condition."""

    condition: str = Field(
        description="Condition name, e.g., 'type 2 diabetes', 'hypertension'"
    )

    active: bool = Field(
        description="True if currently active/being treated",
        validation_alias=AliasChoices('active', 'is_current')
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

    # # --- ADD THIS PRESSURE VALVE ---
    # analysis: str | None = Field(
    #     None, 
    #     description="Optional space for you to think or analyze the clinical text step-by-step before extracting the structured data."
    # )

    # Demographics
    age: int | None = Field(description="Patient age in years")

    sex: Literal["male", "female"] | None = Field(description="Biological sex")

    menopausal_status: Literal[
        "premenopausal", "perimenopausal", "postmenopausal"
    ] | None = Field(
        description="Menopausal status. Null if not mentioned or not applicable.",
    )

    pregnancy_status: Literal["pregnant", "lactating", "not_pregnant"] | None = Field(description="Must be 'pregnant', 'lactating', or 'not_pregnant'. If pregnancy or lactation is NOT explicitly mentioned in the text, you MUST set this to null. Do not infer based on age, menopause, or consent to therapy.")


    # Diagnosis
    primary_diagnosis: str | None = Field(
        description=(
            "Primary cancer diagnosis, e.g., 'invasive ductal carcinoma of the breast'"
        ),
    )

    cancer_stage: Literal[
        "I", "IA", "IB", "II", "IIA", "IIB",
        "III", "IIIA", "IIIB", "IIIC", "IV"
    ] | None = Field(description="AJCC cancer stage. Null if not mentioned."
    )

    is_metastatic: bool | None = Field(description="True if metastatic disease. Null if not stated."
    )

    histology: str | None = Field(description="Histological type, e.g., 'ductal', 'lobular'"
    )

    tumor_size_cm: float | None = Field(description="Size of the primary tumor in centimeters.")
    nodal_status: Literal["N0", "N1", "N2", "N3", "positive", "negative"] | None = Field(None, description="Lymph node involvement status.")
    tumor_grade: Literal[1, 2, 3] | None = Field(description="Nottingham histological grade of the tumor.")
    lymphovascular_invasion: bool | None = Field(description="Presence of lymphovascular invasion (LVI). True if present, False if absent.")
    disease_focality: Literal["unifocal", "multifocal", "multicentric", "bilateral"] | None = Field(description="Focality of the breast cancer.")

    # Biomarkers (breast cancer focused)
    er_status: Literal["positive", "negative"] | None = Field(description="Estrogen receptor status. Null if not mentioned."
    )

    pr_status: Literal["positive", "negative"] | None = Field(description="Progesterone receptor status. Null if not mentioned."
    )

    her2_status: Literal["positive", "negative", "equivocal"] | None = Field(description="HER2 status. Null if not mentioned."
    )

    brca_status: Literal[
        "BRCA1_mutated", "BRCA2_mutated", "BRCA_wildtype"
    ] | None = Field(description="BRCA mutation status. Null if not mentioned."
    )

    ki67_percent: float | None = Field(
        description="Ki-67 proliferation index as percentage. Null if not mentioned.",
    )

    pdl1_status: Literal["positive", "negative"] | None = Field(description="PD-L1 expression status. Null if not mentioned."
    )

    stil_score_percent: float | None = Field( 
        description="Stromal Tumor-Infiltrating Lymphocytes (sTILs or TILs) score as a percentage. Look for terms like 'stromal TILs' or 'sTILs'. Extract only the number."
    )

    received_neoadjuvant_therapy: bool | None = Field( 
        description="True ONLY if the patient has ALREADY received neoadjuvant systemic therapy. MUST be False if the therapy is only planned for the future or if the patient has never received systemic treatment."
    )
    
    received_adjuvant_therapy: bool | None = Field( 
        description="True if patient received systemic therapy after surgery."
    )

    has_recurrence: bool | None = Field( 
        description="True if the patient has experienced a recurrence of their cancer. False if the text explicitly states no recurrence."
    )

    # Clinical status
    ecog_score: Literal[0, 1, 2, 3, 4] | None = Field(
        description="ECOG performance status (0-4). Null if not mentioned."
    )

    # Treatment history
    prior_therapies: list[PriorTherapy] = Field(
        default_factory=list,
        description="List of prior cancer treatments. Empty list if none mentioned.",
    )

    lines_of_therapy: int | None = Field(
        description="Number of prior lines of therapy for current cancer. Null if not mentioned.",
    )

    prior_radiation: bool | None = Field(
        description="Whether patient received prior radiation therapy. Null if not mentioned.",
    )

    prior_surgery: bool | None = Field(
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
    adequate_liver_function: bool | None = Field(description="Whether liver function is adequate. Null if not assessed."
    )

    adequate_renal_function: bool | None = Field(description="Whether renal function is adequate. Null if not assessed."
    )

    adequate_bone_marrow: bool | None = Field(description="Whether bone marrow function is adequate. Null if not assessed."
    )

    # CNS involvement
    brain_metastases: Optional[Literal['none', 'stable_treated', 'active_untreated']] = Field( 
        description=(
            "Status of brain metastases ONLY. "
            "MUST be EXACTLY one of: 'none', 'stable_treated', or 'active_untreated'. "
            "Do NOT put 'oligometastatic' or other general cancer terms here."
        )
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
    
    @model_validator(mode='after')
    def clinical_cross_imputation(self) -> 'PatientProfile':
        """Auto-fills guaranteed clinical facts to cover LLM extraction gaps."""
        
        # 1. Stage IV inherently means metastatic
        if self.cancer_stage == "IV" and self.is_metastatic is None:
            self.is_metastatic = True
            
        # 2. Stage I-III inherently means non-metastatic (usually)
        if self.cancer_stage in ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC"]:
            if self.is_metastatic is None:
                self.is_metastatic = False

        # 3. Triple Negative Breast Cancer (TNBC) auto-fills biomarkers
        if self.primary_diagnosis and "triple-negative" in self.primary_diagnosis.lower():
            if self.er_status is None: self.er_status = "negative"
            if self.pr_status is None: self.pr_status = "negative"
            if self.her2_status is None: self.her2_status = "negative"
            
        # 4. Biological Age Inference (Menopause & Pregnancy)
        if self.age is not None and self.age >= 60 and self.sex == "female":
            if self.menopausal_status is None:
                self.menopausal_status = "postmenopausal"
            if self.pregnancy_status is None:
                self.pregnancy_status = "not_pregnant"
                
        return self
