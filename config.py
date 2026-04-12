"""
Disease Outbreak Detection - Configuration
===========================================
Extensible disease code mapping and system configuration.

To add a new disease:
    1. Add an entry to DISEASE_CODES with:
       - key: short snake_case name
       - "name": display name
       - "codes": list of SNOMED CT diagnosis codes (strings)
       - "category": epidemiological category
       - "alert_threshold": rule-based alert config (to be filled later)

To add a new code to an existing disease:
    1. Append the code string to the "codes" list

Example:
    DISEASE_CODES["typhoid"] = {
        "name": "Typhoid Fever",
        "codes": ["4834000"],
        "category": "waterborne",
    }
"""

# =============================================================================
# DISEASE CODE REGISTRY (SNOMED CT)
# =============================================================================
DISEASE_CODES = {
    "malaria": {
        "name": "Malaria",
        "codes": ["735531008"],
        "category": "vector_borne",
    },
    "dengue": {
        "name": "Dengue",
        "codes": ["38362002"],
        "category": "vector_borne",
    },
    "chikungunya": {
        "name": "Chikungunya",
        "codes": ["111864006"],
        "category": "vector_borne",
    },
    "cholera": {
        "name": "Cholera",
        "codes": ["81020007"],
        "category": "waterborne",
    },
    "gastroenteritis": {
        "name": "Acute Gastroenteritis",
        "codes": ["25374005"],
        "category": "waterborne",
    },
    "mud_fever": {
        "name": "Mud Fever (Leptospirosis)",
        "codes": ["77377001"],
        "category": "zoonotic",
    },
    "ebola": {
        "name": "Ebola",
        "codes": ["37109004"],
        "category": "hemorrhagic",
    },
}

# =============================================================================
# COLUMN MAPPING — Maps raw DataFrame columns to standardized names
# Adjust if your column names differ
# =============================================================================
COLUMN_MAP = {
    # Patient & Clinical
    "op_id": "op_id",
    "diagnosis": "diagnosis_code",
    "diagnosis_name": "diagnosis_name",
    "diagnosis_event_ts": "event_timestamp",
    "visit": "visit",
    # Symptoms
    "complaint": "complaint_code",
    "complaint_name": "complaint_name",
    "complaintsduration": "complaint_duration",
    "duration_days": "duration_days",
    "onset": "onset",
    "severity": "severity",
    # Vitals
    "temperature": "temperature",
    "pulse": "pulse",
    "respiratory_rate": "respiratory_rate",
    "systole": "systole",
    "diastole": "diastole",
    "spo2": "spo2",
    "rbs": "rbs",
    "weight": "weight",
    "bmi": "bmi",
    "bmi_text": "bmi_text",
    # Examination
    "nourishment": "nourishment",
    "pallor": "pallor",
    "cyanosis": "cyanosis",
    "pedal_edema": "pedal_edema",
    "icterus": "icterus",
    "lymphadenopathy": "lymphadenopathy",
    "local_examination": "local_examination",
    "diseases_status": "diseases_status",
    # Lifestyle
    "smoking_yesno": "smoking",
    "drinking": "drinking",
    "allergy": "allergy",
    "allergy_name": "allergy_name",
    "pack_years": "pack_years",
    # Geography
    "city": "city",
    "postal_code": "postal_code",
    "mandal_name": "mandal",
    "district": "district",
    "geolocation": "geolocation",
    "pincode": "pincode",
    "master_facility_id": "facility_id",
    "master_facility_name": "facility_name",
    "district_code": "district_code",
    "sub_district": "sub_district",
    # Meta
    "load_date": "load_date",
}

# =============================================================================
# NUMERIC FIELDS — These are stored as strings and need parsing
# =============================================================================
NUMERIC_FIELDS = [
    "temperature", "pulse", "respiratory_rate",
    "systole", "diastole", "spo2", "rbs",
    "weight", "bmi", "duration_days",
]

# =============================================================================
# VITAL SIGN THRESHOLDS — For feature engineering
# =============================================================================
VITAL_THRESHOLDS = {
    "fever_temp_c": 38.0,        # >= 38°C considered fever
    "low_spo2": 95.0,            # < 95% considered low
    "high_pulse": 100.0,         # >= 100 bpm considered tachycardia
    "high_respiratory": 20.0,    # >= 20 breaths/min
    "high_systole": 140.0,       # >= 140 mmHg
    "low_systole": 90.0,         # < 90 mmHg
    "high_bmi": 30.0,            # >= 30 considered obese
}

# =============================================================================
# SEVERITY MAPPING
# =============================================================================
SEVERITY_MAP = {
    # Numeric string values (as found in AP data)
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    # Text values (fallback)
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

# =============================================================================
# EXAMINATION FLAG FIELDS — Binary yes/no fields
# =============================================================================
EXAMINATION_FLAGS = [
    "pallor", "cyanosis", "pedal_edema", "icterus", "lymphadenopathy",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_codes():
    """Returns a flat set of all tracked diagnosis codes."""
    codes = set()
    for disease_info in DISEASE_CODES.values():
        codes.update(disease_info["codes"])
    return codes


def code_to_disease(code: str) -> str:
    """Maps a diagnosis code to its disease key. Returns None if not tracked."""
    for disease_key, disease_info in DISEASE_CODES.items():
        if code in disease_info["codes"]:
            return disease_key
    return None


def code_to_disease_name(code: str) -> str:
    """Maps a diagnosis code to its display name. Returns None if not tracked."""
    for disease_info in DISEASE_CODES.values():
        if code in disease_info["codes"]:
            return disease_info["name"]
    return None


def get_disease_names():
    """Returns dict of {disease_key: display_name}."""
    return {k: v["name"] for k, v in DISEASE_CODES.items()}


def get_codes_for_disease(disease_key: str) -> list:
    """Returns list of codes for a disease key."""
    if disease_key in DISEASE_CODES:
        return DISEASE_CODES[disease_key]["codes"]
    return []


def add_disease(key: str, name: str, codes: list, category: str = "other"):
    """Programmatically add a new disease at runtime."""
    DISEASE_CODES[key] = {
        "name": name,
        "codes": codes,
        "category": category,
    }


def add_code_to_disease(disease_key: str, code: str):
    """Add a new code to an existing disease."""
    if disease_key in DISEASE_CODES:
        if code not in DISEASE_CODES[disease_key]["codes"]:
            DISEASE_CODES[disease_key]["codes"].append(code)
    else:
        raise KeyError(f"Disease '{disease_key}' not found. Use add_disease() first.")
