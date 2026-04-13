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
        "alert_rules": [
            {
                "level": "P1",
                "name": "New focus",
                "description": "≥1 case in a mandal with no cases for 3 months (IDSP)",
                "type": "gap_reappearance",
                "gap_weeks": 12,
                "min_cases": 1,
            },
            {
                "level": "P2",
                "name": "Surge",
                "description": "Weekly count doubles vs previous 12-week average (IDSP SPR rule)",
                "type": "surge",
                "multiplier": 2.0,
                "baseline_weeks": 12,
            },
        ],
    },
    "dengue": {
        "name": "Dengue",
        "codes": ["38362002"],
        "category": "vector_borne",
        "alert_rules": [
            {
                "level": "P1",
                "name": "Single DHF case",
                "description": "≥1 suspected dengue case in a mandal (IDSP)",
                "type": "threshold_mandal",
                "min_cases": 1,
                "window_weeks": 1,
            },
            {
                "level": "P2",
                "name": "Rising trend",
                "description": "Rising cases for 3 consecutive weeks (IDSP)",
                "type": "consecutive_rise",
                "consecutive_weeks": 3,
            },
        ],
    },
    "chikungunya": {
        "name": "Chikungunya",
        "codes": ["111864006"],
        "category": "vector_borne",
        "alert_rules": [
            {
                "level": "P1",
                "name": "New focus",
                "description": "≥1 case in a new mandal (same vector as Dengue)",
                "type": "gap_reappearance",
                "gap_weeks": 12,
                "min_cases": 1,
            },
            {
                "level": "P2",
                "name": "Cluster",
                "description": "≥5 cases in a mandal in 1 week",
                "type": "threshold_mandal",
                "min_cases": 5,
                "window_weeks": 1,
            },
        ],
    },
    "cholera": {
        "name": "Cholera",
        "codes": ["7718000"],
        "category": "waterborne",
        "alert_rules": [
            {
                "level": "P1",
                "name": "Single case",
                "description": "≥1 case of cholera (IDSP: single case = outbreak)",
                "type": "threshold_mandal",
                "min_cases": 1,
                "window_weeks": 1,
            },
            {
                "level": "P2",
                "name": "Cluster",
                "description": "≥10 cases in a mandal in 1 week (IDSP: 10 households)",
                "type": "threshold_mandal",
                "min_cases": 10,
                "window_weeks": 1,
            },
        ],
    },
    "gastroenteritis": {
        "name": "Acute Gastroenteritis",
        "codes": ["25374005"],
        "category": "waterborne",
        "alert_rules": [
            {
                "level": "P2",
                "name": "Cluster",
                "description": "≥10 cases in a mandal in 1 week (diarrhoea cluster rule)",
                "type": "threshold_mandal",
                "min_cases": 10,
                "window_weeks": 1,
            },
            {
                "level": "P3",
                "name": "Surge",
                "description": "Weekly count exceeds 2× the 4-week rolling average",
                "type": "surge",
                "multiplier": 2.0,
                "baseline_weeks": 4,
            },
        ],
    },
    "mud_fever": {
        "name": "Mud Fever (Leptospirosis)",
        "codes": ["77377001"],
        "category": "zoonotic",
        "alert_rules": [
            {
                "level": "P2",
                "name": "Cluster",
                "description": "≥5 cases in a mandal in 1 week",
                "type": "threshold_mandal",
                "min_cases": 5,
                "window_weeks": 1,
            },
            {
                "level": "P3",
                "name": "Surge",
                "description": "Weekly count exceeds 2× the 4-week rolling average",
                "type": "surge",
                "multiplier": 2.0,
                "baseline_weeks": 4,
            },
        ],
    },
    "ebola": {
        "name": "Ebola",
        "codes": ["37109004"],
        "category": "hemorrhagic",
        "alert_rules": [
            {
                "level": "P0",
                "name": "Any case",
                "description": "ANY single case = immediate P0 alert (IDSP unusual syndrome)",
                "type": "threshold_mandal",
                "min_cases": 1,
                "window_weeks": 1,
            },
        ],
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
