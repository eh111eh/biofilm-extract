"""Extraction schema. One Record per (strain, temperature); five rheology metrics each.

No default values on purpose: Gemini's response-schema converter has historically
rejected `default`, and 'null = not found' should be an explicit decision by the model.
"""
from typing import Optional

from pydantic import BaseModel

STRAINS = ["3610", "2103", "2106", "2107", "2108", "2109", "2125"]
TEMPS = [30, 50]
METRICS = ["G0_prime_Pa", "tan_delta0", "gamma_y_pct", "gamma_f_pct", "WSO_Pa"]

# Physically plausible ranges used by the validator (lo exclusive for strictly positive metrics)
RANGES = {
    "G0_prime_Pa": (0.0, 1e5),
    "tan_delta0": (0.0, 5.0),
    "gamma_y_pct": (0.0, 200.0),
    "gamma_f_pct": (0.0, 250.0),
    "WSO_Pa": (0.0, 1e5),
}


class Metric(BaseModel):
    mean: Optional[float]       # null = value not found in the report
    sd: Optional[float]         # null = no SD reported
    approximate: bool           # report says "~", "≈" or "approximate"
    not_detected: bool          # report explicitly says the quantity was not detected / not reported
    evidence: Optional[str]     # verbatim quote (>= 8 chars) from the report that contains the number


class Record(BaseModel):
    strain: str                 # "3610", "2103", ... (without NRS / NCIB prefix)
    temperature_C: int          # 30 or 50
    G0_prime_Pa: Metric         # plateau storage modulus G'_0 [Pa]
    tan_delta0: Metric          # baseline loss tangent tan(delta)_0 [-]
    gamma_y_pct: Metric         # yield strain [%]
    gamma_f_pct: Metric         # crossover (flow) strain [%]
    WSO_Pa: Metric              # weak strain overshoot [Pa]


class Extraction(BaseModel):
    records: list[Record]


# Format illustration ONLY: strain "XXXX" and all numbers are fake placeholders.
EXAMPLE = """{"records": [{"strain": "XXXX", "temperature_C": 30,
 "G0_prime_Pa": {"mean": 123.4, "sd": 5.6, "approximate": false, "not_detected": false,
                 "evidence": "<verbatim quote containing 123.4>"},
 "tan_delta0": {"mean": null, "sd": null, "approximate": false, "not_detected": false, "evidence": null},
 "gamma_y_pct": {"mean": 7.0, "sd": null, "approximate": true, "not_detected": false,
                 "evidence": "<verbatim quote containing ~7>"},
 "gamma_f_pct": {"mean": null, "sd": null, "approximate": false, "not_detected": true,
                 "evidence": "<verbatim quote saying no crossover was detected>"},
 "WSO_Pa": {"mean": null, "sd": null, "approximate": false, "not_detected": false, "evidence": null}}]}"""
