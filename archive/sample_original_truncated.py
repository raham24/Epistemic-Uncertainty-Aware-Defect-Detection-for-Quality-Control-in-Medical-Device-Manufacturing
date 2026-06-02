from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rdflib import Graph, Namespace, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD, PROV
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

SEED = 42
N_TOTAL = 200_000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

BATCH_SIZE = 2_000
N_BATCHES = N_TOTAL // BATCH_SIZE

# Six terminal process parameters from the paper
PARAMS = [
    "ambient_relative_humidity",
    "ambient_temperature",
    "paste_viscosity",
    "stencil_thickness",
    "time_above_liquidus",
    "peak_reflow_temperature",
]

# Basic engineering specs (example values; adjust to your needs)
SPECS = {
    "ambient_relative_humidity": {"nominal": 50.0, "lsl": 30.0, "usl": 70.0},
    "ambient_temperature": {"nominal": 25.0, "lsl": 20.0, "usl": 30.0},
    "paste_viscosity": {"nominal": 100.0, "lsl": 90.0, "usl": 110.0},
    "stencil_thickness": {"nominal": 100.0, "lsl": 95.0, "usl": 105.0},
    "time_above_liquidus": {"nominal": 60.0, "lsl": 45.0, "usl": 75.0},
    "peak_reflow_temperature": {"nominal": 255.0, "lsl": 250.0, "usl": 260.0},
}

DEFECT_CLASSES = ["no_defect", "open_circuit", "solder_bridging"]

PRINTING_MECHANISMS = ["no_mechanism", "aperture_overfill", "poor_paste_transfer"]
REFLOW_MECHANISMS = ["no_mechanism", "reflow_spreading", "non_coalescence"]

# Causal map used only to generate labels / synthetic targets
CAUSAL_MAP = {
    "open_circuit": {
        "printing": "poor_paste_transfer",
        "reflow": "non_coalescence",
    },
    "solder_bridging": {
        "printing": "aperture_overfill",
        "reflow": "reflow_spreading",
    },
}

PARAM_TO_STAGE = {
    "ambient_relative_humidity": "printing",
    "ambient_temperature": "printing",
    "paste_viscosity": "printing",
    "stencil_thickness": "printing",
    "time_above_liquidus": "reflow",
    "peak_reflow_temperature": "reflow",
}

PARAM_TO_DIRECTION = {
    "ambient_relative_humidity": {"high": "open_circuit", "low": "solder_bridging"},
    "ambient_temperature": {"high": "solder_bridging", "low": "open_circuit"},
    "paste_viscosity": {"high": "open_circuit", "low": "solder_bridging"},
    "stencil_thickness": {"high": "solder_bridging", "low": "open_circuit"},
    "time_above_liquidus": {"high": "solder_bridging", "low": "open_circuit"},
    "peak_reflow_temperature": {"high": "solder_bridging", "low": "open_circuit"},
}


# ============================================================
# Utilities
# ============================================================

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def graded_risk_from_deviation(
    deviation: float,
    low_risk: float = 0.05,
    mid_risk: float = 0.70,
