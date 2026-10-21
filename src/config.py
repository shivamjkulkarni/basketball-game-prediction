"""
Centralized Configuration & Model Hyperparameters for Basketball Prediction Engine.
"""

from typing import Dict, List

# Earth physical constant for spatial Haversine calculations
EARTH_RADIUS_KM: float = 6371.0

# Base Rating Parameters
BASE_ELO: float = 1500.0
K_FACTOR: float = 32.0
HFA_ELO: float = 72.52  # Home Field Advantage points

# Bradley-Terry Win Probability Parameters
BASE_B: float = 99.39      # Exponent base
DIVISOR_C: float = 993.39  # Rating difference divisor
BETA_1: float = 0.02       # Fatigue logarithmic scaling weight

# Fatigue & Spatial Logistics Parameters
BETA_2: float = 1e-6       # Travel distance fatigue penalty weight (0.000001)
TZ_DIVISOR: float = 3.0    # Time zone exponential decay divisor
DEFAULT_REST_DAYS: float = 2.0

# Performance Scaling Parameters
BETA_3: float = 0.70       # Lead Retention Efficiency (LRE) scaling factor
BETA_4: float = 0.02       # Performance Adjustment Factor (PAF) asymmetry weight
K_SCALE_LOGISTIC: float = 1.0  # Logistic slope factor for bounded AGPF

# Box-Score Metric Weights (alpha_1 through alpha_8)
# NetEff (0.25), eFG% (0.20), TS% (0.15), OREB% (0.10), DREB% (0.10), AST:TOV (0.10), TOV% (0.05), STL% (0.05)
ALPHA_WEIGHTS: List[float] = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05]

METRIC_KEYS: List[str] = [
    "NetEff",
    "eFG",
    "TS",
    "OREB",
    "DREB",
    "AST:TOV",
    "TOV",
    "STL",
]

# Baseline Empirical Statistics across the 2021-2022 NCAA Season (5,219 games)
DEFAULT_METRIC_STATS: Dict[str, Dict[str, float]] = {
    "NetEff": {"mean": -0.0066, "std": 28.4223},
    "eFG": {"mean": 0.4509, "std": 0.0900},
    "TS": {"mean": 0.4883, "std": 0.0860},
    "OREB": {"mean": 11.5103, "std": 4.5119},
    "DREB": {"mean": 25.3205, "std": 5.5925},
    "AST:TOV": {"mean": 0.9390, "std": 0.6047},
    "TOV": {"mean": 15.9280, "std": 5.1630},
    "STL": {"mean": 7.8072, "std": 3.5714},
}

# Regional Divisions
REGIONS: List[str] = ["North", "South", "East", "West"]

