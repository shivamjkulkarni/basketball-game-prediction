"""
Box-Score Normalization, Logistic AGPF, Expanding A'GPF, and PAF Engine.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    ALPHA_WEIGHTS,
    DEFAULT_METRIC_STATS,
    K_SCALE_LOGISTIC,
    METRIC_KEYS,
)


def safe_float(val: any, default: float = 0.0) -> float:
    """Safely convert a value to float with robust fallbacks."""
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


class BoxScoreStandardizer:
    """
    Standardizes basketball box-score metrics across the competition dataset.
    Eliminates blowout singularities by mapping standardized performances
    through a logistic sigmoid transformation:
        AGPF = 1 / (1 + exp(-k_scale * sum(alpha_i * Z_i))) in (0, 1)
    """

    def __init__(
        self,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        alpha: Optional[List[float]] = None,
        k_scale: float = K_SCALE_LOGISTIC,
    ):
        self.stats = stats or DEFAULT_METRIC_STATS
        self.alpha = np.array(alpha or ALPHA_WEIGHTS, dtype=float)
        self.k_scale = float(k_scale)
        # In-memory history for expanding A'GPF: team -> list of prior AGPFs
        self.agpf_history: Dict[str, List[float]] = {}

    @classmethod
    def fit_from_dataframe(
        cls,
        df: pd.DataFrame,
        alpha: Optional[List[float]] = None,
        k_scale: float = K_SCALE_LOGISTIC,
    ) -> "BoxScoreStandardizer":
        """Fit empirical mean and std parameters from a DataFrame."""
        fitted_stats = {}
        for m in METRIC_KEYS:
            col_matches = [c for c in df.columns if c.lower() == m.lower()]
            if col_matches:
                s = pd.to_numeric(df[col_matches[0]], errors='coerce').dropna()
                std_v = s.std()
                fitted_stats[m] = {
                    "mean": float(s.mean()),
                    "std": float(std_v) if std_v and std_v > 0 else 1.0,
                }
            else:
                fitted_stats[m] = DEFAULT_METRIC_STATS[m]
        return cls(stats=fitted_stats, alpha=alpha, k_scale=k_scale)

    def standardize(self, raw_metrics: Dict[str, float]) -> np.ndarray:
        """Convert raw box-score metrics into z-scores."""
        z_scores = []
        for m in METRIC_KEYS:
            val = safe_float(raw_metrics.get(m, raw_metrics.get(m.lower(), 0.0)))
            stat = self.stats.get(m, {"mean": 0.0, "std": 1.0})
            mean = stat["mean"]
            std = stat["std"] if stat["std"] > 0 else 1.0
            z = (val - mean) / std
            z_scores.append(z)
        return np.array(z_scores, dtype=float)

    def compute_agpf(self, raw_metrics: Dict[str, float]) -> float:
        """
        Compute bounded Actual Game Performance Factor (AGPF) via logistic transform:
            AGPF = 1 / (1 + exp(-k_scale * dot(alpha, Z)))
        Guarantees AGPF in (0, 1) strictly, avoiding singularities and inversions.
        """
        z_scores = self.standardize(raw_metrics)
        weighted_score = float(np.dot(self.alpha, z_scores))
        clipped_score = max(min(self.k_scale * weighted_score, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-clipped_score))

    def get_expanding_a_gpf(self, team: str, current_agpf: float) -> float:
        """
        Compute cumulative historical expanding average of prior games (A'GPF):
        A'GPF_t = (1 / G) * sum_{g < t} AGPF_g
        """
        history = self.agpf_history.get(team, [])
        if not history:
            return current_agpf
        return float(np.mean(history))

    def record_agpf(self, team: str, agpf: float) -> None:
        """Record game AGPF in team expanding history."""
        self.agpf_history.setdefault(team, []).append(agpf)

    def compute_paf(self, team: str, raw_metrics: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Compute AGPF, expanding A'GPF, and PAF for a team's game performance:
            PAF = AGPF / A'GPF

        Returns:
            Tuple[float, float, float]: (AGPF, A'GPF, PAF)
        """
        agpf = self.compute_agpf(raw_metrics)
        a_gpf = self.get_expanding_a_gpf(team, agpf)
        paf = agpf / a_gpf if a_gpf > 0 else 1.0
        self.record_agpf(team, agpf)
        return agpf, a_gpf, paf
