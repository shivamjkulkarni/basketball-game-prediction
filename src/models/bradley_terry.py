"""
Bradley-Terry Win Probability Model with Fatigue and Home Court Advantage.
"""

import math
from typing import Tuple

from src.config import BASE_B, BETA_1, DIVISOR_C


def bradley_terry_probability(
    ri: float,
    rj: float,
    ortf: float,
    trof: float,
    hfa: float = 0.0,
    b: float = BASE_B,
    c: float = DIVISOR_C,
    beta_1: float = BETA_1,
) -> Tuple[float, float]:
    """
    Calculate modified Bradley-Terry win probability incorporating Elo difference,
    Home Field Advantage (HFA), and fatigue ratios (ORTF / TROF).

    P(i beats j) = 1 / (1 + b^(((rj - (ri + HFA)) / c) + beta_1 * ln(ORTF) - beta_1 * ln(TROF)))

    Args:
        ri (float): Rating of team i.
        rj (float): Rating of team j.
        ortf (float): Opponent Rest Team Fatigue for team i.
        trof (float): Team Rest Opponent Fatigue for team i.
        hfa (float): Home field advantage points applied to team i (default 0.0).
        b (float): Exponent base (default 99.39).
        c (float): Rating-to-odds divisor (default 993.39).
        beta_1 (float): Weight for log fatigue ratio (default 0.02).

    Returns:
        Tuple[float, float]: (P(team_i wins), P(team_j wins)).
    """
    ortf_safe = max(float(ortf), 1e-5)
    trof_safe = max(float(trof), 1e-5)

    # Exponent formulation
    rating_component = (rj - (ri + hfa)) / c
    fatigue_component = beta_1 * (math.log(ortf_safe) - math.log(trof_safe))
    exponent = rating_component + fatigue_component

    # Safeguard against extreme exponentiation overflow
    exponent_bounded = max(min(exponent, 50.0), -50.0)
    prob_i = 1.0 / (1.0 + (b ** exponent_bounded))
    prob_j = 1.0 - prob_i
    return prob_i, prob_j

