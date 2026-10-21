"""
Fatigue Indices Calculation Module: ORTF and TROF Modeling.
"""

import math
from typing import Tuple

from src.config import BETA_2, TZ_DIVISOR


def calculate_fatigue_indices(
    rd_i: float,
    rd_j: float,
    td_i: float,
    td_j: float,
    tzd_i: float,
    tzd_j: float,
    beta_2: float = BETA_2,
    tz_divisor: float = TZ_DIVISOR,
) -> Tuple[float, float]:
    """
    Calculate Opponent Rest Team Fatigue (ORTF) and Team Rest Opponent Fatigue (TROF).

    ORTF = RD_j + (exp(|TzD_i| / tz_divisor) + beta_2 * TD_i)
    TROF = RD_i + (exp(|TzD_j| / tz_divisor) + beta_2 * TD_j)

    Args:
        rd_i, rd_j: Rest days for team i and team j.
        td_i, td_j: Travel distance for team i and team j in kilometers.
        tzd_i, tzd_j: Time zone shift for team i and team j in hours.
        beta_2: Distance penalty scaling factor (default 1e-6).
        tz_divisor: Exponential decay factor for timezone shift (default 3.0).

    Returns:
        Tuple[float, float]: (ORTF, TROF) bounded safely above 1e-5.
    """
    ortf = rd_j + (math.exp(abs(tzd_i) / tz_divisor) + beta_2 * td_i)
    trof = rd_i + (math.exp(abs(tzd_j) / tz_divisor) + beta_2 * td_j)
    return max(ortf, 1e-5), max(trof, 1e-5)

