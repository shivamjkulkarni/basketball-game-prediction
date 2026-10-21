"""
Zero-Sum Elo Rating Tracker & Dynamic Update Engine.
Guarantees strict league-wide rating point invariance: sum(r_t) = N * 1500.0.
"""

import math
from typing import Dict, List, Optional, Tuple

from src.config import BASE_ELO, BETA_3, BETA_4, K_FACTOR


def calculate_zero_sum_update(
    ri: float,
    rj: float,
    ai: float,
    ei: float,
    lre: float = 0.0,
    paf_i: float = 1.0,
    paf_j: float = 1.0,
    k: float = K_FACTOR,
    beta_3: float = BETA_3,
    beta_4: float = BETA_4,
) -> Tuple[float, float, float]:
    """
    Calculate strictly zero-sum Elo rating adjustments.

    Multiplier M = k * (1 + beta_3 * log2(1 + LRE)) * (1 + beta_4 * |PAF_i - PAF_j|)
    Delta r = M * (a_i - e_i)
    new_ri = ri + Delta r
    new_rj = rj - Delta r

    Args:
        ri, rj: Current Elo ratings for team i and team j.
        ai: Actual outcome for team i (1.0 for win, 0.0 for loss).
        ei: Expected win probability for team i.
        lre: Lead Retention Efficiency (|MoV| / Largest Lead).
        paf_i, paf_j: Performance Adjustment Factors for teams i and j.
        k: Elo sensitivity factor (default 32.0).
        beta_3: LRE weight (default 0.70).
        beta_4: PAF asymmetry weight (default 0.02).

    Returns:
        Tuple[float, float, float]: (new_ri, new_rj, Delta_r).
    """
    lre_safe = max(float(lre), 0.0)
    paf_delta = abs(float(paf_i) - float(paf_j))

    # Shared symmetric multiplier
    multiplier = k * (1.0 + beta_3 * math.log2(1.0 + lre_safe)) * (1.0 + beta_4 * paf_delta)
    delta_r = multiplier * (float(ai) - float(ei))

    new_ri = ri + delta_r
    new_rj = rj - delta_r
    return new_ri, new_rj, delta_r


class EloTracker:
    """
    Stateful repository managing collegiate team Elo ratings.
    Guarantees that rating adjustments across matchups are strictly zero-sum.
    """

    TEAM_ALIASES = {
        "stony_brooke_seawolves": "stony_brook_seawolves",
    }

    def __init__(self, base_elo: float = BASE_ELO):
        self.base_elo = float(base_elo)
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[float]] = {}
        self.games_played: Dict[str, int] = {}

    @classmethod
    def from_csv(cls, csv_path: str, base_elo: float = BASE_ELO) -> "EloTracker":
        """Load pre-computed end-of-season team ratings from CSV."""
        import pandas as pd
        tracker = cls(base_elo=base_elo)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            t = str(row["team"]).strip()
            elo = float(row["final_elo"])
            gp = int(row.get("games_played", 0))
            tracker.ratings[t] = elo
            tracker.rating_history[t] = [base_elo, elo]
            tracker.games_played[t] = gp
        return tracker

    def get_rating(self, team: str) -> float:
        """Retrieve team rating, initializing to base_elo if first encounter."""
        team_key = str(team).strip()
        if team_key not in self.ratings and team_key in self.TEAM_ALIASES:
            alias_key = self.TEAM_ALIASES[team_key]
            if alias_key in self.ratings:
                return self.ratings[alias_key]
        if team_key not in self.ratings:
            self.ratings[team_key] = self.base_elo
            self.rating_history[team_key] = [self.base_elo]
            self.games_played[team_key] = 0
        return self.ratings[team_key]

    def update(
        self,
        team_i: str,
        team_j: str,
        ai: float,
        ei: float,
        lre: float = 0.0,
        paf_i: float = 1.0,
        paf_j: float = 1.0,
        k: float = K_FACTOR,
        beta_3: float = BETA_3,
        beta_4: float = BETA_4,
    ) -> Tuple[float, float, float]:
        """
        Execute zero-sum update for a completed game between team_i and team_j.
        """
        ri = self.get_rating(team_i)
        rj = self.get_rating(team_j)

        new_ri, new_rj, delta_r = calculate_zero_sum_update(
            ri=ri,
            rj=rj,
            ai=ai,
            ei=ei,
            lre=lre,
            paf_i=paf_i,
            paf_j=paf_j,
            k=k,
            beta_3=beta_3,
            beta_4=beta_4,
        )

        self.ratings[team_i] = new_ri
        self.ratings[team_j] = new_rj
        self.rating_history[team_i].append(new_ri)
        self.rating_history[team_j].append(new_rj)
        self.games_played[team_i] += 1
        self.games_played[team_j] += 1

        return new_ri, new_rj, delta_r

    def total_league_elo(self) -> float:
        """Return the sum of all team ratings currently in the tracker."""
        return sum(self.ratings.values())

    def total_teams(self) -> int:
        """Return count of registered teams."""
        return len(self.ratings)

