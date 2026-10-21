"""
Tournament Bracket Simulation & Win Probability Evaluation Module.
"""

from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import BASE_B, BETA_1, BETA_2, DIVISOR_C
from src.features.fatigue import calculate_fatigue_indices
from src.models.bradley_terry import bradley_terry_probability
from src.models.elo import EloTracker


# Historical paper predictions published in competition paper (pages 5-6)
PAPER_BASELINE_HOME = [0.45, 0.76, 0.70, 0.62, 0.50, 0.53, 0.73, 0.54, 0.80, 0.54]
PAPER_BASELINE_AWAY = [0.55, 0.24, 0.30, 0.38, 0.50, 0.47, 0.27, 0.46, 0.20, 0.46]


def evaluate_bracket_predictions(
    predict_df: pd.DataFrame,
    tracker: Optional[EloTracker] = None,
    b: float = BASE_B,
    c: float = DIVISOR_C,
    beta_1: float = BETA_1,
    beta_2: float = BETA_2,
    hfa: float = 0.0,
) -> pd.DataFrame:
    """
    Generate tournament game win probabilities for matchups in predict_df.
    Computes:
      - Remediated model probability using the Elo values in the bracket file.
      - Simulated model probability using the stateful EloTracker (if provided).
      - Side-by-side comparison with the paper baseline.

    Args:
        predict_df (pd.DataFrame): DataFrame containing matchup details.
        tracker (Optional[EloTracker]): Optional trained EloTracker from full season simulation.

    Returns:
        pd.DataFrame: Comparison DataFrame with all probabilities.
    """
    remediated_home = []
    remediated_away = []
    simulated_elo_h = []
    simulated_elo_a = []
    simulated_home = []
    simulated_away = []

    for _, row in predict_df.iterrows():
        ri_given = float(row['elo_home'])
        rj_given = float(row['elo_away'])
        rd_h = float(row['rd_home'])
        rd_a = float(row['rd_away'])
        td_h = float(row['td_home'])
        td_a = float(row['td_away'])
        tzd_h = float(row['tzd_home'])
        tzd_a = float(row['tzd_away'])

        # Compute fatigue indices
        ortf, trof = calculate_fatigue_indices(
            rd_i=rd_h, rd_j=rd_a, td_i=td_h, td_j=td_a, tzd_i=tzd_h, tzd_j=tzd_a, beta_2=beta_2
        )

        # 1. Given Elo probabilities (using paper's baseline parameters for direct comparison)
        ph, pa = bradley_terry_probability(
            ri=ri_given, rj=rj_given, ortf=ortf, trof=trof, hfa=hfa, b=BASE_B, c=DIVISOR_C, beta_1=BETA_1
        )
        remediated_home.append(round(ph, 4))
        remediated_away.append(round(pa, 4))

        # 2. Simulated end-of-season Elo probabilities (using active/calibrated parameters)
        if tracker is not None:
            t_h = str(row['team_home']).strip()
            t_a = str(row['team_away']).strip()
            sim_ri = tracker.get_rating(t_h)
            sim_rj = tracker.get_rating(t_a)
            simulated_elo_h.append(round(sim_ri, 2))
            simulated_elo_a.append(round(sim_rj, 2))
            ps_h, ps_a = bradley_terry_probability(
                ri=sim_ri, rj=sim_rj, ortf=ortf, trof=trof, hfa=hfa, b=b, c=c, beta_1=beta_1
            )
            simulated_home.append(round(ps_h, 4))
            simulated_away.append(round(ps_a, 4))
        else:
            simulated_elo_h.append(round(ri_given, 2))
            simulated_elo_a.append(round(rj_given, 2))
            ps_h, ps_a = bradley_terry_probability(
                ri=ri_given, rj=rj_given, ortf=ortf, trof=trof, hfa=hfa, b=b, c=c, beta_1=beta_1
            )
            simulated_home.append(round(ps_h, 4))
            simulated_away.append(round(ps_a, 4))

    result_df = predict_df.copy()
    n_games = len(result_df)
    result_df['paper_baseline_home'] = PAPER_BASELINE_HOME[:n_games] if n_games <= len(PAPER_BASELINE_HOME) else [0.5]*n_games
    result_df['paper_baseline_away'] = PAPER_BASELINE_AWAY[:n_games] if n_games <= len(PAPER_BASELINE_AWAY) else [0.5]*n_games
    result_df['remediated_prob_home'] = remediated_home
    result_df['remediated_prob_away'] = remediated_away
    result_df['simulated_elo_home'] = simulated_elo_h
    result_df['simulated_elo_away'] = simulated_elo_a
    result_df['simulated_prob_home'] = simulated_home
    result_df['simulated_prob_away'] = simulated_away

    return result_df


def plot_bracket_comparison(comparison_df: pd.DataFrame, output_path: str) -> None:
    """Generate and save grouped bar chart comparing paper baseline, remediated, and simulated predictions."""
    plt.figure(figsize=(12, 7))
    n = len(comparison_df)
    x = np.arange(n)
    width = 0.26

    plt.bar(x - width, comparison_df['paper_baseline_home'], width, label='Paper Baseline P(Home)', color='#4C72B0', edgecolor='black')
    plt.bar(x, comparison_df['remediated_prob_home'], width, label='Remediated (Static Elo) P(Home)', color='#55A868', edgecolor='black')
    plt.bar(x + width, comparison_df['simulated_prob_home'], width, label='Simulated (Calibrated) P(Home)', color='#C44E52', edgecolor='black')

    plt.ylabel('Home Win Probability', fontsize=12)
    plt.title('Tournament Bracket Matchups: Paper vs Remediated vs Simulated Calibrated Model', fontsize=14, pad=12)
    plt.xticks(
        x,
        [f"Gm {g}\n{h[:11]}..\nvs {a[:11]}.." for g, h, a in zip(comparison_df['game'], comparison_df['team_home'], comparison_df['team_away'])],
        fontsize=9,
    )
    plt.ylim(0.0, 1.0)
    plt.axhline(0.50, color='gray', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

