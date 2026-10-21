"""
Evaluation & Diagnostic Engine: Conservation Validation, Distribution Analysis, and Regional Rankings.
"""

from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.config import BASE_ELO, REGIONS
from src.models.elo import EloTracker


def compute_conservation_metrics(tracker: EloTracker) -> Dict[str, float]:
    """
    Validate mathematical conservation of total league rating points:
    Ensures that sum(r_t) == N * 1500.00 and mean(r_t) == 1500.0000.

    Args:
        tracker (EloTracker): Stateful tracker containing all team ratings.

    Returns:
        Dict[str, float]: Diagnostic metrics dictionary.
    """
    teams = list(tracker.ratings.keys())
    ratings = [tracker.ratings[t] for t in teams]
    num_teams = len(teams)

    if num_teams == 0:
        return {"total_teams": 0, "mean_elo": BASE_ELO, "sum_discrepancy": 0.0}

    total_sum = sum(ratings)
    expected_sum = num_teams * tracker.base_elo
    discrepancy = abs(total_sum - expected_sum)
    mean_elo = float(np.mean(ratings))
    std_elo = float(np.std(ratings))
    median_elo = float(np.median(ratings))
    min_elo = float(np.min(ratings))
    max_elo = float(np.max(ratings))

    return {
        "total_teams": num_teams,
        "total_elo_sum": total_sum,
        "expected_elo_sum": expected_sum,
        "sum_discrepancy": discrepancy,
        "mean_elo": mean_elo,
        "std_elo": std_elo,
        "median_elo": median_elo,
        "min_elo": min_elo,
        "max_elo": max_elo,
        "span": max_elo - min_elo,
    }


def export_regional_standings(
    ratings_df: pd.DataFrame,
    regions: Optional[List[str]] = None,
    top_n: int = 16,
) -> pd.DataFrame:
    """
    Extract top N collegiate teams per geographic region.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing 'team', 'region', 'final_elo'.
        regions (Optional[List[str]]): List of region names (default North, South, East, West).
        top_n (int): Number of teams to return per region (default 16).

    Returns:
        pd.DataFrame: DataFrame containing regional top N rankings.
    """
    target_regions = regions or REGIONS
    records = []

    for reg in target_regions:
        reg_teams = ratings_df[ratings_df['region'].str.lower() == reg.lower()].sort_values(
            by='final_elo', ascending=False
        ).head(top_n)

        for rank, (_, row) in enumerate(reg_teams.iterrows(), 1):
            records.append({
                "region": reg,
                "rank": rank,
                "team": row['team'],
                "final_elo": row['final_elo'],
                "games_played": row.get('games_played', 0),
                "mean_agpf": row.get('mean_agpf', 0.50),
            })

    return pd.DataFrame(records)


def plot_elo_distribution(ratings: List[float], output_path: str, mean_elo: float = BASE_ELO) -> None:
    """Generate and save distribution histogram of final team ratings."""
    plt.figure(figsize=(10, 6))
    plt.hist(ratings, bins=35, color='#1f77b4', edgecolor='black', alpha=0.75, density=True)
    plt.axvline(mean_elo, color='red', linestyle='dashed', linewidth=2, label=f'Mean Elo: {mean_elo:.2f}')
    plt.axvline(BASE_ELO, color='gold', linestyle='dotted', linewidth=2, label=f'Baseline: {BASE_ELO:.2f}')

    plt.title("Distribution of Final Team Elo Ratings (Zero-Sum Invariant Engine)", fontsize=14, pad=12)
    plt.xlabel("Elo Rating", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_regional_top16(top16_df: pd.DataFrame, output_path: str, regions: Optional[List[str]] = None) -> None:
    """Generate 2x2 regional horizontal bar charts for Top 16 programs."""
    target_regions = regions or REGIONS
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top 16 Elo Ratings Per Geographic Region", fontsize=18, y=0.98)
    cmap = plt.cm.viridis

    for idx, reg in enumerate(target_regions):
        ax = axes[idx // 2, idx % 2]
        reg_sub = top16_df[top16_df['region'].str.lower() == reg.lower()].sort_values(
            by='final_elo', ascending=True
        )

        teams = reg_sub['team'].tolist()
        elos = reg_sub['final_elo'].tolist()

        if not elos:
            ax.set_visible(False)
            continue

        norm = mcolors.Normalize(vmin=min(elos), vmax=max(elos))
        bar_colors = [cmap(norm(e)) for e in elos]

        bars = ax.barh(teams, elos, color=bar_colors, edgecolor='black', height=0.7)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 3, bar.get_y() + bar.get_height() / 2, f'{w:.1f}', va='center', fontsize=9, fontweight='bold')

        ax.set_title(f"{reg} Region", fontsize=14, pad=8)
        ax.set_xlabel("Elo Rating", fontsize=11)
        ax.set_xlim(min(elos) * 0.95, max(elos) * 1.06)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

