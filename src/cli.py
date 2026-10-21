"""
Unified Command-Line Interface (CLI) Orchestrator for Basketball Prediction Engine.
Supports:
  python -m src.cli run --data-path data/final_final_data.csv --export-standings --plot-diagnostics
  python -m src.cli predict --bracket-file data/predict.csv
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    ALPHA_WEIGHTS,
    BASE_B,
    BASE_ELO,
    BETA_1,
    BETA_2,
    BETA_3,
    BETA_4,
    DIVISOR_C,
    HFA_ELO,
    K_FACTOR,
    K_SCALE_LOGISTIC,
)
from src.data.loader import extract_team_regions, load_game_logs
from src.evaluation.diagnostics import (
    compute_conservation_metrics,
    export_regional_standings,
    plot_elo_distribution,
    plot_regional_top16,
)
from src.evaluation.predict import evaluate_bracket_predictions, plot_bracket_comparison
from src.features.fatigue import calculate_fatigue_indices
from src.features.metrics import BoxScoreStandardizer
from src.features.schedule import ScheduleTracker
from src.models.bradley_terry import bradley_terry_probability
from src.models.elo import EloTracker


def safe_num(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def load_active_parameters(
    output_dir: str = "output_data",
    use_optimized: bool = True,
) -> Dict[str, Any]:
    """
    Load calibrated parameters from optimized_parameters.json if available,
    otherwise fall back to paper baseline constants.
    """
    opt_path = os.path.join(output_dir, "optimized_parameters.json")
    if use_optimized and os.path.exists(opt_path):
        with open(opt_path, "r") as f:
            data = json.load(f)
        opt_params = data.get("optimized_parameters", {})
        if opt_params:
            return {
                "source": f"Optimized ({opt_path})",
                "k": float(opt_params.get("k", K_FACTOR)),
                "c": float(opt_params.get("c", DIVISOR_C)),
                "b": float(opt_params.get("b", BASE_B)),
                "hfa": float(opt_params.get("hfa", HFA_ELO)),
                "beta_1": float(opt_params.get("beta_1", BETA_1)),
                "beta_2": float(opt_params.get("beta_2", BETA_2)),
                "beta_3": float(opt_params.get("beta_3", BETA_3)),
                "beta_4": float(opt_params.get("beta_4", BETA_4)),
                "k_scale": float(opt_params.get("k_scale", K_SCALE_LOGISTIC)),
                "alpha": list(opt_params.get("alpha", ALPHA_WEIGHTS)),
            }

    return {
        "source": "Paper Baseline (src/config.py)",
        "k": float(K_FACTOR),
        "c": float(DIVISOR_C),
        "b": float(BASE_B),
        "hfa": float(HFA_ELO),
        "beta_1": float(BETA_1),
        "beta_2": float(BETA_2),
        "beta_3": float(BETA_3),
        "beta_4": float(BETA_4),
        "k_scale": float(K_SCALE_LOGISTIC),
        "alpha": list(ALPHA_WEIGHTS),
    }


def run_full_season(
    data_path: str,
    output_dir: str = "output_data",
    figures_dir: str = "output_figures",
    export_standings: bool = True,
    plot_diagnostics: bool = True,
    use_optimized: bool = True,
) -> Tuple[EloTracker, ScheduleTracker, BoxScoreStandardizer]:
    """
    Execute end-to-end full season simulation across historical games.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    params = load_active_parameters(output_dir=output_dir, use_optimized=use_optimized)
    print(f"Active Parameter Configuration: {params['source']}")
    print(
        f"  k={params['k']:.0f}, c={params['c']:.2f}, b={params['b']:.2f}, "
        f"HFA={params['hfa']:.2f}, k_scale={params['k_scale']:.2f}"
    )

    print(f"Loading and sorting game data from: {data_path}")
    clean_df = load_game_logs(data_path)
    team_regions = extract_team_regions(clean_df)

    unique_game_ids = clean_df['game_id'].unique()
    num_games = len(unique_game_ids)
    print(f"Ingested {len(clean_df)} game logs across {num_games} unique matchups.")

    # Initialize modules with active parameters
    standardizer = BoxScoreStandardizer.fit_from_dataframe(
        clean_df,
        alpha=params["alpha"],
        k_scale=params["k_scale"],
    )
    elo_tracker = EloTracker(base_elo=BASE_ELO)
    schedule_tracker = ScheduleTracker(default_prior_wr=0.50)

    processed_records = []
    print(f"Executing dynamic chronological rating updates across {num_games} games...")

    for g_id in unique_game_ids:
        rows = clean_df[clean_df['game_id'] == g_id]
        if len(rows) != 2:
            continue

        r_i = rows.iloc[0]
        r_j = rows.iloc[1]

        t_i = str(r_i['team']).strip()
        t_j = str(r_j['team']).strip()

        res_i = str(r_i.get('w/l', '')).strip().upper()
        winner = t_i if res_i == 'W' else t_j
        loser = t_j if winner == t_i else t_i

        # Dynamic Strength of Schedule prior to this game
        sos_i = schedule_tracker.compute_sos(t_i, t_j)
        sos_j = schedule_tracker.compute_sos(t_j, t_i)

        # Performance Factor computation
        def _get_metrics(r):
            return {
                "NetEff": safe_num(r.get("neteff")),
                "eFG": safe_num(r.get("efg")),
                "TS": safe_num(r.get("ts")),
                "OREB": safe_num(r.get("oreb")),
                "DREB": safe_num(r.get("dreb")),
                "AST:TOV": safe_num(r.get("ast:tov")),
                "TOV": safe_num(r.get("tov")),
                "STL": safe_num(r.get("stl")),
            }

        agpf_i, a_gpf_i, paf_i = standardizer.compute_paf(t_i, _get_metrics(r_i))
        agpf_j, a_gpf_j, paf_j = standardizer.compute_paf(t_j, _get_metrics(r_j))

        # Fatigue parameters
        rd_i = safe_num(r_i.get("rest_days", r_j.get("oprd", 2.0)), 2.0)
        rd_j = safe_num(r_j.get("rest_days", r_i.get("oprd", 2.0)), 2.0)
        td_i = safe_num(r_i.get("travel_distance", r_j.get("optd", 0.0)), 0.0)
        td_j = safe_num(r_j.get("travel_distance", r_i.get("optd", 0.0)), 0.0)
        tzd_i = safe_num(r_i.get("tz_dif_h_e", 0.0), 0.0)
        tzd_j = safe_num(r_j.get("tz_dif_h_e", 0.0), 0.0)

        ortf, trof = calculate_fatigue_indices(
            rd_i=rd_i, rd_j=rd_j, td_i=td_i, td_j=td_j, tzd_i=tzd_i, tzd_j=tzd_j, beta_2=params["beta_2"]
        )

        # Home field advantage
        ha_i = safe_num(r_i.get("home_away", 0.0))
        hfa_val = params["hfa"]
        hfa_points = hfa_val if ha_i == 0.0 else (-hfa_val if ha_i == 1.0 else 0.0)

        # Expected probability
        ri_current = elo_tracker.get_rating(t_i)
        rj_current = elo_tracker.get_rating(t_j)
        prob_i, prob_j = bradley_terry_probability(
            ri=ri_current,
            rj=rj_current,
            ortf=ortf,
            trof=trof,
            hfa=hfa_points,
            b=params["b"],
            c=params["c"],
            beta_1=params["beta_1"],
        )

        lre_val = safe_num(r_i.get("lre", r_j.get("lre", 0.0)), 0.0)
        a_i = 1.0 if winner == t_i else 0.0

        # Zero-Sum Elo Update
        new_ri, new_rj, delta_r = elo_tracker.update(
            team_i=t_i,
            team_j=t_j,
            ai=a_i,
            ei=prob_i,
            lre=lre_val,
            paf_i=paf_i,
            paf_j=paf_j,
            k=params["k"],
            beta_3=params["beta_3"],
            beta_4=params["beta_4"],
        )

        # Record game outcome in schedule tracker AFTER computing this game's ratings
        schedule_tracker.record_game(winner=winner, loser=loser)

        processed_records.append({
            "game_id": g_id,
            "game_date": r_i['game_date'],
            "team_i": t_i,
            "team_j": t_j,
            "winner": winner,
            "old_elo_i": ri_current,
            "old_elo_j": rj_current,
            "new_elo_i": new_ri,
            "new_elo_j": new_rj,
            "delta_elo": delta_r,
            "prob_i": prob_i,
            "prob_j": prob_j,
            "agpf_i": agpf_i,
            "agpf_j": agpf_j,
            "paf_i": paf_i,
            "paf_j": paf_j,
            "sos_i": sos_i,
            "sos_j": sos_j,
        })

    # Save processed games
    games_df = pd.DataFrame(processed_records)
    games_csv = os.path.join(output_dir, 'full_season_processed_games.csv')
    games_df.to_csv(games_csv, index=False)

    # Validate conservation
    metrics = compute_conservation_metrics(elo_tracker)
    print("\n" + "=" * 60)
    print("CONSERVATION & LEAGUE DIAGNOSTICS")
    print("=" * 60)
    print(f"Total Teams Registered:  {metrics['total_teams']}")
    print(f"Total League Elo Sum:    {metrics['total_elo_sum']:.2f}")
    print(f"Expected Elo Sum:        {metrics['expected_elo_sum']:.2f}")
    print(f"Sum Discrepancy:         {metrics['sum_discrepancy']:.6e}")
    print(f"League Mean Elo:         {metrics['mean_elo']:.4f} (Benchmark: 1500.00 ± 0.01)")
    print(f"Standard Deviation:      {metrics['std_elo']:.2f}")
    print(f"Rating Range:            [{metrics['min_elo']:.2f}, {metrics['max_elo']:.2f}]")
    print("=" * 60)

    # Save final team ratings
    ratings_records = [
        {
            "team": t,
            "region": team_regions.get(t, "Central"),
            "final_elo": elo_tracker.get_rating(t),
            "games_played": elo_tracker.games_played.get(t, 0),
            "mean_agpf": float(np.mean(standardizer.agpf_history.get(t, [0.50]))),
        }
        for t in elo_tracker.ratings.keys()
    ]
    ratings_df = pd.DataFrame(ratings_records).sort_values(by="final_elo", ascending=False).reset_index(drop=True)
    ratings_csv = os.path.join(output_dir, 'final_team_ratings.csv')
    ratings_df.to_csv(ratings_csv, index=False)
    print(f"Saved team ratings to: {ratings_csv}")

    # Export regional standings
    if export_standings:
        top16_df = export_regional_standings(ratings_df, top_n=16)
        top16_csv = os.path.join(output_dir, 'regional_top16_standings.csv')
        top16_df.to_csv(top16_csv, index=False)
        print(f"Saved regional top 16 standings to: {top16_csv}")

        if plot_diagnostics:
            fig2_path = os.path.join(figures_dir, 'regional_top16_standings.png')
            plot_regional_top16(top16_df, fig2_path)
            print(f"Saved figure: {fig2_path}")

    # Plot diagnostics
    if plot_diagnostics:
        fig1_path = os.path.join(figures_dir, 'elo_distribution.png')
        plot_elo_distribution(list(elo_tracker.ratings.values()), fig1_path, mean_elo=metrics['mean_elo'])
        print(f"Saved figure: {fig1_path}")

    return elo_tracker, schedule_tracker, standardizer


def run_bracket_predictions(
    bracket_file: str,
    output_dir: str = "output_data",
    figures_dir: str = "output_figures",
    tracker: Optional[EloTracker] = None,
    use_optimized: bool = True,
) -> pd.DataFrame:
    """
    Generate bracket win probabilities for tournament matchups.
    Automatically loads simulated ratings from output_dir/final_team_ratings.csv
    and calibrated parameters from output_dir/optimized_parameters.json if available.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    params = load_active_parameters(output_dir=output_dir, use_optimized=use_optimized)
    ratings_csv = os.path.join(output_dir, "final_team_ratings.csv")
    if tracker is None and os.path.exists(ratings_csv):
        print(f"Loading simulated end-of-season ratings from: {ratings_csv}")
        tracker = EloTracker.from_csv(ratings_csv, base_elo=BASE_ELO)

    print(f"Loading bracket matchups from: {bracket_file}")
    predict_df = pd.read_csv(bracket_file)

    comparison_df = evaluate_bracket_predictions(
        predict_df,
        tracker=tracker,
        b=params["b"],
        c=params["c"],
        beta_1=params["beta_1"],
        beta_2=params["beta_2"],
    )
    comp_csv = os.path.join(output_dir, 'bracket_predictions.csv')
    comparison_df.to_csv(comp_csv, index=False)
    print(f"Saved bracket predictions to: {comp_csv}")

    fig3_path = os.path.join(figures_dir, 'bracket_predictions_comparison.png')
    plot_bracket_comparison(comparison_df, fig3_path)
    print(f"Saved figure: {fig3_path}")

    print("\n" + "=" * 92)
    print("BRACKET PREDICTIONS COMPARISON (EAST REGION)")
    print("=" * 92)
    print(f"{'Gm':<3} {'Home Team':<26} {'Away Team':<26} {'Paper P(H)':<11} {'Remed P(H)':<11} {'Sim P(H)':<10}")
    print("-" * 92)
    for _, r in comparison_df.iterrows():
        print(
            f"{int(r['game']):<3} {r['team_home']:<26} {r['team_away']:<26} "
            f"{r['paper_baseline_home']:<11.2f} {r['remediated_prob_home']:<11.2f} {r['simulated_prob_home']:<10.2f}"
        )
    print("=" * 92)
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Basketball Prediction & Rating Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run full season simulation")
    run_parser.add_argument("--data-path", default="data/final_final_data.csv", help="Path to raw game logs CSV")
    run_parser.add_argument("--output-dir", default="output_data", help="Output directory for data tables")
    run_parser.add_argument("--figures-dir", default="output_figures", help="Output directory for figures")
    run_parser.add_argument("--export-standings", action="store_true", default=True, help="Export regional top 16 standings")
    run_parser.add_argument("--plot-diagnostics", action="store_true", default=True, help="Generate diagnostic plots")
    run_parser.add_argument("--baseline-params", action="store_true", default=False, help="Use paper baseline parameters instead of optimized")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Generate tournament bracket predictions")
    pred_parser.add_argument("--bracket-file", default="data/predict.csv", help="Path to bracket matchups CSV")
    pred_parser.add_argument("--output-dir", default="output_data", help="Output directory for data tables")
    pred_parser.add_argument("--figures-dir", default="output_figures", help="Output directory for figures")
    pred_parser.add_argument("--baseline-params", action="store_true", default=False, help="Use paper baseline parameters instead of optimized")

    # Command: tune
    tune_parser = subparsers.add_parser("tune", help="Run empirical Bayesian hyperparameter optimization")
    tune_parser.add_argument("--data-path", default="data/final_final_data.csv", help="Path to raw game logs CSV")
    tune_parser.add_argument("--bracket-file", default="data/predict.csv", help="Path to bracket matchups CSV")
    tune_parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials")
    tune_parser.add_argument("--output-dir", default="output_data", help="Output directory for data tables")
    tune_parser.add_argument("--figures-dir", default="output_figures", help="Output directory for figures")
    tune_parser.add_argument("--export-best", action="store_true", default=True, help="Export optimized parameters")
    tune_parser.add_argument("--seed", type=int, default=42, help="Random seed for optimization")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-model competitive benchmarking suite")
    bench_parser.add_argument("--data-path", default="data/final_final_data.csv", help="Path to raw game logs CSV")
    bench_parser.add_argument("--optimized-params", default="output_data/optimized_parameters.json", help="Path to optimized parameters JSON")
    bench_parser.add_argument("--output-dir", default="output_data", help="Output directory for data tables")
    bench_parser.add_argument("--figures-dir", default="output_figures", help="Output directory for figures")
    bench_parser.add_argument("--export-table", action="store_true", default=True, help="Export comparison table CSV")

    args = parser.parse_args()

    if args.command == "run":
        run_full_season(
            data_path=args.data_path,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            export_standings=args.export_standings,
            plot_diagnostics=args.plot_diagnostics,
            use_optimized=not args.baseline_params,
        )
    elif args.command == "predict":
        run_bracket_predictions(
            bracket_file=args.bracket_file,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            use_optimized=not args.baseline_params,
        )
    elif args.command == "tune":
        from src.optimization.tune import run_optimization_study
        run_optimization_study(
            data_path=args.data_path,
            bracket_path=args.bracket_file,
            n_trials=args.trials,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            seed=args.seed,
        )
    elif args.command == "benchmark":
        from src.evaluation.benchmark import BenchmarkSuite
        suite = BenchmarkSuite(
            data_path=args.data_path,
            optimized_params_path=args.optimized_params,
        )
        suite.run_benchmark(
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
