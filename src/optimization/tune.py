"""
Optuna Study Orchestrator, Pruning Controller, and Diagnostic Visualizer.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import optuna.visualization.matplotlib as ovm

from src.optimization.objective import ChronologicalObjective
from src.optimization.sampler import HyperparameterSampler


def run_optimization_study(
    data_path: str = "data/final_final_data.csv",
    bracket_path: str = "data/predict.csv",
    n_trials: int = 200,
    output_dir: str = "output_data",
    figures_dir: str = "output_figures",
    seed: int = 42,
) -> Tuple[optuna.Study, Dict[str, Any], Dict[str, Any]]:
    """
    Execute empirical Bayesian hyperparameter optimization over 70/30 chronological split.

    Args:
        data_path: Path to chronological match logs.
        bracket_path: Path to bracket prediction matches.
        n_trials: Number of Optuna trials to evaluate.
        output_dir: Directory for output JSON and CSV records.
        figures_dir: Directory for Optuna diagnostic plots.
        seed: Random seed for TPESampler.

    Returns:
        Tuple[optuna.Study, Dict, Dict]: (study, baseline_metrics, optimized_summary)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Initialize fast objective evaluator
    objective = ChronologicalObjective(data_path=data_path, train_ratio=0.70, pruning_interval=250)

    # 2. Benchmark baseline parameters (Trial 0 equivalent)
    baseline_raw = HyperparameterSampler.get_baseline_params()
    baseline_params = HyperparameterSampler.format_params_dict(baseline_raw)
    base_loss, base_brier, base_acc = objective.evaluate(baseline_params)

    baseline_metrics = {
        "log_loss": float(base_loss),
        "brier_score": float(base_brier),
        "accuracy": float(base_acc),
        "parameters": baseline_params,
    }

    print("\n" + "=" * 65)
    print("PAPER BASELINE PERFORMANCE (OUT-OF-SAMPLE VALIDATION 30%)")
    print("=" * 65)
    print(f"Baseline Log-Loss:    {base_loss:.5f}")
    print(f"Baseline Brier Score: {base_brier:.5f}")
    print(f"Baseline Accuracy:    {base_acc * 100:.2f}%")
    print("=" * 65 + "\n")

    # 3. Setup Optuna Study with Multivariate TPESampler and MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1, interval_steps=1)

    study = optuna.create_study(
        study_name="basketball_elo_bayesian_optimization",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Enqueue paper baseline trial
    study.enqueue_trial(baseline_raw)
    print(f"Enqueued baseline trial. Launching {n_trials} Optuna optimization trials...")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 4. Extract best trial
    best_trial = study.best_trial
    best_loss = float(best_trial.value)
    best_brier = float(best_trial.user_attrs.get("brier_score", 0.0))
    best_acc = float(best_trial.user_attrs.get("accuracy", 0.0))
    best_params = HyperparameterSampler.format_params_dict(best_trial.params)

    loss_lift_pct = ((base_loss - best_loss) / base_loss) * 100.0
    brier_lift_pct = ((base_brier - best_brier) / base_brier) * 100.0
    acc_lift_pct = (best_acc - base_acc) * 100.0

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print("\n" + "=" * 65)
    print("BAYESIAN OPTIMIZATION STUDY CONVERGENCE")
    print("=" * 65)
    print(f"Total Trials:         {len(study.trials)}")
    print(f"Completed Trials:     {completed_trials}")
    print(f"Pruned Trials:        {pruned_trials}")
    print(f"Best Trial Index:     #{best_trial.number}")
    print(f"Optimized Log-Loss:   {best_loss:.5f} (Lift: {loss_lift_pct:+.2f}%)")
    print(f"Optimized Brier:      {best_brier:.5f} (Lift: {brier_lift_pct:+.2f}%)")
    print(f"Optimized Accuracy:   {best_acc * 100:.2f}% (Lift: {acc_lift_pct:+.2f}%)")
    print("=" * 65 + "\n")

    summary_data = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "completed_trials": completed_trials,
        "pruned_trials": pruned_trials,
        "best_trial_number": best_trial.number,
        "metrics": {
            "baseline": {
                "log_loss": base_loss,
                "brier_score": base_brier,
                "accuracy": base_acc,
            },
            "optimized": {
                "log_loss": best_loss,
                "brier_score": best_brier,
                "accuracy": best_acc,
            },
            "lift": {
                "log_loss_pct": loss_lift_pct,
                "brier_score_pct": brier_lift_pct,
                "accuracy_pts": acc_lift_pct,
            },
        },
        "optimized_parameters": best_params,
        "baseline_parameters": baseline_params,
    }

    # 5. Persist optimized parameters JSON
    params_json_path = os.path.join(output_dir, "optimized_parameters.json")
    with open(params_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved optimized hyperparameters to: {params_json_path}")

    # 6. Generate Optuna Diagnostic Visualizations
    _generate_diagnostic_plots(study, figures_dir)

    return study, baseline_metrics, summary_data


def _generate_diagnostic_plots(study: optuna.Study, figures_dir: str) -> None:
    """Generate and save publication-quality Optuna diagnostic charts."""
    try:
        # Plot 1: Optimization History
        fig_hist = ovm.plot_optimization_history(study)
        hist_path = os.path.join(figures_dir, "optuna_optimization_history.png")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"Saved figure: {hist_path}")
    except Exception as e:
        print(f"[Warning] Failed to export optimization history plot: {e}")

    try:
        # Plot 2: Parameter Importances
        fig_imp = ovm.plot_param_importances(study)
        imp_path = os.path.join(figures_dir, "optuna_param_importances.png")
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"Saved figure: {imp_path}")
    except Exception as e:
        print(f"[Warning] Failed to export parameter importances plot: {e}")

    try:
        # Plot 3: Slice Plot for key parameters
        fig_slice = ovm.plot_slice(study, params=["k", "c", "b", "hfa", "k_scale"])
        slice_path = os.path.join(figures_dir, "optuna_slice_plot.png")
        plt.tight_layout()
        plt.savefig(slice_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"Saved figure: {slice_path}")
    except Exception as e:
        print(f"[Warning] Failed to export slice plot: {e}")


