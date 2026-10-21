"""
Hyperparameter Search Space & Simplex Sampler for Bradley-Terry / Elo Optimization.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import optuna

from src.config import (
    ALPHA_WEIGHTS,
    BASE_B,
    BETA_1,
    BETA_2,
    BETA_3,
    BETA_4,
    DIVISOR_C,
    HFA_ELO,
    K_FACTOR,
    K_SCALE_LOGISTIC,
    METRIC_KEYS,
)


class HyperparameterSampler:
    """
    Constructs the Optuna search distributions covering the complete mathematical parameter surface:
    - k (Update sensitivity): integer in [8, 64] (step 2)
    - c (Odds divisor): float uniform in [200.0, 1500.0]
    - b (Probability base): float uniform in [2.0, 150.0]
    - HFA (Home-field advantage): float uniform in [0.0, 120.0]
    - beta_1 (Fatigue sensitivity): float log-uniform in [0.001, 0.20]
    - beta_2 (Travel distance penalty): float log-uniform in [1e-8, 1e-4]
    - beta_3 (Lead retention scaling): float uniform in [0.10, 2.00]
    - beta_4 (Performance dampening): float log-uniform in [0.001, 0.20]
    - k_scale (Logistic AGPF scale): float uniform in [0.20, 3.00]
    - alpha_1 ... alpha_8: 8-D normalized simplex (sum = 1.0)
    """

    NUM_BOX_METRICS = len(METRIC_KEYS)  # 8

    @classmethod
    def sample(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample a candidate hyperparameter vector from the search distributions.
        """
        k = trial.suggest_int("k", 8, 64, step=2)
        c = trial.suggest_float("c", 200.0, 1500.0)
        b = trial.suggest_float("b", 2.0, 150.0)
        hfa = trial.suggest_float("hfa", 0.0, 120.0)
        beta_1 = trial.suggest_float("beta_1", 0.001, 0.20, log=True)
        beta_2 = trial.suggest_float("beta_2", 1e-8, 1e-4, log=True)
        beta_3 = trial.suggest_float("beta_3", 0.10, 2.00)
        beta_4 = trial.suggest_float("beta_4", 0.001, 0.20, log=True)
        k_scale = trial.suggest_float("k_scale", 0.20, 3.00)

        # 8-D Dirichlet / Normalized Simplex representation
        raw_weights = []
        for i in range(cls.NUM_BOX_METRICS):
            w = trial.suggest_float(f"w_alpha_{i}", 0.01, 1.0)
            raw_weights.append(w)

        sum_weights = sum(raw_weights)
        alpha_weights = [w / sum_weights for w in raw_weights]

        return {
            "k": k,
            "c": c,
            "b": b,
            "hfa": hfa,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "beta_3": beta_3,
            "beta_4": beta_4,
            "k_scale": k_scale,
            "alpha": alpha_weights,
            "raw_weights": raw_weights,
        }

    @classmethod
    def get_baseline_params(cls) -> Dict[str, Any]:
        """
        Return the paper baseline configuration formatted for study enqueuing.
        """
        baseline_dict = {
            "k": K_FACTOR,
            "c": DIVISOR_C,
            "b": BASE_B,
            "hfa": HFA_ELO,
            "beta_1": BETA_1,
            "beta_2": BETA_2,
            "beta_3": BETA_3,
            "beta_4": BETA_4,
            "k_scale": K_SCALE_LOGISTIC,
        }
        for i, weight in enumerate(ALPHA_WEIGHTS):
            baseline_dict[f"w_alpha_{i}"] = float(weight)

        return baseline_dict

    @classmethod
    def format_params_dict(cls, optuna_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Optuna's flat params dict into structured hyperparameter format.
        """
        raw_weights = [optuna_params.get(f"w_alpha_{i}", ALPHA_WEIGHTS[i]) for i in range(cls.NUM_BOX_METRICS)]
        total_w = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
        normalized_alpha = [float(w / total_w) for w in raw_weights]

        return {
            "k": int(optuna_params.get("k", K_FACTOR)),
            "c": float(optuna_params.get("c", DIVISOR_C)),
            "b": float(optuna_params.get("b", BASE_B)),
            "hfa": float(optuna_params.get("hfa", HFA_ELO)),
            "beta_1": float(optuna_params.get("beta_1", BETA_1)),
            "beta_2": float(optuna_params.get("beta_2", BETA_2)),
            "beta_3": float(optuna_params.get("beta_3", BETA_3)),
            "beta_4": float(optuna_params.get("beta_4", BETA_4)),
            "k_scale": float(optuna_params.get("k_scale", K_SCALE_LOGISTIC)),
            "alpha": normalized_alpha,
            "metric_names": METRIC_KEYS,
        }

