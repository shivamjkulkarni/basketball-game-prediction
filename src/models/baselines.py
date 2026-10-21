"""
Competitive Benchmark Baselines for Basketball Game Prediction:
1. Standard Zero-Sum Vanilla Elo (Fixed k=32, HFA=34.5)
2. Efficiency Margin Regression (KenPom / Torvik Style)
3. Static Bradley-Terry MLE (L2-Regularized Latent Strengths)
4. Gradient Boosted Box-Score Classifier (LightGBM on 10-game rolling differentials)
"""

from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    # Test if libomp dylib actually loads
    _ = lgb.__version__
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False


class VanillaElo:
    """
    Standard zero-sum Vanilla Elo baseline without dynamic WHR adjustments.
    - k = 32
    - HFA = 34.52 (empirical home-field advantage)
    - c = 400.0 (classical logistic scale)
    - No fatigue, LRE, or box-score PAF adjustments
    """

    def __init__(self, k: float = 32.0, hfa: float = 34.52, c: float = 400.0, base_elo: float = 1500.0):
        self.k = float(k)
        self.hfa = float(hfa)
        self.c = float(c)
        self.base_elo = float(base_elo)
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_elo)

    def predict_proba(self, team_i: str, team_j: str, ha_sign: float = 0.0) -> float:
        ri = self.get_rating(team_i)
        rj = self.get_rating(team_j)
        hfa_pts = ha_sign * self.hfa
        exp_val = (rj - (ri + hfa_pts)) / self.c
        exp_clamped = max(min(exp_val, 50.0), -50.0)
        return 1.0 / (1.0 + (10.0 ** exp_clamped))

    def update(self, team_i: str, team_j: str, y_i: float, prob_i: float) -> None:
        delta = self.k * (y_i - prob_i)
        ri = self.get_rating(team_i)
        rj = self.get_rating(team_j)
        self.ratings[team_i] = ri + delta
        self.ratings[team_j] = rj - delta


class EfficiencyMarginRegression:
    """
    Efficiency Margin Regression (KenPom / Torvik Style).
    Maps expanding rolling Adjusted Net Efficiency differentials to win probability via Logistic Regression:
        P(i wins) = sigmoid(beta_0 + beta_1 * (NetEff_i - NetEff_j) + beta_2 * HomeIndicator)
    """

    def __init__(self, default_net_eff: float = 0.0):
        self.default_net_eff = float(default_net_eff)
        self.model = LogisticRegression(fit_intercept=True, C=1.0, solver="lbfgs", random_state=42)
        self.is_fitted = False

    def extract_features(
        self,
        net_eff_i: float,
        net_eff_j: float,
        ha_sign: float,
    ) -> np.ndarray:
        return np.array([net_eff_i - net_eff_j, ha_sign], dtype=np.float64)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, net_eff_i: float, net_eff_j: float, ha_sign: float) -> float:
        if not self.is_fitted:
            # Fallback prior sigmoid
            diff = net_eff_i - net_eff_j + (ha_sign * 3.0)
            return 1.0 / (1.0 + math.exp(-0.08 * diff))
        x = np.array([[net_eff_i - net_eff_j, ha_sign]], dtype=np.float64)
        return float(self.model.predict_proba(x)[0, 1])


class StaticBradleyTerryMLE:
    """
    Season-Aggregate Static Bradley-Terry Maximum Likelihood Estimator (MLE) with L2 Shrinkage.
    Fits latent team strength vector lambda across training matchups without chronological updates:
        logit(P(i wins)) = lambda_i - lambda_j + mu_H * HomeIndicator
    """

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = float(c_reg)
        self.model = LogisticRegression(penalty="l2", C=self.c_reg, fit_intercept=False, solver="lbfgs", random_state=42)
        self.team_to_idx: Dict[str, int] = {}
        self.num_teams = 0
        self.is_fitted = False

    def fit(
        self,
        matchups: List[Tuple[str, str, float, float]],
        all_teams: List[str],
    ) -> None:
        """
        Args:
            matchups: List of (team_i, team_j, ha_sign, y_i) tuples.
            all_teams: List of all unique team names in dataset.
        """
        self.team_to_idx = {t: idx for idx, t in enumerate(all_teams)}
        self.num_teams = len(all_teams)

        # Design matrix: columns 0..N-1 are team indicators (+1 for i, -1 for j), column N is home indicator
        X = np.zeros((len(matchups), self.num_teams + 1), dtype=np.float64)
        y = np.zeros(len(matchups), dtype=np.float64)

        for m_idx, (t_i, t_j, ha_sign, outcome) in enumerate(matchups):
            i_idx = self.team_to_idx.get(t_i)
            j_idx = self.team_to_idx.get(t_j)
            if i_idx is not None:
                X[m_idx, i_idx] = 1.0
            if j_idx is not None:
                X[m_idx, j_idx] = -1.0
            X[m_idx, self.num_teams] = ha_sign
            y[m_idx] = outcome

        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, team_i: str, team_j: str, ha_sign: float = 0.0) -> float:
        if not self.is_fitted:
            return 0.50

        x = np.zeros((1, self.num_teams + 1), dtype=np.float64)
        i_idx = self.team_to_idx.get(team_i)
        j_idx = self.team_to_idx.get(team_j)

        if i_idx is not None:
            x[0, i_idx] = 1.0
        if j_idx is not None:
            x[0, j_idx] = -1.0
        x[0, self.num_teams] = ha_sign

        return float(self.model.predict_proba(x)[0, 1])


class LightGBMBoxScoreClassifier:
    """
    Supervised Gradient Boosted Decision Tree (LightGBM) trained on expanding 10-game
    rolling box-score differentials (eFG%, TS%, TOV%, DREB%, OREB%, NetEff) and fatigue factors.
    """

    FEATURE_NAMES = [
        "diff_net_eff",
        "diff_efg",
        "diff_ts",
        "diff_tov",
        "diff_dreb",
        "diff_oreb",
        "diff_rd",
        "diff_td",
        "diff_tzd",
        "ha_sign",
    ]

    def __init__(self, window_size: int = 10, random_state: int = 42):
        self.window_size = window_size
        self.random_state = random_state
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_depth=4,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, x: np.ndarray) -> float:
        if not self.is_fitted:
            return 0.50
        if x.ndim == 1:
            x = x.reshape(1, -1)
        prob = float(self.model.predict_proba(x)[0, 1])
        return max(min(prob, 1.0 - 1e-15), 1e-15)
