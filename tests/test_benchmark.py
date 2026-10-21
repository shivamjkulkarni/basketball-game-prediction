"""
Unit tests for competitive baseline models and evaluation calibration metrics.
"""

import unittest
import numpy as np

from src.evaluation.benchmark import compute_expected_calibration_error
from src.models.baselines import (
    EfficiencyMarginRegression,
    LightGBMBoxScoreClassifier,
    StaticBradleyTerryMLE,
    VanillaElo,
)


class TestBenchmark(unittest.TestCase):

    def test_expected_calibration_error(self):
        """Verify ECE calculation for perfectly calibrated and miscalibrated predictions."""
        # Perfect calibration: predicted 0.8 always, 80% 1s
        y_true_perf = np.array([1, 1, 1, 1, 0] * 20)  # 80% 1s
        y_prob_perf = np.full(100, 0.8)
        ece_perf = compute_expected_calibration_error(y_true_perf, y_prob_perf, n_bins=10)
        self.assertAlmostEqual(ece_perf, 0.0, places=6)

        # Miscalibrated: predicted 0.9 always, but only 50% 1s -> ECE = |0.5 - 0.9| = 0.40
        y_true_bad = np.array([1, 0] * 50)
        y_prob_bad = np.full(100, 0.9)
        ece_bad = compute_expected_calibration_error(y_true_bad, y_prob_bad, n_bins=10)
        self.assertAlmostEqual(ece_bad, 0.40, places=5)

    def test_vanilla_elo(self):
        """Verify Vanilla Elo probability bounds and zero-sum conservation."""
        model = VanillaElo(k=32.0, hfa=34.52, c=400.0)
        p_home = model.predict_proba("Duke", "UNC", ha_sign=1.0)
        self.assertGreater(p_home, 0.50)  # Home advantage should tip parity

        r_duke_old = model.get_rating("Duke")
        r_unc_old = model.get_rating("UNC")
        model.update("Duke", "UNC", y_i=1.0, prob_i=p_home)

        delta_duke = model.get_rating("Duke") - r_duke_old
        delta_unc = model.get_rating("UNC") - r_unc_old
        self.assertAlmostEqual(delta_duke + delta_unc, 0.0, places=7)

    def test_efficiency_margin_regression(self):
        """Verify rolling net efficiency regression model."""
        model = EfficiencyMarginRegression()
        X_toy = np.array([
            [15.0, 1.0],
            [-15.0, -1.0],
            [5.0, 0.0],
            [-5.0, 0.0],
        ])
        y_toy = np.array([1.0, 0.0, 1.0, 0.0])
        model.fit(X_toy, y_toy)

        p_strong = model.predict_proba(15.0, -15.0, 1.0)
        p_weak = model.predict_proba(-15.0, 15.0, -1.0)

        self.assertGreater(p_strong, 0.50)
        self.assertLess(p_weak, 0.50)
        self.assertTrue(0.0 < p_strong < 1.0)

    def test_static_bradley_terry_mle(self):
        """Verify Static Bradley-Terry MLE model."""
        model = StaticBradleyTerryMLE(c_reg=1.0)
        matchups = [
            ("TeamA", "TeamB", 1.0, 1.0),
            ("TeamA", "TeamB", -1.0, 1.0),
            ("TeamB", "TeamC", 0.0, 1.0),
            ("TeamC", "TeamA", 0.0, 0.0),
        ]
        all_teams = ["TeamA", "TeamB", "TeamC"]
        model.fit(matchups, all_teams)

        # TeamA is undefeated vs B and C -> prob(TeamA vs TeamC) > 0.50
        p_ac = model.predict_proba("TeamA", "TeamC", 0.0)
        self.assertGreater(p_ac, 0.50)
        self.assertTrue(0.0 < p_ac < 1.0)

    def test_lightgbm_box_score_classifier(self):
        """Verify LightGBM box-score classifier predictions."""
        model = LightGBMBoxScoreClassifier(random_state=42)
        X_toy = np.random.randn(50, 10)
        y_toy = (X_toy[:, 0] > 0).astype(float)
        model.fit(X_toy, y_toy)

        p = model.predict_proba(X_toy[0])
        self.assertTrue(0.0 < p < 1.0)


if __name__ == "__main__":
    unittest.main()

