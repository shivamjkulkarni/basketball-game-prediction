"""
Unit tests for Bayesian optimization modules (sampler, objective, and tuner).
"""

import os
import unittest
import numpy as np
import optuna

from src.optimization.objective import ChronologicalObjective
from src.optimization.sampler import HyperparameterSampler


class TestOptimization(unittest.TestCase):

    def test_sampler_simplex_constraint(self):
        """Verify that sampled box-score weights strictly form an 8-D normalized simplex."""
        study = optuna.create_study(direction="minimize")
        trial = study.ask()

        params = HyperparameterSampler.sample(trial)

        self.assertIn("alpha", params)
        self.assertEqual(len(params["alpha"]), 8)
        self.assertAlmostEqual(sum(params["alpha"]), 1.0, places=6)
        for a in params["alpha"]:
            self.assertGreater(a, 0.0)

        # Check parameter bounds
        self.assertTrue(8 <= params["k"] <= 64)
        self.assertTrue(200.0 <= params["c"] <= 1500.0)
        self.assertTrue(2.0 <= params["b"] <= 150.0)
        self.assertTrue(0.0 <= params["hfa"] <= 120.0)
        self.assertTrue(0.001 <= params["beta_1"] <= 0.20)
        self.assertTrue(1e-8 <= params["beta_2"] <= 1e-4)
        self.assertTrue(0.10 <= params["beta_3"] <= 2.00)
        self.assertTrue(0.001 <= params["beta_4"] <= 0.20)
        self.assertTrue(0.20 <= params["k_scale"] <= 3.00)

    def test_baseline_params_formatting(self):
        """Verify baseline parameters dictionary formatting."""
        baseline_raw = HyperparameterSampler.get_baseline_params()
        formatted = HyperparameterSampler.format_params_dict(baseline_raw)

        self.assertEqual(formatted["k"], 32)
        self.assertAlmostEqual(formatted["c"], 993.39, places=2)
        self.assertAlmostEqual(formatted["b"], 99.39, places=2)
        self.assertAlmostEqual(formatted["hfa"], 72.52, places=2)
        self.assertAlmostEqual(formatted["beta_1"], 0.02, places=4)
        self.assertAlmostEqual(formatted["beta_2"], 1e-6, places=8)
        self.assertAlmostEqual(sum(formatted["alpha"]), 1.0, places=6)

    def test_objective_evaluation_integrity(self):
        """Verify chronological objective evaluation on historical games."""
        data_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            "final_final_data.csv",
        )
        if not os.path.exists(data_path):
            self.skipTest(f"Data file not found at {data_path}")

        objective = ChronologicalObjective(data_path=data_path, train_ratio=0.70)
        self.assertEqual(objective.num_train_games, 3653)
        self.assertEqual(objective.num_val_games, 1566)

        baseline_params = HyperparameterSampler.format_params_dict(
            HyperparameterSampler.get_baseline_params()
        )
        loss, brier, acc = objective.evaluate(baseline_params)

        # Log-Loss should be finite and realistic (~0.5 - 0.7)
        self.assertGreater(loss, 0.35)
        self.assertLess(loss, 0.85)

        # Brier score in (0.1, 0.3)
        self.assertGreater(brier, 0.10)
        self.assertLess(brier, 0.35)

        # Accuracy above 65% for college basketball
        self.assertGreater(acc, 0.65)
        self.assertLessEqual(acc, 1.00)


if __name__ == "__main__":
    unittest.main()

