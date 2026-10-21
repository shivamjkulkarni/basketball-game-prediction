"""
End-to-end integration test for the pure Python rating pipeline.
Runs a multi-game simulation, verifying strict rating conservation and artifact generation.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.cli import run_bracket_predictions, run_full_season
from src.evaluation.diagnostics import compute_conservation_metrics
from src.models.elo import EloTracker


class TestPipelineE2E(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_data = os.path.join(self.temp_dir, "output_data")
        self.output_figures = os.path.join(self.temp_dir, "output_figures")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_e2e_subset_simulation_and_conservation(self):
        """
        Run simulation on a 200-game (400 rows) chronological subset from final_final_data.csv.
        Assert that mean Elo is conserved at 1500.00 +/- 0.01, CSV and figures are generated.
        """
        raw_data_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            "final_final_data.csv",
        )

        if not os.path.exists(raw_data_path):
            self.skipTest(f"Data file not found at {raw_data_path}")

        # Ingest first 200 unique games
        from src.data.loader import load_game_logs
        full_df = load_game_logs(raw_data_path)
        first_200_game_ids = full_df["game_id"].unique()[:200]
        subset_df = full_df[full_df["game_id"].isin(first_200_game_ids)].copy()

        temp_csv = os.path.join(self.temp_dir, "subset_games.csv")
        subset_df.to_csv(temp_csv, index=False)

        # Run pipeline
        elo_tracker, schedule_tracker, standardizer = run_full_season(
            data_path=temp_csv,
            output_dir=self.output_data,
            figures_dir=self.output_figures,
            export_standings=True,
            plot_diagnostics=True,
        )

        # 1. Assert conservation
        metrics = compute_conservation_metrics(elo_tracker)
        self.assertAlmostEqual(metrics["mean_elo"], 1500.0, places=2)
        self.assertAlmostEqual(
            metrics["total_elo_sum"],
            metrics["expected_elo_sum"],
            places=3,
        )

        # 2. Assert output files exist
        self.assertTrue(os.path.exists(os.path.join(self.output_data, "final_team_ratings.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_data, "full_season_processed_games.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_data, "regional_top16_standings.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_figures, "elo_distribution.png")))
        self.assertTrue(os.path.exists(os.path.join(self.output_figures, "regional_top16_standings.png")))

        # 3. Test bracket prediction with trained tracker
        predict_csv_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            "predict.csv",
        )
        if os.path.exists(predict_csv_path):
            pred_df = run_bracket_predictions(
                bracket_file=predict_csv_path,
                output_dir=self.output_data,
                figures_dir=self.output_figures,
                tracker=elo_tracker,
            )
            self.assertEqual(len(pred_df), 10)
            self.assertTrue(os.path.exists(os.path.join(self.output_data, "bracket_predictions.csv")))
            self.assertTrue(os.path.exists(os.path.join(self.output_figures, "bracket_predictions_comparison.png")))


if __name__ == "__main__":
    unittest.main()

