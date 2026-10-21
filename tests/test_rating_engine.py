"""
Unit test suite for the refactored basketball rating engine.
Verifies all 5 synthetic and edge scenarios required by acceptance criteria.
"""

import math
import random
import unittest
import numpy as np

from src.features.metrics import BoxScoreStandardizer
from src.features.fatigue import calculate_fatigue_indices
from src.models.bradley_terry import bradley_terry_probability
from src.models.elo import EloTracker, calculate_zero_sum_update


class TestRatingEngine(unittest.TestCase):

    def setUp(self):
        self.standardizer = BoxScoreStandardizer()
        self.tracker = EloTracker(base_elo=1500.0)

    def test_1_balanced_neutral_game(self):
        """Assertion 1: Balanced neutral game (ri=rj=1500, HFA=0, ORTF=TROF) -> P(i) == 0.50."""
        ri = 1500.0
        rj = 1500.0
        ortf = 5.0
        trof = 5.0
        hfa = 0.0

        prob_i, prob_j = bradley_terry_probability(
            ri=ri, rj=rj, ortf=ortf, trof=trof, hfa=hfa, b=99.39, c=993.39, beta_1=0.02
        )

        self.assertAlmostEqual(prob_i, 0.50, places=7)
        self.assertAlmostEqual(prob_j, 0.50, places=7)
        self.assertAlmostEqual(prob_i + prob_j, 1.0, places=7)

    def test_2_blowout_agpf_monotonicity_and_bounds(self):
        """
        Assertion 2: Severe blowout win (NetEff = +45) vs severe blowout loss (NetEff = -45).
        Confirm AGPF_winner > AGPF_loser and AGPF in (0, 1) strictly.
        """
        winner_metrics = {
            "NetEff": 45.0,
            "eFG": 0.65,
            "TS": 0.68,
            "OREB": 15.0,
            "DREB": 32.0,
            "AST:TOV": 2.2,
            "TOV": 8.0,
            "STL": 12.0,
        }

        loser_metrics = {
            "NetEff": -45.0,
            "eFG": 0.28,
            "TS": 0.32,
            "OREB": 6.0,
            "DREB": 18.0,
            "AST:TOV": 0.4,
            "TOV": 24.0,
            "STL": 3.0,
        }

        agpf_winner = self.standardizer.compute_agpf(winner_metrics)
        agpf_loser = self.standardizer.compute_agpf(loser_metrics)

        # Monotonicity: Winner should strictly outperform loser
        self.assertGreater(agpf_winner, agpf_loser)

        # Strict bounds in (0, 1)
        self.assertGreater(agpf_winner, 0.0)
        self.assertLess(agpf_winner, 1.0)
        self.assertGreater(agpf_loser, 0.0)
        self.assertLess(agpf_loser, 1.0)

        # Extreme blowout edge check (NetEff = -120 to +120)
        catastrophic_loss = {"NetEff": -120.0, "eFG": 0.1, "TS": 0.1, "OREB": 0, "DREB": 5, "AST:TOV": 0.1, "TOV": 35, "STL": 0}
        astronomical_win = {"NetEff": 120.0, "eFG": 0.9, "TS": 0.9, "OREB": 25, "DREB": 45, "AST:TOV": 5.0, "TOV": 2, "STL": 20}
        
        agpf_catastrophic = self.standardizer.compute_agpf(catastrophic_loss)
        agpf_astronomical = self.standardizer.compute_agpf(astronomical_win)

        self.assertGreater(agpf_astronomical, agpf_catastrophic)
        self.assertGreater(agpf_catastrophic, 0.0)
        self.assertLess(agpf_catastrophic, 1.0)
        self.assertGreater(agpf_astronomical, 0.0)
        self.assertLess(agpf_astronomical, 1.0)

    def test_3_rating_conservation_across_season(self):
        """
        Assertion 3: Rating conservation check across 350+ teams.
        Confirm total rating sum remains exactly N * 1500 after 1,500 randomized games.
        """
        num_teams = 360
        teams = [f"team_{i:03d}" for i in range(num_teams)]
        tracker = EloTracker(base_elo=1500.0)
        std = BoxScoreStandardizer()

        # Initial sum check
        initial_expected_sum = num_teams * 1500.0
        for t in teams:
            tracker.get_rating(t)
        self.assertAlmostEqual(tracker.total_league_elo(), initial_expected_sum, places=6)

        # Simulate 1,500 games between random teams
        random.seed(42)
        for _ in range(1500):
            t_i, t_j = random.sample(teams, 2)
            ai = 1.0 if random.random() < 0.55 else 0.0

            net_eff = random.uniform(-40.0, 40.0)
            metrics_i = {"NetEff": net_eff, "eFG": 0.45 + net_eff * 0.003, "TS": 0.50, "OREB": 11, "DREB": 25, "AST:TOV": 1.0, "TOV": 15, "STL": 7}
            metrics_j = {"NetEff": -net_eff, "eFG": 0.45 - net_eff * 0.003, "TS": 0.50, "OREB": 11, "DREB": 25, "AST:TOV": 1.0, "TOV": 15, "STL": 7}

            _, _, paf_i = std.compute_paf(t_i, metrics_i)
            _, _, paf_j = std.compute_paf(t_j, metrics_j)
            ortf, trof = calculate_fatigue_indices(
                rd_i=random.uniform(1, 10),
                rd_j=random.uniform(1, 10),
                td_i=random.uniform(0, 2500),
                td_j=random.uniform(0, 2500),
                tzd_i=random.choice([0, 1, 2, 3]),
                tzd_j=random.choice([0, 1, 2, 3]),
            )
            hfa = 34.52 if random.random() < 0.7 else 0.0
            ei, _ = bradley_terry_probability(
                ri=tracker.get_rating(t_i),
                rj=tracker.get_rating(t_j),
                ortf=ortf,
                trof=trof,
                hfa=hfa,
            )
            tracker.update(
                team_i=t_i,
                team_j=t_j,
                ai=ai,
                ei=ei,
                lre=random.uniform(0.1, 2.5),
                paf_i=paf_i,
                paf_j=paf_j,
            )

        # End of season sum must strictly equal initial sum
        final_sum = tracker.total_league_elo()
        self.assertAlmostEqual(final_sum, initial_expected_sum, places=5)
        # Average rating must equal 1500.0
        avg_rating = final_sum / num_teams
        self.assertAlmostEqual(avg_rating, 1500.0, places=5)

    def test_4_fatigue_and_rest_monotonicity(self):
        """
        Assertion 4: Well-rested home team vs fatigued traveling team.
        Assert well-rested team probability > 0.50 and strictly bounded in (0, 1).
        """
        ri = 1500.0
        rj = 1500.0

        # Team i: 8 days rest, 0 travel distance, 0 timezone shift
        # Team j: 1 day rest, 2800 travel distance, 3 hours timezone shift
        ortf, trof = calculate_fatigue_indices(
            rd_i=8.0, rd_j=1.0, td_i=0.0, td_j=2800.0, tzd_i=0.0, tzd_j=3.0, beta_2=1e-6
        )

        # In Bradley-Terry with fatigue, well-rested team i has lower TROF, higher ORTF
        # Expected: prob_i > 0.50 even on neutral floor (HFA=0)
        prob_i, prob_j = bradley_terry_probability(
            ri=ri, rj=rj, ortf=ortf, trof=trof, hfa=0.0, b=99.39, c=993.39, beta_1=0.02
        )

        self.assertGreater(prob_i, 0.50)
        self.assertLess(prob_j, 0.50)
        self.assertGreater(prob_i, 0.0)
        self.assertLess(prob_i, 1.0)
        self.assertAlmostEqual(prob_i + prob_j, 1.0, places=7)

    def test_5_zero_sum_symmetry_with_asymmetric_performance(self):
        """
        Assertion 5: Rating update delta_ri + delta_rj == 0 to numerical precision (< 1e-12)
        even under extreme performance and LRE asymmetry.
        """
        ri = 1620.5
        rj = 1485.2
        ai = 1.0
        ei = 0.684
        lre = 3.45
        paf_i = 1.35
        paf_j = 0.72

        new_ri, new_rj, delta_r = calculate_zero_sum_update(
            ri=ri,
            rj=rj,
            ai=ai,
            ei=ei,
            lre=lre,
            paf_i=paf_i,
            paf_j=paf_j,
            k=32.0,
            beta_3=0.70,
            beta_4=0.02,
        )

        delta_ri = new_ri - ri
        delta_rj = new_rj - rj

        self.assertAlmostEqual(delta_ri + delta_rj, 0.0, delta=1e-12)
        self.assertAlmostEqual(delta_ri, delta_r, places=7)
        self.assertAlmostEqual(delta_rj, -delta_r, places=7)


if __name__ == "__main__":
    unittest.main()

