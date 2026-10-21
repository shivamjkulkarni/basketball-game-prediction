"""
Unit tests for the in-memory dynamic ScheduleTracker and SOS_t calculation.
"""

import unittest
from src.features.schedule import ScheduleTracker


class TestScheduleTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = ScheduleTracker(default_prior_wr=0.50)

    def test_default_prior_win_rate(self):
        """Unseen teams should have default prior win rate of 0.50."""
        self.assertEqual(self.tracker.get_win_rate("TeamA"), 0.50)
        self.assertEqual(self.tracker.get_opponents_faced("TeamA"), set())

    def test_win_rate_updates_accurately(self):
        """Win rates should update strictly with recorded games."""
        self.tracker.record_game(winner="TeamA", loser="TeamB")
        self.tracker.record_game(winner="TeamA", loser="TeamC")
        self.tracker.record_game(winner="TeamD", loser="TeamA")

        # TeamA: 2 wins, 1 loss -> 2/3 win rate
        self.assertAlmostEqual(self.tracker.get_win_rate("TeamA"), 2.0 / 3.0, places=6)
        # TeamB: 0 wins, 1 loss -> 0.0 win rate
        self.assertAlmostEqual(self.tracker.get_win_rate("TeamB"), 0.0, places=6)
        # TeamD: 1 win, 0 loss -> 1.0 win rate
        self.assertAlmostEqual(self.tracker.get_win_rate("TeamD"), 1.0, places=6)

    def test_opponent_tracking_and_avg_opponent_win_rate(self):
        """Verify unique opponents set and average opponent win rate."""
        # TeamA plays TeamB and TeamC
        self.tracker.record_game("TeamA", "TeamB")
        self.tracker.record_game("TeamA", "TeamC")
        # TeamB wins against TeamD
        self.tracker.record_game("TeamB", "TeamD")
        # TeamC loses to TeamE
        self.tracker.record_game("TeamE", "TeamC")

        # Opponents of TeamA: {TeamB, TeamC}
        opps = self.tracker.get_opponents_faced("TeamA")
        self.assertEqual(opps, {"TeamB", "TeamC"})

        # TeamB: 1 win (vs D), 1 loss (vs A) -> WR = 0.5
        # TeamC: 0 wins, 2 losses (vs A, E) -> WR = 0.0
        # Avg opponent WR for TeamA: (0.5 + 0.0) / 2 = 0.25
        avg_opp_wr = self.tracker.get_avg_opponent_win_rate("TeamA")
        self.assertAlmostEqual(avg_opp_wr, 0.25, places=6)

    def test_dynamic_sos_formula(self):
        """
        Verify SOS_t = (2 * WR_opp + AvgWR_opp_opp) / 3.
        """
        # Let Opponent (TeamO) have played against X and Y
        # TeamO beat X, lost to Y -> WR_opp = 0.5
        self.tracker.record_game("TeamO", "TeamX")
        self.tracker.record_game("TeamY", "TeamO")

        # TeamX won 2 of 2 games: beat Z1, Z2
        self.tracker.record_game("TeamX", "TeamZ1")
        self.tracker.record_game("TeamX", "TeamZ2")
        # WR of TeamX = 2 / 3 (wins vs Z1, Z2, lost to O)

        # TeamY lost to Z3
        self.tracker.record_game("TeamZ3", "TeamY")
        # WR of TeamY = 1 / 2 (won vs O, lost to Z3)

        wr_x = 2.0 / 3.0
        wr_y = 1.0 / 2.0
        expected_avg_opp_opp = (wr_x + wr_y) / 2.0  # (2/3 + 1/2)/2 = 7/12 = ~0.58333
        expected_wr_opp = 1.0 / 2.0  # 0.5
        expected_sos = (2.0 * expected_wr_opp + expected_avg_opp_opp) / 3.0

        computed_sos = self.tracker.compute_sos("TeamA", "TeamO")
        self.assertAlmostEqual(computed_sos, expected_sos, places=6)

    def test_chronological_isolation_prevents_lookahead(self):
        """Ensure SOS_t queried prior to game completion does not reflect game outcome."""
        # Initial query before any game
        sos_initial = self.tracker.compute_sos("TeamA", "TeamB")
        self.assertEqual(sos_initial, 0.50)

        # Game played: TeamB beats TeamC
        self.tracker.record_game("TeamB", "TeamC")

        # Subsequent query now reflects TeamB's win
        sos_after_b_win = self.tracker.compute_sos("TeamA", "TeamB")
        self.assertGreater(sos_after_b_win, sos_initial)


if __name__ == "__main__":
    unittest.main()

