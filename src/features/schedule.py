"""
Dynamic In-Memory Schedule Graph & Strength of Schedule (SOS_t) Engine.
Enforces strict chronological isolation to prevent lookahead bias.
"""

from typing import Dict, List, Optional, Set, Tuple


class ScheduleTracker:
    """
    Stateful in-memory tracker of completed collegiate matchups.
    Computes dynamic game-wise Strength of Schedule (SOS_t) strictly on the fly:
        SOS_t = (2 * WR_opp + AvgWR_opp_opp) / 3
    without leaking games occurring on or after timestamp t.
    """

    def __init__(self, default_prior_wr: float = 0.50):
        self.default_prior_wr = float(default_prior_wr)
        # Mapping: team -> total wins prior to current timestamp
        self.team_wins: Dict[str, int] = {}
        # Mapping: team -> total games played prior to current timestamp
        self.team_games: Dict[str, int] = {}
        # Mapping: team -> set of unique opponents faced prior to current timestamp
        self.team_opponents: Dict[str, Set[str]] = {}

    def get_win_rate(self, team: str) -> float:
        """
        Compute historical win rate of a team strictly prior to current evaluation:
        WR = sum(W_g) / sum(G_g)
        """
        games = self.team_games.get(team, 0)
        if games == 0:
            return self.default_prior_wr
        wins = self.team_wins.get(team, 0)
        return float(wins) / float(games)

    def get_opponents_faced(self, team: str) -> Set[str]:
        """Return the set of unique opponents faced by the team prior to current evaluation."""
        return set(self.team_opponents.get(team, set()))

    def get_avg_opponent_win_rate(self, team: str) -> float:
        """
        Compute the average win rate of all unique opponents faced by the team prior to t:
        AvgWR_opp_opp = (1 / M) * sum_{j=1}^M WR_j(t)
        """
        opponents = self.get_opponents_faced(team)
        if not opponents:
            return self.get_win_rate(team)
        opp_win_rates = [self.get_win_rate(opp) for opp in opponents]
        return float(sum(opp_win_rates)) / float(len(opp_win_rates))

    def compute_sos(self, team: str, opponent: str) -> float:
        """
        Compute dynamic strength of schedule for a matchup against an opponent:
            SOS_t = (2 * WR_opp + AvgWR_opp_opp) / 3

        Args:
            team (str): Evaluated team identifier.
            opponent (str): Opponent identifier.

        Returns:
            float: Dynamic schedule strength in [0, 1].
        """
        wr_opp = self.get_win_rate(opponent)
        avg_wr_opp_opp = self.get_avg_opponent_win_rate(opponent)
        sos_t = (2.0 * wr_opp + avg_wr_opp_opp) / 3.0
        return max(min(sos_t, 1.0), 0.0)

    def record_game(self, winner: str, loser: str) -> None:
        """
        Record the completion of a game, updating historical wins, games, and opponents.
        Must be invoked strictly AFTER querying SOS_t for the game.

        Args:
            winner (str): Winning team identifier.
            loser (str): Losing team identifier.
        """
        # Update games count
        self.team_games[winner] = self.team_games.get(winner, 0) + 1
        self.team_games[loser] = self.team_games.get(loser, 0) + 1

        # Update wins count
        self.team_wins[winner] = self.team_wins.get(winner, 0) + 1
        self.team_wins.setdefault(loser, self.team_wins.get(loser, 0))

        # Update opponent sets
        self.team_opponents.setdefault(winner, set()).add(loser)
        self.team_opponents.setdefault(loser, set()).add(winner)

    def get_team_summary(self, team: str) -> Dict[str, float]:
        """Return diagnostic summary for a team."""
        return {
            "wins": self.team_wins.get(team, 0),
            "games": self.team_games.get(team, 0),
            "win_rate": self.get_win_rate(team),
            "opponents_count": len(self.get_opponents_faced(team)),
        }

