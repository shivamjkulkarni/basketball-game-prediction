"""
High-Throughput Chronological Objective & Loss Evaluator for Optuna.
Executes 5,219 games in ~50ms per trial with walk-forward 70/30 train/val split.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from src.config import BASE_ELO, DEFAULT_METRIC_STATS, METRIC_KEYS
from src.data.loader import load_game_logs
from src.optimization.sampler import HyperparameterSampler


class ChronologicalObjective:
    """
    Evaluates out-of-sample Log-Loss, Brier Score, and Binary Accuracy
    using a strict chronological 70/30 walk-forward partition:
    - Train (70%): First 3,653 games to evolve team ratings from 1500.00
    - Val (30%): Remaining 1,566 games evaluated out-of-sample prior to updates
    """

    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.70,
        pruning_interval: int = 250,
    ):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.pruning_interval = pruning_interval
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """Pre-extract all 5,219 game logs into contiguous NumPy vectors."""
        print(f"[Objective] Ingesting and caching game logs from: {self.data_path}")
        clean_df = load_game_logs(self.data_path)

        # Compute metric means and standard deviations across entire dataset
        metric_cols = {
            "NetEff": "neteff",
            "eFG": "efg",
            "TS": "ts",
            "OREB": "oreb",
            "DREB": "dreb",
            "AST:TOV": "ast:tov",
            "TOV": "tov",
            "STL": "stl",
        }

        self.stats = {}
        for key in METRIC_KEYS:
            col = metric_cols.get(key, key.lower())
            if col in clean_df.columns:
                series = pd.to_numeric(clean_df[col], errors="coerce").dropna()
                self.stats[key] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()) if float(series.std()) > 1e-6 else 1.0,
                }
            else:
                self.stats[key] = DEFAULT_METRIC_STATS.get(key, {"mean": 0.0, "std": 1.0})

        # Map all unique teams to integer IDs
        all_teams = sorted(list(clean_df["team"].astype(str).str.strip().unique()))
        self.team_to_id = {t: idx for idx, t in enumerate(all_teams)}
        self.id_to_team = {idx: t for idx, t in enumerate(all_teams)}
        self.num_teams = len(all_teams)

        # Pair up games
        unique_game_ids = clean_df["game_id"].unique()
        self.num_total_games = len(unique_game_ids)
        self.num_train_games = int(round(self.num_total_games * self.train_ratio))
        self.num_val_games = self.num_total_games - self.num_train_games

        print(
            f"[Objective] Cached {self.num_total_games} games across {self.num_teams} teams: "
            f"{self.num_train_games} train (70%), {self.num_val_games} val (30%)."
        )

        # Allocate arrays
        self.t_i = np.zeros(self.num_total_games, dtype=np.int32)
        self.t_j = np.zeros(self.num_total_games, dtype=np.int32)
        self.y = np.zeros(self.num_total_games, dtype=np.float64)
        self.ha_sign = np.zeros(self.num_total_games, dtype=np.float64)
        self.rd_i = np.zeros(self.num_total_games, dtype=np.float64)
        self.rd_j = np.zeros(self.num_total_games, dtype=np.float64)
        self.td_i = np.zeros(self.num_total_games, dtype=np.float64)
        self.td_j = np.zeros(self.num_total_games, dtype=np.float64)
        self.tzd_i = np.zeros(self.num_total_games, dtype=np.float64)
        self.tzd_j = np.zeros(self.num_total_games, dtype=np.float64)
        self.lre = np.zeros(self.num_total_games, dtype=np.float64)
        self.z_i = np.zeros((self.num_total_games, len(METRIC_KEYS)), dtype=np.float64)
        self.z_j = np.zeros((self.num_total_games, len(METRIC_KEYS)), dtype=np.float64)

        def _get_val(row, col, default=0.0):
            val = row.get(col, default)
            try:
                if val is None or pd.isna(val):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        for g_idx, g_id in enumerate(unique_game_ids):
            rows = clean_df[clean_df["game_id"] == g_id]
            if len(rows) != 2:
                continue
            r_i = rows.iloc[0]
            r_j = rows.iloc[1]

            team_name_i = str(r_i["team"]).strip()
            team_name_j = str(r_j["team"]).strip()

            self.t_i[g_idx] = self.team_to_id[team_name_i]
            self.t_j[g_idx] = self.team_to_id[team_name_j]

            res_i = str(r_i.get("w/l", "")).strip().upper()
            self.y[g_idx] = 1.0 if res_i == "W" else 0.0

            ha_i = _get_val(r_i, "home_away", 0.0)
            self.ha_sign[g_idx] = 1.0 if ha_i == 0.0 else (-1.0 if ha_i == 1.0 else 0.0)

            self.rd_i[g_idx] = _get_val(r_i, "rest_days", _get_val(r_j, "oprd", 2.0))
            self.rd_j[g_idx] = _get_val(r_j, "rest_days", _get_val(r_i, "oprd", 2.0))
            self.td_i[g_idx] = _get_val(r_i, "travel_distance", _get_val(r_j, "optd", 0.0))
            self.td_j[g_idx] = _get_val(r_j, "travel_distance", _get_val(r_i, "optd", 0.0))
            self.tzd_i[g_idx] = _get_val(r_i, "tz_dif_h_e", 0.0)
            self.tzd_j[g_idx] = _get_val(r_j, "tz_dif_h_e", 0.0)
            self.lre[g_idx] = _get_val(r_i, "lre", _get_val(r_j, "lre", 0.0))

            for m_idx, m_key in enumerate(METRIC_KEYS):
                col_name = metric_cols.get(m_key, m_key.lower())
                val_i = _get_val(r_i, col_name, self.stats[m_key]["mean"])
                val_j = _get_val(r_j, col_name, self.stats[m_key]["mean"])
                self.z_i[g_idx, m_idx] = (val_i - self.stats[m_key]["mean"]) / self.stats[m_key]["std"]
                self.z_j[g_idx, m_idx] = (val_j - self.stats[m_key]["mean"]) / self.stats[m_key]["std"]

    def evaluate(
        self,
        params: Dict[str, Any],
        trial: Optional[optuna.Trial] = None,
    ) -> Tuple[float, float, float]:
        """
        Execute full season walk-forward simulation with candidate parameter set.

        Returns:
            Tuple[float, float, float]: (out_of_sample_log_loss, brier_score, accuracy)
        """
        k = float(params["k"])
        c = float(params["c"])
        b = float(params["b"])
        hfa = float(params["hfa"])
        beta_1 = float(params["beta_1"])
        beta_2 = float(params["beta_2"])
        beta_3 = float(params["beta_3"])
        beta_4 = float(params["beta_4"])
        k_scale = float(params["k_scale"])
        alpha = np.array(params["alpha"], dtype=np.float64)

        # In-memory tracking state
        ratings = np.full(self.num_teams, BASE_ELO, dtype=np.float64)
        agpf_sums = np.zeros(self.num_teams, dtype=np.float64)
        agpf_counts = np.zeros(self.num_teams, dtype=np.int32)

        val_losses = []
        val_briers = []
        val_correct = 0

        # Run through all chronological games
        for g in range(self.num_total_games):
            ti = self.t_i[g]
            tj = self.t_j[g]
            ri = ratings[ti]
            rj = ratings[tj]

            # Fatigue calculation
            ortf = self.rd_j[g] + math.exp(abs(self.tzd_i[g]) / 3.0) + beta_2 * self.td_i[g]
            trof = self.rd_i[g] + math.exp(abs(self.tzd_j[g]) / 3.0) + beta_2 * self.td_j[g]

            # Bradley-Terry win probability
            hfa_pts = self.ha_sign[g] * hfa
            exponent = ((rj - (ri + hfa_pts)) / c) + beta_1 * math.log(max(ortf, 1e-5)) - beta_1 * math.log(max(trof, 1e-5))
            exponent_clamped = max(min(exponent, 50.0), -50.0)
            prob_i = 1.0 / (1.0 + (b ** exponent_clamped))

            y_actual = self.y[g]

            # Out-of-sample evaluation on validation partition (games >= 3653)
            if g >= self.num_train_games:
                p_clamped = max(min(prob_i, 1.0 - 1e-15), 1e-15)
                loss_g = -(y_actual * math.log(p_clamped) + (1.0 - y_actual) * math.log(1.0 - p_clamped))
                brier_g = (p_clamped - y_actual) ** 2
                is_correct = 1 if ((prob_i >= 0.5) == (y_actual == 1.0)) else 0

                val_losses.append(loss_g)
                val_briers.append(brier_g)
                val_correct += is_correct

                # Intermediate pruning check
                val_idx = g - self.num_train_games
                if trial is not None and (val_idx + 1) % self.pruning_interval == 0:
                    running_loss = float(np.mean(val_losses))
                    trial.report(running_loss, step=val_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # AGPF computation
            s_i = float(np.dot(alpha, self.z_i[g])) * k_scale
            s_j = float(np.dot(alpha, self.z_j[g])) * k_scale
            agpf_i = 1.0 / (1.0 + math.exp(-max(min(s_i, 50.0), -50.0)))
            agpf_j = 1.0 / (1.0 + math.exp(-max(min(s_j, 50.0), -50.0)))

            ci = agpf_counts[ti]
            cj = agpf_counts[tj]
            a_gpf_i = (agpf_sums[ti] / ci) if ci > 0 else agpf_i
            a_gpf_j = (agpf_sums[tj] / cj) if cj > 0 else agpf_j

            agpf_sums[ti] += agpf_i
            agpf_counts[ti] += 1
            agpf_sums[tj] += agpf_j
            agpf_counts[tj] += 1

            paf_i = (agpf_i / a_gpf_i) if a_gpf_i > 0 else 1.0
            paf_j = (agpf_j / a_gpf_j) if a_gpf_j > 0 else 1.0

            # Zero-sum symmetric Elo update
            m = k * (1.0 + beta_3 * math.log2(1.0 + self.lre[g])) * (1.0 + beta_4 * abs(paf_i - paf_j))
            delta = m * (y_actual - prob_i)
            ratings[ti] += delta
            ratings[tj] -= delta

        final_loss = float(np.mean(val_losses))
        final_brier = float(np.mean(val_briers))
        final_acc = float(val_correct) / float(len(val_losses))

        return final_loss, final_brier, final_acc

    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective callable."""
        sampled = HyperparameterSampler.sample(trial)
        loss, brier, acc = self.evaluate(sampled, trial=trial)
        trial.set_user_attr("brier_score", brier)
        trial.set_user_attr("accuracy", acc)
        return loss

