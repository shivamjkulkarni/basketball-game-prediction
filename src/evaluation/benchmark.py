"""
Model Benchmarking, Probability Calibration & Baseline Comparative Analysis Engine.
Evaluates the optimized Bradley-Terry / WHR dynamic model against 4 competitive baselines:
1. Standard Zero-Sum Vanilla Elo
2. Efficiency Margin Regression (KenPom / Torvik Style)
3. Static Bradley-Terry MLE
4. Gradient Boosted Box-Score Classifier (LightGBM)
"""

from collections import deque
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.config import BASE_ELO
from src.data.loader import load_game_logs
from src.models.baselines import (
    EfficiencyMarginRegression,
    LightGBMBoxScoreClassifier,
    StaticBradleyTerryMLE,
    VanillaElo,
)
from src.optimization.sampler import HyperparameterSampler


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) across equal-width probability bins:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


class BenchmarkSuite:
    """
    Executes chronological walk-forward benchmark across 1,566 out-of-sample validation games.
    """

    def __init__(
        self,
        data_path: str = "data/final_final_data.csv",
        optimized_params_path: str = "output_data/optimized_parameters.json",
        train_ratio: float = 0.70,
    ):
        self.data_path = data_path
        self.optimized_params_path = optimized_params_path
        self.train_ratio = train_ratio
        self._load_optimized_parameters()
        self._prepare_data()

    def _load_optimized_parameters(self) -> None:
        """Load optimized parameters from study JSON or fallback to defaults."""
        if os.path.exists(self.optimized_params_path):
            with open(self.optimized_params_path, "r") as f:
                data = json.load(f)
                self.opt_params = data.get("optimized_parameters", {})
        else:
            # High-performance defaults from optimization
            self.opt_params = {
                "k": 38,
                "c": 484.07,
                "b": 115.60,
                "hfa": 34.52,
                "beta_1": 0.0022,
                "beta_2": 4.34e-08,
                "beta_3": 0.22,
                "beta_4": 0.0031,
                "k_scale": 1.71,
                "alpha": [0.0792, 0.1635, 0.2517, 0.0199, 0.2558, 0.0080, 0.0813, 0.1405],
            }

    def _prepare_data(self) -> None:
        """Ingest, clean, and pair all chronological game logs."""
        print(f"[Benchmark] Ingesting match logs from: {self.data_path}")
        clean_df = load_game_logs(self.data_path)
        unique_game_ids = clean_df["game_id"].unique()
        self.num_total = len(unique_game_ids)
        self.num_train = int(round(self.num_total * self.train_ratio))
        self.num_val = self.num_total - self.num_train

        self.games = []
        all_teams_set = set()

        def _val(row, col, default=0.0):
            v = row.get(col, default)
            try:
                if v is None or pd.isna(v):
                    return default
                return float(v)
            except (ValueError, TypeError):
                return default

        for g_id in unique_game_ids:
            rows = clean_df[clean_df["game_id"] == g_id]
            if len(rows) != 2:
                continue
            r_i = rows.iloc[0]
            r_j = rows.iloc[1]

            t_i = str(r_i["team"]).strip()
            t_j = str(r_j["team"]).strip()
            all_teams_set.add(t_i)
            all_teams_set.add(t_j)

            res_i = str(r_i.get("w/l", "")).strip().upper()
            y_i = 1.0 if res_i == "W" else 0.0

            ha_i = _val(r_i, "home_away", 0.0)
            ha_sign = 1.0 if ha_i == 0.0 else (-1.0 if ha_i == 1.0 else 0.0)

            rd_i = _val(r_i, "rest_days", _val(r_j, "oprd", 2.0))
            rd_j = _val(r_j, "rest_days", _val(r_i, "oprd", 2.0))
            td_i = _val(r_i, "travel_distance", _val(r_j, "optd", 0.0))
            td_j = _val(r_j, "travel_distance", _val(r_i, "optd", 0.0))
            tzd_i = _val(r_i, "tz_dif_h_e", 0.0)
            tzd_j = _val(r_j, "tz_dif_h_e", 0.0)
            lre = _val(r_i, "lre", _val(r_j, "lre", 0.0))

            box_i = {
                "NetEff": _val(r_i, "neteff"),
                "eFG": _val(r_i, "efg"),
                "TS": _val(r_i, "ts"),
                "OREB": _val(r_i, "oreb"),
                "DREB": _val(r_i, "dreb"),
                "AST:TOV": _val(r_i, "ast:tov"),
                "TOV": _val(r_i, "tov"),
                "STL": _val(r_i, "stl"),
            }
            box_j = {
                "NetEff": _val(r_j, "neteff"),
                "eFG": _val(r_j, "efg"),
                "TS": _val(r_j, "ts"),
                "OREB": _val(r_j, "oreb"),
                "DREB": _val(r_j, "dreb"),
                "AST:TOV": _val(r_j, "ast:tov"),
                "TOV": _val(r_j, "tov"),
                "STL": _val(r_j, "stl"),
            }

            self.games.append({
                "game_id": g_id,
                "game_date": str(r_i.get("game_date", "")),
                "team_i": t_i,
                "team_j": t_j,
                "y": y_i,
                "ha_sign": ha_sign,
                "rd_i": rd_i,
                "rd_j": rd_j,
                "td_i": td_i,
                "td_j": td_j,
                "tzd_i": tzd_i,
                "tzd_j": tzd_j,
                "lre": lre,
                "box_i": box_i,
                "box_j": box_j,
            })

        self.all_teams = sorted(list(all_teams_set))
        print(f"[Benchmark] Ingested {len(self.games)} matchups ({self.num_train} train, {self.num_val} val).")

    def run_benchmark(
        self,
        output_dir: str = "output_data",
        figures_dir: str = "output_figures",
    ) -> pd.DataFrame:
        """Execute chronological evaluation across all 5 models."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        print("\n" + "=" * 75)
        print("EXECUTING MULTI-MODEL COMPETITIVE BENCHMARK (1,566 VALIDATION MATCHES)")
        print("=" * 75)

        # 1. Instantiate models
        vanilla_elo = VanillaElo(k=32.0, hfa=34.52, c=400.0)
        eff_regression = EfficiencyMarginRegression()
        static_bt = StaticBradleyTerryMLE(c_reg=1.0)
        lgbm_classifier = LightGBMBoxScoreClassifier(window_size=10, random_state=42)

        # 2. Pre-train supervised models on the 70% training window
        print("1/4. Fitting Static Bradley-Terry MLE on training partition...")
        train_matchups_bt = [
            (g["team_i"], g["team_j"], g["ha_sign"], g["y"])
            for g in self.games[:self.num_train]
        ]
        static_bt.fit(train_matchups_bt, self.all_teams)

        print("2/4. Fitting Efficiency Margin Regression & LightGBM on rolling histories...")
        # Maintain rolling histories during training window
        team_rolling_stats = {
            t: {
                "NetEff": deque(maxlen=10),
                "eFG": deque(maxlen=10),
                "TS": deque(maxlen=10),
                "TOV": deque(maxlen=10),
                "DREB": deque(maxlen=10),
                "OREB": deque(maxlen=10),
            }
            for t in self.all_teams
        }

        X_train_eff = []
        X_train_lgb = []
        y_train = []

        for g in self.games[:self.num_train]:
            ti = g["team_i"]
            tj = g["team_j"]

            # Pre-game rolling averages (strictly prior to this game)
            def _get_avg(team, key, default=0.0):
                d = team_rolling_stats[team][key]
                return float(np.mean(d)) if len(d) > 0 else default

            net_i = _get_avg(ti, "NetEff")
            net_j = _get_avg(tj, "NetEff")
            X_train_eff.append([net_i - net_j, g["ha_sign"]])

            # LightGBM feature vector
            feat_lgb = [
                net_i - net_j,
                _get_avg(ti, "eFG", 0.50) - _get_avg(tj, "eFG", 0.50),
                _get_avg(ti, "TS", 0.50) - _get_avg(tj, "TS", 0.50),
                _get_avg(ti, "TOV", 15.0) - _get_avg(tj, "TOV", 15.0),
                _get_avg(ti, "DREB", 25.0) - _get_avg(tj, "DREB", 25.0),
                _get_avg(ti, "OREB", 10.0) - _get_avg(tj, "OREB", 10.0),
                g["rd_i"] - g["rd_j"],
                g["td_i"] - g["td_j"],
                g["tzd_i"] - g["tzd_j"],
                g["ha_sign"],
            ]
            X_train_lgb.append(feat_lgb)
            y_train.append(g["y"])

            # Update rolling stats AFTER extracting pre-game features
            for k_m in ["NetEff", "eFG", "TS", "TOV", "DREB", "OREB"]:
                team_rolling_stats[ti][k_m].append(g["box_i"][k_m])
                team_rolling_stats[tj][k_m].append(g["box_j"][k_m])

            # Update Vanilla Elo on training games
            p_vanilla = vanilla_elo.predict_proba(ti, tj, g["ha_sign"])
            vanilla_elo.update(ti, tj, g["y"], p_vanilla)

        # Fit supervised models
        eff_regression.fit(np.array(X_train_eff), np.array(y_train))
        lgbm_classifier.fit(np.array(X_train_lgb), np.array(y_train))

        # 3. Setup Dynamic Optimized Model tracking state
        opt_k = float(self.opt_params["k"])
        opt_c = float(self.opt_params["c"])
        opt_b = float(self.opt_params["b"])
        opt_hfa = float(self.opt_params["hfa"])
        opt_beta_1 = float(self.opt_params["beta_1"])
        opt_beta_2 = float(self.opt_params["beta_2"])
        opt_beta_3 = float(self.opt_params["beta_3"])
        opt_beta_4 = float(self.opt_params["beta_4"])
        opt_k_scale = float(self.opt_params["k_scale"])
        opt_alpha = np.array(self.opt_params["alpha"], dtype=np.float64)

        opt_ratings = {t: BASE_ELO for t in self.all_teams}
        opt_agpf_sums = {t: 0.0 for t in self.all_teams}
        opt_agpf_counts = {t: 0 for t in self.all_teams}

        # Calculate dataset metric statistics for Z-scoring
        metric_means = {}
        metric_stds = {}
        all_box_records = [g["box_i"] for g in self.games] + [g["box_j"] for g in self.games]
        box_df = pd.DataFrame(all_box_records)
        for col in ["NetEff", "eFG", "TS", "OREB", "DREB", "AST:TOV", "TOV", "STL"]:
            metric_means[col] = float(box_df[col].mean())
            metric_stds[col] = float(box_df[col].std()) if float(box_df[col].std()) > 1e-6 else 1.0

        def _get_z(box_dict):
            z = []
            for col in ["NetEff", "eFG", "TS", "OREB", "DREB", "AST:TOV", "TOV", "STL"]:
                z.append((box_dict[col] - metric_means[col]) / metric_stds[col])
            return np.array(z, dtype=np.float64)

        # Run Dynamic Optimized Model through training window
        for g in self.games[:self.num_train]:
            ti = g["team_i"]
            tj = g["team_j"]
            ri = opt_ratings[ti]
            rj = opt_ratings[tj]

            ortf = g["rd_j"] + math.exp(abs(g["tzd_i"]) / 3.0) + opt_beta_2 * g["td_i"]
            trof = g["rd_i"] + math.exp(abs(g["tzd_j"]) / 3.0) + opt_beta_2 * g["td_j"]
            exp_term = ((rj - (ri + g["ha_sign"] * opt_hfa)) / opt_c) + opt_beta_1 * math.log(max(ortf, 1e-5)) - opt_beta_1 * math.log(max(trof, 1e-5))
            prob_i = 1.0 / (1.0 + (opt_b ** max(min(exp_term, 50.0), -50.0)))

            # AGPF
            zi = _get_z(g["box_i"])
            zj = _get_z(g["box_j"])
            si = float(np.dot(opt_alpha, zi)) * opt_k_scale
            sj = float(np.dot(opt_alpha, zj)) * opt_k_scale
            agpf_i = 1.0 / (1.0 + math.exp(-max(min(si, 50.0), -50.0)))
            agpf_j = 1.0 / (1.0 + math.exp(-max(min(sj, 50.0), -50.0)))

            ci = opt_agpf_counts[ti]
            cj = opt_agpf_counts[tj]
            a_gpf_i = (opt_agpf_sums[ti] / ci) if ci > 0 else agpf_i
            a_gpf_j = (opt_agpf_sums[tj] / cj) if cj > 0 else agpf_j

            opt_agpf_sums[ti] += agpf_i
            opt_agpf_counts[ti] += 1
            opt_agpf_sums[tj] += agpf_j
            opt_agpf_counts[tj] += 1

            paf_i = (agpf_i / a_gpf_i) if a_gpf_i > 0 else 1.0
            paf_j = (agpf_j / a_gpf_j) if a_gpf_j > 0 else 1.0

            m = opt_k * (1.0 + opt_beta_3 * math.log2(1.0 + g["lre"])) * (1.0 + opt_beta_4 * abs(paf_i - paf_j))
            delta = m * (g["y"] - prob_i)
            opt_ratings[ti] += delta
            opt_ratings[tj] -= delta

        # 4. Out-of-Sample Validation Loop across remaining 1,566 games
        print("3/4. Evaluating all 5 models out-of-sample on validation matches...")
        val_y = []
        preds = {
            "Dynamic Optimized WHR": [],
            "Vanilla Zero-Sum Elo": [],
            "Efficiency Margin Regression": [],
            "Static Bradley-Terry MLE": [],
            "LightGBM Classifier": [],
        }

        timeline_records = []

        for g in self.games[self.num_train:]:
            ti = g["team_i"]
            tj = g["team_j"]
            y_actual = g["y"]
            val_y.append(y_actual)

            # Model 1: Dynamic Optimized Model
            ri = opt_ratings[ti]
            rj = opt_ratings[tj]
            ortf = g["rd_j"] + math.exp(abs(g["tzd_i"]) / 3.0) + opt_beta_2 * g["td_i"]
            trof = g["rd_i"] + math.exp(abs(g["tzd_j"]) / 3.0) + opt_beta_2 * g["td_j"]
            exp_term = ((rj - (ri + g["ha_sign"] * opt_hfa)) / opt_c) + opt_beta_1 * math.log(max(ortf, 1e-5)) - opt_beta_1 * math.log(max(trof, 1e-5))
            p_opt = 1.0 / (1.0 + (opt_b ** max(min(exp_term, 50.0), -50.0)))
            preds["Dynamic Optimized WHR"].append(max(min(p_opt, 1.0 - 1e-15), 1e-15))

            # Model 2: Vanilla Elo
            p_vanilla = vanilla_elo.predict_proba(ti, tj, g["ha_sign"])
            preds["Vanilla Zero-Sum Elo"].append(max(min(p_vanilla, 1.0 - 1e-15), 1e-15))

            # Model 3: Efficiency Margin
            net_i = _get_avg(ti, "NetEff")
            net_j = _get_avg(tj, "NetEff")
            p_eff = eff_regression.predict_proba(net_i, net_j, g["ha_sign"])
            preds["Efficiency Margin Regression"].append(max(min(p_eff, 1.0 - 1e-15), 1e-15))

            # Model 4: Static Bradley-Terry MLE
            p_bt = static_bt.predict_proba(ti, tj, g["ha_sign"])
            preds["Static Bradley-Terry MLE"].append(max(min(p_bt, 1.0 - 1e-15), 1e-15))

            # Model 5: LightGBM
            feat_val_lgb = np.array([
                net_i - net_j,
                _get_avg(ti, "eFG", 0.50) - _get_avg(tj, "eFG", 0.50),
                _get_avg(ti, "TS", 0.50) - _get_avg(tj, "TS", 0.50),
                _get_avg(ti, "TOV", 15.0) - _get_avg(tj, "TOV", 15.0),
                _get_avg(ti, "DREB", 25.0) - _get_avg(tj, "DREB", 25.0),
                _get_avg(ti, "OREB", 10.0) - _get_avg(tj, "OREB", 10.0),
                g["rd_i"] - g["rd_j"],
                g["td_i"] - g["td_j"],
                g["tzd_i"] - g["tzd_j"],
                g["ha_sign"],
            ], dtype=np.float64)
            p_lgb = lgbm_classifier.predict_proba(feat_val_lgb)
            preds["LightGBM Classifier"].append(p_lgb)

            # Record timeline entry
            timeline_records.append({
                "game_date": g["game_date"],
                "y": y_actual,
                "p_opt": p_opt,
                "p_vanilla": p_vanilla,
                "p_eff": p_eff,
                "p_bt": p_bt,
                "p_lgb": p_lgb,
            })

            # Update dynamic models AFTER predicting
            # Vanilla Elo update
            vanilla_elo.update(ti, tj, y_actual, p_vanilla)

            # Dynamic Optimized update
            zi = _get_z(g["box_i"])
            zj = _get_z(g["box_j"])
            si = float(np.dot(opt_alpha, zi)) * opt_k_scale
            sj = float(np.dot(opt_alpha, zj)) * opt_k_scale
            agpf_i = 1.0 / (1.0 + math.exp(-max(min(si, 50.0), -50.0)))
            agpf_j = 1.0 / (1.0 + math.exp(-max(min(sj, 50.0), -50.0)))

            ci = opt_agpf_counts[ti]
            cj = opt_agpf_counts[tj]
            a_gpf_i = (opt_agpf_sums[ti] / ci) if ci > 0 else agpf_i
            a_gpf_j = (opt_agpf_sums[tj] / cj) if cj > 0 else agpf_j

            opt_agpf_sums[ti] += agpf_i
            opt_agpf_counts[ti] += 1
            opt_agpf_sums[tj] += agpf_j
            opt_agpf_counts[tj] += 1

            paf_i = (agpf_i / a_gpf_i) if a_gpf_i > 0 else 1.0
            paf_j = (agpf_j / a_gpf_j) if a_gpf_j > 0 else 1.0

            m = opt_k * (1.0 + opt_beta_3 * math.log2(1.0 + g["lre"])) * (1.0 + opt_beta_4 * abs(paf_i - paf_j))
            delta = m * (y_actual - p_opt)
            opt_ratings[ti] += delta
            opt_ratings[tj] -= delta

            # Rolling stats update for eff and lgbm
            for k_m in ["NetEff", "eFG", "TS", "TOV", "DREB", "OREB"]:
                team_rolling_stats[ti][k_m].append(g["box_i"][k_m])
                team_rolling_stats[tj][k_m].append(g["box_j"][k_m])

        val_y = np.array(val_y, dtype=np.float64)

        # 5. Compute Quantitative Metrics
        print("4/4. Computing diagnostic evaluation metrics...")
        vanilla_brier = float(np.mean((np.array(preds["Vanilla Zero-Sum Elo"]) - val_y) ** 2))

        metrics_list = []
        for model_name, p_list in preds.items():
            p_arr = np.array(p_list, dtype=np.float64)

            ll = float(log_loss(val_y, p_arr))
            bs = float(np.mean((p_arr - val_y) ** 2))
            ece = compute_expected_calibration_error(val_y, p_arr, n_bins=10)
            roc_auc = float(roc_auc_score(val_y, p_arr))
            pr_auc = float(average_precision_score(val_y, p_arr))
            acc = float(np.mean((p_arr >= 0.5) == val_y))
            bss = 1.0 - (bs / vanilla_brier) if vanilla_brier > 0 else 0.0

            metrics_list.append({
                "Model": model_name,
                "Log-Loss": round(ll, 5),
                "Brier Score": round(bs, 5),
                "ECE": round(ece, 5),
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Accuracy": round(acc * 100.0, 2),
                "BSS vs Vanilla Elo (%)": round(bss * 100.0, 2),
            })

        comparison_df = pd.DataFrame(metrics_list).sort_values(by="Log-Loss", ascending=True).reset_index(drop=True)

        # Export CSV table
        csv_out = os.path.join(output_dir, "model_benchmark_comparison.csv")
        comparison_df.to_csv(csv_out, index=False)
        print(f"Saved benchmark comparison to: {csv_out}")

        # Generate Visualizations
        self._plot_calibration_curve(val_y, preds, figures_dir)
        self._plot_brier_skill_scores(comparison_df, figures_dir)
        self._plot_cumulative_logloss(timeline_records, figures_dir)

        # Print Console Table
        print("\n" + "=" * 95)
        print("OUT-OF-SAMPLE BENCHMARK RESULTS (1,566 VALIDATION MATCHES)")
        print("=" * 95)
        print(comparison_df.to_string(index=False))
        print("=" * 95 + "\n")

        return comparison_df

    def _plot_calibration_curve(
        self,
        val_y: np.ndarray,
        preds: Dict[str, List[float]],
        figures_dir: str,
    ) -> None:
        """Plot publication-quality calibration reliability diagrams for all 5 models."""
        plt.figure(figsize=(9, 7))
        colors = {
            "Dynamic Optimized WHR": "#1f77b4",
            "Vanilla Zero-Sum Elo": "#ff7f0e",
            "Efficiency Margin Regression": "#2ca02c",
            "Static Bradley-Terry MLE": "#d62728",
            "LightGBM Classifier": "#9467bd",
        }

        # Perfect calibration line
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

        for name, p_list in preds.items():
            prob_true, prob_pred = calibration_curve(val_y, p_list, n_bins=10, strategy="uniform")
            plt.plot(
                prob_pred,
                prob_true,
                marker="o",
                linewidth=2,
                color=colors.get(name, "gray"),
                label=name,
            )

        plt.xlabel("Mean Predicted Win Probability", fontsize=12)
        plt.ylabel("Observed Empirical Win Fraction", fontsize=12)
        plt.title("Reliability Diagram: Out-of-Sample Calibration (1,566 Games)", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", frameon=True, fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(figures_dir, "calibration_curve.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_path}")

    def _plot_brier_skill_scores(
        self,
        comparison_df: pd.DataFrame,
        figures_dir: str,
    ) -> None:
        """Plot horizontal bar chart of Brier Skill Scores referenced to Vanilla Elo."""
        plt.figure(figsize=(8, 5))
        plot_df = comparison_df.sort_values(by="BSS vs Vanilla Elo (%)", ascending=True)

        bars = plt.barh(
            plot_df["Model"],
            plot_df["BSS vs Vanilla Elo (%)"],
            color=["#d62728" if v < 0 else "#2ca02c" for v in plot_df["BSS vs Vanilla Elo (%)"]],
            edgecolor="black",
            linewidth=0.8,
        )

        plt.axvline(0, color="black", linestyle="--", linewidth=1.2)
        plt.xlabel("Brier Skill Score vs Vanilla Elo Reference (%)", fontsize=11, fontweight="bold")
        plt.title("Brier Skill Score (BSS) Comparison: Out-of-Sample Edge", fontsize=13, fontweight="bold")

        # Annotate bars
        for bar in bars:
            val = bar.get_width()
            offset = 0.15 if val >= 0 else -0.4
            plt.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(figures_dir, "brier_skill_score.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_path}")

    def _plot_cumulative_logloss(
        self,
        timeline_records: List[Dict[str, Any]],
        figures_dir: str,
    ) -> None:
        """Plot chronological running cumulative Log-Loss over the validation season."""
        df = pd.DataFrame(timeline_records)

        def _cum_ll(p_col):
            losses = []
            cum = 0.0
            for idx, (y, p) in enumerate(zip(df["y"], df[p_col])):
                p_c = max(min(p, 1.0 - 1e-15), 1e-15)
                loss_i = -(y * math.log(p_c) + (1.0 - y) * math.log(1.0 - p_c))
                cum += loss_i
                losses.append(cum / (idx + 1))
            return losses

        plt.figure(figsize=(10, 6))
        x_axis = np.arange(1, len(df) + 1)

        plt.plot(x_axis, _cum_ll("p_opt"), label="Dynamic Optimized WHR", color="#1f77b4", linewidth=2.2)
        plt.plot(x_axis, _cum_ll("p_vanilla"), label="Vanilla Zero-Sum Elo", color="#ff7f0e", linewidth=1.8)
        plt.plot(x_axis, _cum_ll("p_eff"), label="Efficiency Margin Regression", color="#2ca02c", linewidth=1.6)
        plt.plot(x_axis, _cum_ll("p_bt"), label="Static Bradley-Terry MLE", color="#d62728", linewidth=1.5, linestyle="--")
        plt.plot(x_axis, _cum_ll("p_lgb"), label="LightGBM Classifier", color="#9467bd", linewidth=1.6)

        plt.xlabel("Validation Match Sequence (Chronological Jan - Mar 2022)", fontsize=11)
        plt.ylabel("Cumulative Out-of-Sample Log-Loss", fontsize=11)
        plt.title("Chronological Log-Loss Stability: Mid-Season to March Tournament Finals", fontsize=13, fontweight="bold")
        plt.legend(loc="upper right", frameon=True, fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(figures_dir, "cumulative_logloss_timeline.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_path}")

