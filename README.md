# Basketball Game Prediction Engine

A modular collegiate basketball rating and win-probability engine implementing a zero-sum Multi-Factor Bradley-Terry / Elo system with spatial travel fatigue, dynamic strength of schedule ($SOS_t$), and bounded logistic box-score performance adjustments ($AGPF$).

## Architecture

```text
├── data/
│   ├── final_final_data.csv          # Historical NCAA game logs (10,438 team-game rows)
│   ├── predict.csv                   # Tournament bracket matchups
│   └── team_airports.json            # Home venue & airport coordinates
├── output_data/
│   └── optimized_parameters.json     # Walk-forward Bayesian calibrated model weights
├── src/
│   ├── cli.py                        # Unified CLI entry point (run, predict, tune, benchmark)
│   ├── config.py                     # Default constants and parameter definitions
│   ├── data/
│   │   ├── loader.py                 # Chronological game pairing & validation
│   │   └── spatial.py                # Vectorized Haversine distance & timezone shift
│   ├── features/
│   │   ├── fatigue.py                # Rest day & travel fatigue indices (ORTF, TROF)
│   │   ├── metrics.py                # Z-scored box-score normalization & logistic AGPF
│   │   └── schedule.py               # Dynamic in-memory Strength of Schedule (SOS_t)
│   ├── models/
│   │   ├── bradley_terry.py          # Modified Bradley-Terry win probability function
│   │   ├── elo.py                    # Strictly zero-sum Elo tracker & update rule
│   │   └── baselines.py              # Comparative baselines (Vanilla Elo, Logistic, Seed)
│   ├── optimization/
│   │   ├── objective.py              # Walk-forward out-of-sample Log-Loss objective
│   │   ├── sampler.py                # Optuna hyperparameter simplex & parameter sampler
│   │   └── tune.py                   # Bayesian hyperparameter tuning runner
│   └── evaluation/
│       ├── benchmark.py              # Out-of-sample calibration, Brier, ECE & bootstrap CI
│       ├── diagnostics.py            # Diagnostic plots
│       └── predict.py                # Tournament bracket matchup simulator
└── tests/                            # Unit & end-to-end verification suite
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Run Full Season Simulation
Simulates all 5,219 historical games chronologically using the calibrated parameters in `output_data/optimized_parameters.json`, exporting final team ratings (`output_data/final_team_ratings.csv`) and bracket predictions (`output_data/bracket_predictions.csv`):
```bash
python -m src.cli run
```

### 2. Predict Bracket Matchups
Evaluates win probabilities for matchups in `data/predict.csv` using end-of-season ratings:
```bash
python -m src.cli predict
```

### 3. Benchmark Against Baselines
Evaluates out-of-sample Log-Loss, Brier Score, Expected Calibration Error (ECE), and 1,000-sample paired bootstrap significance tests against Vanilla Elo, Rolling Net Efficiency Logistic Regression, and Tournament Seed baselines:
```bash
python -m src.cli benchmark
```

### 4. Tune Hyperparameters
Runs walk-forward Bayesian optimization with Optuna and updates `output_data/optimized_parameters.json`:
```bash
python -m src.cli tune --n-trials 150
```

## Testing

```bash
python -m unittest discover -s tests -v
```
