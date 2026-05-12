# NBA Fantasy Points Prediction Project

This repository package contains a supervised machine-learning project for predicting NBA player-game fantasy points.

## Data

The notebook downloads the raw data directly from Kaggle using `kagglehub`. Internet access is required on the first run. If Kaggle requests authentication, configure a Kaggle API token before running the notebook.

## Objective

Predict one player's fantasy points for one regular-season NBA game before the game is played.

Target formula:

```text
fantasy_points = points + 1.2 * reboundsTotal + 1.5 * assists + 3 * steals + 3 * blocks - turnovers
```

## Split Design

The project uses a chronological split:

```text
Train:      2021, 2022, 2023 seasons
Validation: 2024 season
Test:       2025 season
```

The 2025 test season is used once after model selection.

## Workflow

The final notebook follows the course workflow: data preparation, table unification, cleansing checks, EDA, feature engineering, feature selection, modeling, tuning, final evaluation, and reporting.

## Leakage Control

Direct current-game outcome fields are not valid model features. In particular, direct `starter_minutes` is treated only as a leakage diagnostic because it includes actual minutes from the game being predicted.

## Best Validation Result

Best valid validation row:

```text
Best Isolation Forest anomaly model / lineup_aware_starter_flag_roll10_best_pca_plus_isolation / Best Isolation Forest anomaly model
MAE  = 7.5797
RMSE = 9.6947
R2   = 0.5773
```

## Final 2025 Test Result

The selected controlled final model was evaluated once on 2025:

```text
Feature set = lineup_aware_starter_flag_roll10
Model       = catboost
MAE         = 7.6712
RMSE        = 9.7909
R2          = 0.5440
```

## How To Run

Recommended setup: create an environment from `requirements.txt`, then run the notebook. The first notebook cell also checks for missing packages and installs them into the active Python environment if needed. The notebook downloads the Kaggle dataset automatically:

Use Python 3.10 or newer. Python 3.11 or 3.12 is recommended, but the notebook will continue on newer versions when the required packages are available.

```text
runs/final/notebooks/NBA_ML_project.ipynb
```

All final outputs are written under `reports/` inside this final project repo. If the notebook is run from the original parent project layout, outputs still resolve to `runs/final/reports/`.

## GitHub Notes

Do not upload the regenerated Kaggle cache, processed modeling tables, or large prediction-dump CSV files. They are recreated by the notebook and are ignored by `.gitignore` because several are larger than GitHub's normal file limits.
