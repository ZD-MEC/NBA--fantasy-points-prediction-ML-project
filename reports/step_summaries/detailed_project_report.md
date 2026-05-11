# Detailed NBA ML Project Report

This report explains the final notebook in depth for project partners. It covers what each section does, why it exists, how it works, important parameters, outputs, results, and every function defined in the notebook.

Source notebook: `C:\Users\zivdi\fantasy_basketball_fp_starter\NBA_ML_PROJ_2\runs\final\notebooks\NBA_ML_project.ipynb`

## Executive Summary

The project predicts one player-game fantasy-points value before the game is played. The final workflow downloads NBA data from Kaggle, builds a legal time-aware modeling table, checks leakage, performs EDA, creates rolling historical features, compares model families, tests PCA and Isolation Forest feature ideas, evaluates final candidates on the 2024 validation season, and uses the 2025 test season once after model choice.

Target: `fantasy_points = points + 1.2 * reboundsTotal + 1.5 * assists + 3 * steals + 3 * blocks - turnovers`.

Core lesson: same-game actual minutes explain the below-7 MAE results, but they are leakage. Valid models use shifted historical minutes such as `starter_minutes_roll_10`, and lineup-aware models may use `current_is_starter` only when starting lineups are known before prediction.

## Libraries And Why They Are Used

| Library | Why used | Important methods/classes |
| --- | --- | --- |
| pathlib/os | Portable paths and environment setup | Path, mkdir, glob, os.environ |
| pandas | Main tabular data tool | read_csv, read_parquet, merge, groupby, rolling, to_csv, corr |
| numpy | Vectorized numeric calculations | np.where, arrays, missing-value handling |
| matplotlib/seaborn | EDA and model figures | histograms, boxplots, heatmaps, line plots |
| kagglehub | Direct Kaggle dataset download | dataset_download |
| scikit-learn | Preprocessing, models, metrics, PCA, anomaly detection | Pipeline, SimpleImputer, StandardScaler, Ridge, Lasso, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, PCA, IsolationForest, permutation_importance |
| xgboost | Boosted-tree comparison | XGBRegressor |
| lightgbm | Fast boosted-tree comparison | LGBMRegressor |
| catboost | Final controlled boosted-tree candidate | CatBoostRegressor |

## Important Parameters And Configurations

| Parameter | Value | Meaning |
| --- | --- | --- |
| Kaggle dataset | eoinamoore/historical-nba-data-and-player-box-scores | Raw data source downloaded in the notebook |
| Run folder | runs/final | All final outputs are organized here |
| Seasons | 2021-2025 | Full data scope |
| Train | 2021-2023 | Fitting models and train-only transforms |
| Validation | 2024 | Model selection and comparison |
| Test | 2025 | Used once after final selection |
| Split column | split_2021_train_2024_validation_2025_test | Explicit split label |
| Rolling windows | [3, 5, 10, 30] | Recent-form historical averages |
| PCA top candidates | 100 | Most correlated legal unused train features |
| PCA missingness cap | 40 percent | Drops sparse PCA candidates |
| PCA components | 10, 20, 30, and 95 percent variance | Compression grid |
| Isolation Forest contamination | 0.05 | Outlier/anomaly share |
| Random state | 42 | Reproducibility |
| RandomForest cap | 150 estimators | Runtime control |

## Split Summary

| split | rows | unique_games | unique_players | date_min | date_max | seasons | target_mean | target_median | target_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 76937 | 3624 | 818 | 2021-10-19 19:30:00 | 2024-04-14 15:30:00 | 2021, 2022, 2023 | 21.4586 | 19.4000 | 14.8669 |
| validation | 26162 | 1223 | 569 | 2024-10-22 19:30:00 | 2025-04-13 15:30:00 | 2024 | 21.7225 | 20.0000 | 14.9117 |
| test | 26651 | 1230 | 582 | 2025-10-21 19:30:00 | 2026-04-12 20:30:00 | 2025 | 21.6201 | 20.0000 | 14.4996 |

Team game-type repair check:

| check | rows | regular_season_rows |
| --- | --- | --- |
| 2021 regular-season team rows after Games.csv gameType repair | 2460 | 2460 |

## Section By Section Explanation

### 1. Objective And Target Definition

**What was done:** Defines target, row grain, prediction moment, and leakage rules.

**Why:** Without this, model features and metrics cannot be judged correctly.

**How:** The section states fantasy-points formula and says current-game outcomes are invalid inputs.

**Important parameters/configuration:** Target formula and pre-game prediction moment.

### 2. Imports, Paths, And Constants

**What was done:** Imports libraries, finds project root, downloads Kaggle data, creates output folders, fixes seeds.

**Why:** Makes the notebook rerunnable on a new computer.

**How:** Uses KaggleHub cache under runs/final and central path constants.

**Important parameters/configuration:** KAGGLE_DATASET, RUN_DIR, RAW_DIR, PROCESSED_DIR, RANDOM_STATE.

### 3. Raw Data Collection And Inventory

**What was done:** Scans raw Kaggle tables and profiles columns, sizes, keys, and candidate sources.

**Why:** Before modeling, we need to know what data exists and which joins are possible.

**How:** Reads headers and table metadata, then writes inventory tables.

**Important parameters/configuration:** Supported CSV/Parquet files, key columns gameId/personId/teamId/date.

### 3.1 Raw Data Inventory Execution

**What was done:** Runs the inventory and displays/writes the tables.

**Why:** Creates an audit trail for the raw-data choice.

**How:** Writes raw_table_inventory, table_column_summary, candidate_base_tables.

**Important parameters/configuration:** No modeling here.

### 4. Data Preparation And Table Unification

**What was done:** Creates one player-game source table and full modeling table.

**Why:** Models need one legal row per player per game.

**How:** Loads player/team stats, repairs team game type, filters regular seasons, computes target, creates shifted rolling features, merges team/opponent/profile context.

**Important parameters/configuration:** TARGET_WEIGHTS and rolling plans.

### 5. Data Cleansing Checks

**What was done:** Checks missingness, row grain, leakage, column decisions, and 2021 repair.

**Why:** Catches data problems before model training.

**How:** Writes decision and check tables.

**Important parameters/configuration:** team_2021_game_type_repair_check confirms 2460 regular-season team rows.

### 6. Exploratory Data Analysis

**What was done:** Studies target distribution, correlations, missingness, outliers, starter behavior, minutes buckets, and position patterns.

**Why:** EDA explains the data and guides feature/model choices.

**How:** Writes tables and figures for target summaries, correlations, missingness, and unusual games.

**Important parameters/configuration:** Diagnostic correlations are not automatic feature approval.

### 7. Feature Engineering

**What was done:** Builds valid feature sets and feature drop decisions.

**Why:** Different prediction moments need different valid inputs.

**How:** Filters direct same-game features, builds pre-lineup, lineup-aware, compact, broad, and starter sets.

**Important parameters/configuration:** Key sets: pre_lineup_player_team_opp, pre_lineup_compact_40_plus_scoring, pre_lineup_starter_history_roll10, lineup_aware_starter_flag_roll10.

### 8. Chronological Train/Validation/Test Split

**What was done:** Assigns rows to train, validation, and test by season.

**Why:** Sports prediction must evaluate future seasons from past seasons.

**How:** 2021-2023 train, 2024 validation, 2025 test.

**Important parameters/configuration:** SPLIT_COL records the split on every row.

### 9. Feature Sets And Modeling Datasets

**What was done:** Materializes reusable model feature sets.

**Why:** Prevents ad hoc column choices later in the notebook.

**How:** Writes feature_set_membership and feature_set_summary.

**Important parameters/configuration:** Starter features exclude direct starter_minutes.

### 10. Main Model Comparison

**What was done:** Runs initial baselines, Ridge, and HGB.

**Why:** Creates a baseline before trying heavier methods.

**How:** Fits on train and evaluates on 2024 validation, writing predictions and segment errors.

**Important parameters/configuration:** Metrics: MAE, RMSE, R2, mean error, median absolute error.

### 10b. Alternative Modeling Approach And Validity Check

**What was done:** Integrates useful ideas from the other notebook and labels invalid leakage results.

**Why:** The below-7 MAE result was informative but not valid for pre-game prediction.

**How:** Writes alternative_modeling_review and leakage_diagnostic_actual_minutes.

**Important parameters/configuration:** Direct starter_minutes is diagnostic only.

### 10c. Controlled Model Comparison With Alternative Modeling Ideas

**What was done:** Runs Ridge, Lasso, HGB, XGBoost, LightGBM, CatBoost, ExtraTrees, RandomForest, and time decay variants.

**Why:** Tests stronger model ideas under the clean split.

**How:** A model factory creates controlled configs and writes train/validation overfitting gaps.

**Important parameters/configuration:** CatBoost 500 iterations; RandomForest/ExtraTrees max 150 estimators.

### 11. Starter-Focused Model Comparison

**What was done:** Tests starter-history and lineup-aware starter feature families.

**Why:** Role and minutes are central fantasy drivers.

**How:** Pre-lineup starter uses shifted history; lineup-aware also uses current_is_starter.

**Important parameters/configuration:** Best starter HGB MAE 7.6559.

### 12. Expanding Time Cross-Validation

**What was done:** Runs time-aware CV inside train seasons only.

**Why:** Random CV would leak future timing.

**How:** Four expanding folds train on earlier blocks and validate on later blocks.

**Important parameters/configuration:** CV_N_SPLITS=4, CV_MIN_TRAIN_FRACTION=0.40.

### 13. PCA And Dimensionality Reduction

**What was done:** Compresses unused broad legal pre-lineup features.

**Why:** Tests whether many legal features help after compression.

**How:** Ranks train-only candidates, imputes/scales/fits PCA on train, transforms validation only.

**Important parameters/configuration:** Top 100 candidates, 40 percent missingness cap.

### 13b. Starter PCA Comparison

**What was done:** Adds PCA components to starter feature sets.

**Why:** The starter family was strong, so broad compressed context may help.

**How:** Excludes starter base features from the PCA pool and compares Ridge/HGB.

**Important parameters/configuration:** Best starter PCA MAE 7.5815.

### 13c. Isolation Forest Anomaly Features

**What was done:** Creates anomaly scores, outlier flags, and deciles.

**Why:** Unusual rows may be less dependable and useful as model features.

**How:** Fits IsolationForest on train only and transforms validation.

**Important parameters/configuration:** Best anomaly combo MAE 7.5797.

### 14. HGB Tuning

**What was done:** Tunes compact HGB configurations and compact refinements.

**Why:** Checks if an explainable compact model can be improved.

**How:** Tests learning rate, max_iter, max_leaf_nodes, L2, min_samples_leaf, early stopping.

**Important parameters/configuration:** Best tuned compact MAE 7.7700.

### 14b. Valid 11-Feature Replication

**What was done:** Recreates the 11-feature idea without direct leakage.

**Why:** Shows how much performance remains after fixing same-game minutes.

**How:** Uses valid shifted starter history instead of actual starter_minutes.

**Important parameters/configuration:** Best valid 11-feature MAE 7.8180.

### 15. Final Model Evaluation And Recommendation

**What was done:** Summarizes best valid model families.

**Why:** Separates pre-lineup, lineup-aware, compact, and diagnostic results.

**How:** Writes final comparison and recommendation files.

**Important parameters/configuration:** Best pre-lineup MAE 7.7315; best lineup-aware MAE 7.5797.

### 17b. Integrated Final Validation Ranking And One-Time 2025 Test Evaluation

**What was done:** Ranks final validation candidates and evaluates one selected final model on test.

**Why:** The test set must not influence model choice.

**How:** Selects a controlled CatBoost lineup-aware model and evaluates once on 2025.

**Important parameters/configuration:** Test MAE 7.6712, RMSE 9.7909, R2 0.5440.

### 16. Feature Importance And Interpretability

**What was done:** Explains drivers using final importance and profile/extended rolling-10 experiment.

**Why:** Model results need interpretation, not only metrics.

**How:** Builds profile/extended table and computes model/permutation importance.

**Important parameters/configuration:** ext_ means shifted rolling feature from PlayerStatisticsExtended.csv.

### 17. Artifact Inventory And Final Report Links

**What was done:** Inventories generated tables, figures, and summaries.

**Why:** Makes outputs easy to find and verify.

**How:** Scans output folders and writes inventory tables.

**Important parameters/configuration:** artifact_inventory_final.csv.

### 19. Final GitHub Package And Presentation Materials

**What was done:** Writes README, requirements, NotebookLM brief, presentation outline.

**Why:** Supports lecturer review and project presentation.

**How:** Documents run instructions, data download, leakage warning, and final results.

**Important parameters/configuration:** Large regenerated files are ignored by .gitignore.

## Main Results

### Final Validation Ranking
| model_label | feature_set | model | n_features | validity | mae | rmse | r2 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best Isolation Forest anomaly model | lineup_aware_starter_flag_roll10_best_pca_plus_isolation | Best Isolation Forest anomaly model | 44 | lineup-aware | 7.5797 | 9.6947 | 0.5773 | Isolation Forest anomaly features; variant lineup_aware_anomaly. |
| Best valid starter PCA model | lineup_aware_starter_flag_roll10_plus_pca_30 | Best valid starter PCA model | 41 | lineup-aware | 7.5815 | 9.6940 | 0.5774 | Starter model plus 30 PCA components. |
| Controlled model | lineup_aware_starter_flag_roll10 | catboost | 11 | lineup-aware | 7.6437 | 9.7800 | 0.5698 | catboost without time-decay sample weights. |
| Best valid starter-focused model | lineup_aware_starter_flag_roll10 | Best valid starter-focused model | 11 | lineup-aware | 7.6559 | 9.7825 | 0.5696 | Starter model family included as a main comparison group. |
| Lineup-aware starter-flag HGB | lineup_aware_starter_flag_roll10 | Lineup-aware starter-flag HGB | 11 | lineup-aware | 7.6559 | 9.7825 | 0.5696 | Uses current_is_starter plus starter_minutes_roll_10, not actual current-game minutes. |
| Controlled model | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 11 | lineup-aware | 7.6559 | 9.7825 | 0.5696 | hist_gradient_boosting without time-decay sample weights. |
| Controlled model with time-decay weights | lineup_aware_starter_flag_roll10 | catboost | 11 | lineup-aware | 7.6607 | 9.7944 | 0.5686 | catboost with train-only time-decay sample weights. |
| Controlled model with time-decay weights | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 11 | lineup-aware | 7.6618 | 9.7856 | 0.5693 | hist_gradient_boosting with train-only time-decay sample weights. |
| Controlled model | lineup_aware_starter_flag_roll10 | lightgbm | 11 | lineup-aware | 7.6679 | 9.8118 | 0.5670 | lightgbm without time-decay sample weights. |
| Controlled model with time-decay weights | lineup_aware_starter_flag_roll10 | extra_trees | 11 | lineup-aware | 7.6721 | 9.8002 | 0.5681 | extra_trees with train-only time-decay sample weights. |
| Controlled model | lineup_aware_starter_flag_roll10 | xgboost | 11 | lineup-aware | 7.6785 | 9.8319 | 0.5653 | xgboost without time-decay sample weights. |
| Controlled model with time-decay weights | lineup_aware_starter_flag_roll10 | random_forest | 11 | lineup-aware | 7.6797 | 9.8090 | 0.5673 | random_forest with train-only time-decay sample weights. |

### One-Time 2025 Test
| selected_by | feature_set | model | uses_time_decay | n_features | development_rows_2021_2024 | test_rows_2025 | test_mae | test_rmse | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lowest_2024_validation_mae_from_controlled_valid_models | lineup_aware_starter_flag_roll10 | catboost | False | 11 | 103099 | 26651 | 7.6712 | 9.7909 | 0.5440 |

### Controlled Models
| feature_set | model | uses_time_decay | validity | n_features | train_mae | validation_mae | mae_gap | validation_rmse | validation_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10 | catboost | False | lineup-aware | 11 | 7.6140 | 7.6437 | 0.0297 | 9.7800 | 0.5698 |
| lineup_aware_starter_flag_roll10 | hist_gradient_boosting | False | lineup-aware | 11 | 7.6220 | 7.6559 | 0.0339 | 9.7825 | 0.5696 |
| lineup_aware_starter_flag_roll10 | catboost | True | lineup-aware | 11 | 7.6214 | 7.6607 | 0.0393 | 9.7944 | 0.5686 |
| lineup_aware_starter_flag_roll10 | hist_gradient_boosting | True | lineup-aware | 11 | 7.6636 | 7.6618 | -0.0019 | 9.7856 | 0.5693 |
| lineup_aware_starter_flag_roll10 | lightgbm | False | lineup-aware | 11 | 7.4455 | 7.6679 | 0.2223 | 9.8118 | 0.5670 |
| lineup_aware_starter_flag_roll10 | extra_trees | True | lineup-aware | 11 | 7.4325 | 7.6721 | 0.2396 | 9.8002 | 0.5681 |
| lineup_aware_starter_flag_roll10 | xgboost | False | lineup-aware | 11 | 7.3319 | 7.6785 | 0.3466 | 9.8319 | 0.5653 |
| lineup_aware_starter_flag_roll10 | random_forest | True | lineup-aware | 11 | 7.3328 | 7.6797 | 0.3469 | 9.8090 | 0.5673 |
| lineup_aware_starter_flag_roll10 | random_forest | False | lineup-aware | 11 | 7.2778 | 7.6919 | 0.4141 | 9.8072 | 0.5674 |
| lineup_aware_starter_flag_roll10 | extra_trees | False | lineup-aware | 11 | 7.4191 | 7.6937 | 0.2745 | 9.8042 | 0.5677 |
| lineup_aware_starter_flag_roll10 | lightgbm | True | lineup-aware | 11 | 7.4791 | 7.7033 | 0.2243 | 9.8644 | 0.5624 |
| lineup_aware_starter_flag_roll10 | lasso | True | lineup-aware | 11 | 7.8074 | 7.7039 | -0.1035 | 9.8481 | 0.5638 |

### Starter Models
| feature_set | model | n_features | train_rows | validation_rows | fit_predict_seconds | mae | rmse | r2 | mean_error | median_absolute_error | validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 11 | 76937 | 26162 | 1.2520 | 7.6559 | 9.7825 | 0.5696 | 0.1455 | 6.2971 | lineup-aware |
| lineup_aware_starter_flag_roll10 | ridge | 11 | 76937 | 26162 | 0.1220 | 7.7216 | 9.8659 | 0.5622 | -0.0856 | 6.3781 | lineup-aware |
| pre_lineup_starter_history_roll10 | hist_gradient_boosting | 10 | 76937 | 26162 | 1.0350 | 7.9097 | 10.0878 | 0.5423 | 0.1016 | 6.5588 | pre-lineup |
| pre_lineup_starter_history_roll10 | ridge | 10 | 76937 | 26162 | 0.1360 | 7.9525 | 10.1508 | 0.5366 | -0.0795 | 6.6003 | pre-lineup |

### Time Cross-Validation
| feature_set | model | n_features | folds | mean_mae | std_mae | min_mae | max_mae | mean_rmse | mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre_lineup_player_team_opp | hist_gradient_boosting | 486 | 4 | 7.7063 | 0.0896 | 7.6017 | 7.8202 | 9.8657 | 0.5716 |
| pre_lineup_compact_40_plus_scoring | hist_gradient_boosting | 45 | 4 | 7.7547 | 0.0812 | 7.6593 | 7.8580 | 9.9127 | 0.5675 |
| pre_lineup_compact_40 | hist_gradient_boosting | 40 | 4 | 7.7550 | 0.0850 | 7.6540 | 7.8620 | 9.9187 | 0.5669 |
| pre_lineup_compact_40 | ridge | 40 | 4 | 7.7775 | 0.0923 | 7.6724 | 7.8968 | 9.9269 | 0.5663 |
| pre_lineup_compact_40_plus_scoring | ridge | 45 | 4 | 7.7777 | 0.0928 | 7.6736 | 7.8988 | 9.9266 | 0.5663 |
| pre_lineup_baseline | ridge | 4 | 4 | 7.8744 | 0.1200 | 7.7351 | 8.0283 | 10.0350 | 0.5567 |
| pre_lineup_baseline | hist_gradient_boosting | 4 | 4 | 7.9027 | 0.1105 | 7.7662 | 8.0359 | 10.0482 | 0.5556 |
| pre_lineup_baseline | rolling_fp_10_baseline | 4 | 4 | 7.9303 | 0.1671 | 7.7391 | 8.1463 | 10.2042 | 0.5416 |
| pre_lineup_compact_40 | rolling_fp_10_baseline | 40 | 4 | 7.9303 | 0.1671 | 7.7391 | 8.1463 | 10.2042 | 0.5416 |
| pre_lineup_compact_40_plus_scoring | rolling_fp_10_baseline | 45 | 4 | 7.9303 | 0.1671 | 7.7391 | 8.1463 | 10.2042 | 0.5416 |
| pre_lineup_player_team_opp | rolling_fp_10_baseline | 486 | 4 | 7.9303 | 0.1671 | 7.7391 | 8.1463 | 10.2042 | 0.5416 |
| pre_lineup_player_team_opp | ridge | 486 | 4 | 8.1015 | 0.6315 | 7.7254 | 9.0407 | 10.3651 | 0.5245 |

### PCA
| experiment | model | feature_set | n_features | n_pca_source_features | n_pca_components | pca_explained_variance_ratio_sum | train_mae | validation_mae | mae_gap | validation_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_hgb_reference | hist_gradient_boosting | reference | 484 | 0 | 0 |  | 7.3541 | 7.7315 | 0.3775 | 0.5591 |
| compact_scoring_plus_pca_30 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 75 | 100 | 30 | 0.9080 | 7.4692 | 7.7480 | 0.2787 | 0.5585 |
| compact_scoring_plus_pca_41 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 86 | 100 | 41 | 0.9519 | 7.4528 | 7.7491 | 0.2963 | 0.5589 |
| compact_scoring_plus_pca_30 | ridge | pre_lineup_compact_40_plus_scoring | 75 | 100 | 30 | 0.9080 | 7.7126 | 7.7519 | 0.0393 | 0.5592 |
| compact_scoring_plus_pca_20 | ridge | pre_lineup_compact_40_plus_scoring | 65 | 100 | 20 | 0.8322 | 7.7153 | 7.7523 | 0.0370 | 0.5589 |
| compact_scoring_plus_pca_10 | ridge | pre_lineup_compact_40_plus_scoring | 55 | 100 | 10 | 0.6679 | 7.7233 | 7.7546 | 0.0313 | 0.5585 |
| compact_scoring_plus_pca_41 | ridge | pre_lineup_compact_40_plus_scoring | 86 | 100 | 41 | 0.9519 | 7.7091 | 7.7564 | 0.0472 | 0.5590 |
| compact_scoring_plus_pca_20 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 65 | 100 | 20 | 0.8322 | 7.5131 | 7.7620 | 0.2489 | 0.5572 |
| compact_scoring_plus_pca_10 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 55 | 100 | 10 | 0.6679 | 7.5229 | 7.7633 | 0.2404 | 0.5576 |
| compact_scoring_hgb_reference | hist_gradient_boosting | reference | 45 | 0 | 0 |  | 7.4903 | 7.7818 | 0.2915 | 0.5549 |

### Starter PCA
| experiment | base_feature_set | model | n_features | n_pca_components | pca_explained_variance_ratio_sum | train_mae | validation_mae | mae_gap | validation_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10_plus_pca_30 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 41 | 30 | 0.9276 | 7.4090 | 7.5815 | 0.1725 | 0.5774 |
| lineup_aware_starter_flag_roll10_plus_pca_37 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 48 | 37 | 0.9529 | 7.3495 | 7.5816 | 0.2321 | 0.5778 |
| lineup_aware_starter_flag_roll10_plus_pca_20 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 31 | 20 | 0.8606 | 7.4275 | 7.5848 | 0.1573 | 0.5770 |
| lineup_aware_starter_flag_roll10_plus_pca_30 | lineup_aware_starter_flag_roll10 | ridge | 41 | 30 | 0.9276 | 7.6379 | 7.5897 | -0.0482 | 0.5752 |
| lineup_aware_starter_flag_roll10_plus_pca_37 | lineup_aware_starter_flag_roll10 | ridge | 48 | 37 | 0.9529 | 7.6319 | 7.5914 | -0.0405 | 0.5757 |
| lineup_aware_starter_flag_roll10_plus_pca_10 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 21 | 10 | 0.7216 | 7.4356 | 7.5958 | 0.1602 | 0.5760 |
| lineup_aware_starter_flag_roll10_plus_pca_20 | lineup_aware_starter_flag_roll10 | ridge | 31 | 20 | 0.8606 | 7.6454 | 7.5976 | -0.0478 | 0.5746 |
| lineup_aware_starter_flag_roll10_plus_pca_10 | lineup_aware_starter_flag_roll10 | ridge | 21 | 10 | 0.7216 | 7.6691 | 7.6157 | -0.0534 | 0.5724 |
| lineup_aware_starter_flag_roll10_reference | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 11 | 0 |  | 7.6220 | 7.6559 | 0.0339 | 0.5696 |
| lineup_aware_starter_flag_roll10_reference | lineup_aware_starter_flag_roll10 | ridge | 11 | 0 |  | 7.7942 | 7.7216 | -0.0726 | 0.5622 |

### Isolation Forest
| experiment | base_feature_set | anomaly_variant | model | n_features | train_mae | validation_mae | mae_gap | validation_r2 | overfitting_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10_best_pca_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | hist_gradient_boosting | 44 | 7.3596 | 7.5797 | 0.2201 | 0.5773 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_best_pca_reference | lineup_aware_starter_flag_roll10 | none | hist_gradient_boosting | 41 | 7.4090 | 7.5815 | 0.1725 | 0.5774 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_best_pca_reference | lineup_aware_starter_flag_roll10 | none | ridge | 41 | 7.6379 | 7.5897 | -0.0482 | 0.5752 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_best_pca_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | ridge | 44 | 7.6376 | 7.5901 | -0.0475 | 0.5752 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_reference | lineup_aware_starter_flag_roll10 | none | hist_gradient_boosting | 11 | 7.6220 | 7.6559 | 0.0339 | 0.5696 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| lineup_aware_starter_flag_roll10_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | hist_gradient_boosting | 14 | 7.6346 | 7.6622 | 0.0276 | 0.5697 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| lineup_aware_starter_flag_roll10_reference | lineup_aware_starter_flag_roll10 | none | ridge | 11 | 7.7942 | 7.7216 | -0.0726 | 0.5622 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| lineup_aware_starter_flag_roll10_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | ridge | 14 | 7.7930 | 7.7231 | -0.0698 | 0.5620 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| pre_lineup_player_team_opp_plus_isolation | pre_lineup_player_team_opp | pre_lineup_anomaly | hist_gradient_boosting | 487 | 7.3494 | 7.7308 | 0.3814 | 0.5594 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| pre_lineup_player_team_opp_reference | pre_lineup_player_team_opp | none | hist_gradient_boosting | 484 | 7.3541 | 7.7315 | 0.3775 | 0.5591 | Small gap with worse validation MAE = underpowered or unhelpful features. |

### HGB Tuning
| feature_set | config_name | n_features | mae | rmse | r2 | learning_rate | max_iter | max_leaf_nodes | l2_regularization | min_samples_leaf | early_stopping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre_lineup_compact_40_plus_scoring | no_early_stop_more_trees | 45 | 7.7700 | 9.9330 | 0.5563 | 0.0300 | 260 | 31 | 0.3000 | 40 | False |
| pre_lineup_compact_40_plus_scoring | large_leaf | 45 | 7.7721 | 9.9349 | 0.5561 | 0.0500 | 180 | 31 | 0.1000 | 80 | auto |
| pre_lineup_compact_40_plus_scoring | no_internal_early_stop_reference | 45 | 7.7734 | 9.9369 | 0.5559 | 0.0500 | 160 | 31 | 0.1000 | 20 | False |
| pre_lineup_compact_40_plus_scoring | slower_learning_more_trees | 45 | 7.7769 | 9.9395 | 0.5557 | 0.0300 | 260 | 31 | 0.1000 | 20 | auto |
| pre_lineup_compact_40_plus_scoring | slower_learning_more_trees_l2_03 | 45 | 7.7776 | 9.9377 | 0.5558 | 0.0300 | 260 | 31 | 0.3000 | 20 | auto |
| pre_lineup_compact_40_plus_scoring | no_early_stop_small_trees | 45 | 7.7780 | 9.9366 | 0.5559 | 0.0300 | 300 | 15 | 0.3000 | 40 | False |
| pre_lineup_compact_40_plus_scoring | medium_leaf | 45 | 7.7782 | 9.9347 | 0.5561 | 0.0500 | 180 | 31 | 0.1000 | 40 | auto |
| pre_lineup_compact_40_plus_scoring | medium_l2 | 45 | 7.7797 | 9.9420 | 0.5555 | 0.0500 | 180 | 31 | 0.3000 | 20 | auto |
| pre_lineup_compact_40_plus_scoring | high_l2 | 45 | 7.7813 | 9.9460 | 0.5551 | 0.0500 | 180 | 31 | 0.8000 | 20 | auto |
| pre_lineup_compact_40_plus_scoring | low_l2 | 45 | 7.7817 | 9.9421 | 0.5555 | 0.0500 | 180 | 31 | 0.0000 | 20 | auto |

### Valid 11-Feature Replication
| feature_set | model | n_features | train_rows | validation_rows | fit_predict_seconds | mae | rmse | r2 | mean_error | median_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replication_11_safe_recent_starter | hist_gradient_boosting | 11 | 76937 | 26162 | 1.1630 | 7.8180 | 9.9750 | 0.5525 | 0.0836 | 6.4605 |
| replication_11_safe | hist_gradient_boosting | 11 | 76937 | 26162 | 1.2180 | 7.8297 | 9.9988 | 0.5504 | 0.0761 | 6.4821 |
| replication_11_safe_recent_starter | ridge | 11 | 76937 | 26162 | 0.1240 | 7.8497 | 10.0201 | 0.5485 | -0.0962 | 6.4910 |
| replication_11_safe | ridge | 11 | 76937 | 26162 | 0.1320 | 7.8560 | 10.0314 | 0.5474 | -0.0554 | 6.5019 |

### Leakage Diagnostic
| model_family | reported_mae | valid_for_final_ranking | leakage_feature | explanation |
| --- | --- | --- | --- | --- |
| Alternative XGBoost final | 6.7061 | False | starter_minutes | Uses actual current-game minutes through starter_minutes. |
| Alternative ExtraTrees final | 6.7576 | False | starter_minutes | Uses actual current-game minutes through starter_minutes. |
| Alternative 11-feature ExtraTrees | 6.9955 | False | starter_minutes | This explains why the below-7 MAE was possible but not a valid pre-game prediction result. |

### Alternative Modeling Review
| item | value | status | reason |
| --- | --- | --- | --- |
| reported_best_xgboost_mae | 6.7061 | diagnostic_only | Reported on an 80/20 chronological row split using direct starter_minutes and other non-final assumptions. |
| reported_best_extra_trees_mae | 6.7576 | diagnostic_only | Strong result, but not comparable to the clean 2021-2023 train / 2024 validation / 2025 test design. |
| direct_starter_minutes | used | leakage | starter_minutes = current_is_starter * actual numMinutes, and actual minutes are known only after/during the game. |
| target_formula | alternative notebook used -1.5 * turnovers in places | corrected | This project target uses -1.0 * turnovers, matching the agreed fantasy-points formula. |
| valid_ideas_kept | tree ensembles, boosted models, time decay, rolling windows, momentum features | kept | These ideas can be evaluated under the clean split and leakage rules. |

### Profile/Extended Rolling-10 Interpretability
| feature_set | model | n_features | train_rows | validation_rows | fit_predict_seconds | mae | rmse | r2 | mean_error | median_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| profile_extended_driver_10 | hist_gradient_boosting | 144 | 76937 | 26162 | 11.7570 | 7.8685 | 10.0754 | 0.5435 | 0.3738 | 6.4401 |
| profile_extended_driver_10 | ridge | 144 | 76937 | 26162 | 4.5700 | 7.8929 | 10.0689 | 0.5440 | 0.0066 | 6.5593 |
| profile_extended_driver_10 | random_forest | 144 | 76937 | 26162 | 146.5740 | 7.9404 | 10.1307 | 0.5384 | 0.2070 | 6.5226 |
| profile_extended_driver_10 | mean_baseline | 144 | 76937 | 26162 | 0.0010 | 12.0732 | 14.9138 | -0.0003 | 0.2639 | 10.7586 |

### Final Feature Importance
| feature | importance | importance_type |
| --- | --- | --- |
| fantasy_points_roll_10 | 63.1268 | model_feature_importance |
| current_is_starter | 13.1747 | model_feature_importance |
| numMinutes_roll_10 | 8.1164 | model_feature_importance |
| usagePercentage_roll_10 | 5.0638 | model_feature_importance |
| pos_opp_difficulty_roll_30 | 2.8271 | model_feature_importance |
| days_since_last_game | 2.6418 | model_feature_importance |
| starter_minutes_share_roll_10 | 1.9702 | model_feature_importance |
| starter_minutes_roll_10 | 1.3301 | model_feature_importance |
| current_is_starter_roll_10 | 1.0987 | model_feature_importance |
| home | 0.4131 | model_feature_importance |
| is_back_to_back | 0.2373 | model_feature_importance |

## Function Catalog

The following table documents every function defined in the notebook. `Receives` is the signature, `Purpose` is the function role, `Output` is the returned object or side effect, and `Used in` is the notebook section that owns it.

### Functions In 2. Imports, Paths, And Constants

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `find_project_root` | `find_project_root(start)` | Find the NBA_ML_PROJ_2 folder even when Jupyter starts inside runs/final/notebooks. | Path | 2. Imports, Paths, And Constants |
### Functions In 3. Raw Data Collection And Inventory

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `list_raw_table_files` | `list_raw_table_files(raw_dir=RAW_DIR)` | Return supported raw table files in deterministic order. | list[Path] | 3. Raw Data Collection And Inventory |
| `list_raw_csv_files` | `list_raw_csv_files(raw_dir=RAW_DIR)` | Return raw CSV files in deterministic order. | list[Path] | 3. Raw Data Collection And Inventory |
| `table_name_from_path` | `table_name_from_path(path)` | Convert a raw table path into a stable table name. | str | 3. Raw Data Collection And Inventory |
| `resolve_raw_table_path` | `resolve_raw_table_path(table_name_or_path, raw_dir=RAW_DIR)` | Resolve a table name, file name, or path to a raw table path. | Path | 3. Raw Data Collection And Inventory |
| `_read_table` | `_read_table(path, columns=None, nrows=None)` | Read a CSV or Parquet raw table with optional column and row limits. | pd.DataFrame | 3. Raw Data Collection And Inventory |
| `_raw_table_metadata` | `_raw_table_metadata(path)` | Return file-level metadata for a raw table path without loading the full table. | tuple[list[str] \| None, int \| None, str \| None] | 3. Raw Data Collection And Inventory |
| `load_raw_table` | `load_raw_table(table_name_or_path, columns=None, nrows=None, raw_dir=RAW_DIR)` | Load one raw table without mutating it. | pd.DataFrame | 3. Raw Data Collection And Inventory |
| `count_csv_rows` | `count_csv_rows(path)` | Count data rows in a CSV file, excluding the header. | int | 3. Raw Data Collection And Inventory |
| `detect_suspected_keys` | `detect_suspected_keys(columns)` | Return key-like columns found in a table. | str | 3. Raw Data Collection And Inventory |
| `suspected_key_columns` | `suspected_key_columns(columns)` | Return key-like columns as (role, column) pairs. | list[tuple[str, str]] | 3. Raw Data Collection And Inventory |
| `raw_table_inventory` | `raw_table_inventory(raw_dir=RAW_DIR, count_rows=False)` | Summarize every supported table currently available in the Kaggle download folder. | pd.DataFrame | 3. Raw Data Collection And Inventory |
| `dataframe_info_summary` | `dataframe_info_summary(df)` | Return the text form of DataFrame.info() for notebook display. | str | 3. Raw Data Collection And Inventory |
| `_profile_value` | `_profile_value(value)` | Return a hashable representation for raw profiling output. | object | 3. Raw Data Collection And Inventory |
| `_safe_unique_count` | `_safe_unique_count(series)` | Count unique values, falling back for array/list-like raw values. | int | 3. Raw Data Collection And Inventory |
| `profile_dataframe_columns` | `profile_dataframe_columns(df)` | Return a notebook-friendly per-column profile for one DataFrame. | pd.DataFrame | 3. Raw Data Collection And Inventory |
| `table_column_summary` | `table_column_summary(raw_dir=RAW_DIR, sample_rows=None)` | Create a column-level profile for every supported raw table. | pd.DataFrame | 3. Raw Data Collection And Inventory |
| `candidate_base_tables` | `candidate_base_tables(column_summary)` | Find raw tables that could support one row per player-game target creation. | pd.DataFrame | 3. Raw Data Collection And Inventory |
### Functions In 4. Data Preparation And Table Unification

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `_safe_divide` | `_safe_divide(numerator, denominator)` | Divide two values while returning NaN when the denominator is zero or missing. | pd.Series | 4. Data Preparation And Table Unification |
| `add_lagged_rolling_features` | `add_lagged_rolling_features(df, group_col, date_col, value_cols, windows, min_periods=1)` | Adds lagged rolling features columns. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_lagged_rolling_summary_features` | `add_lagged_rolling_summary_features(df, group_col, date_col, value_cols, windows=None, min_periods=1)` | Adds lagged rolling summary features columns. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_lagged_rolling_plan` | `add_lagged_rolling_plan(df, group_col, date_col, rolling_plan, min_periods=1)` | Apply a column-to-window rolling plan with shift(1) leakage protection. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_player_rolling_features` | `add_player_rolling_features(df, player_col='personId', date_col='game_date', rolling_plan=None)` | Add player-level shifted rolling features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `candidate_feature_columns` | `candidate_feature_columns(df)` | Return numeric engineered feature columns and avoid raw same-game stats. | list[str] | 4. Data Preparation And Table Unification |
| `_available_columns` | `_available_columns(requested, actual)` | Return requested columns that actually exist in the source schema. | list[str] | 4. Data Preparation And Table Unification |
| `_normalize_id` | `_normalize_id(series)` | Normalize numeric-looking IDs to nullable strings for stable merges. | pd.Series | 4. Data Preparation And Table Unification |
| `add_game_date_and_season` | `add_game_date_and_season(df, date_col='gameDateTimeEst')` | Add parsed game_date and NBA season_start. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `filter_regular_last4` | `filter_regular_last4(df, season_starts=None)` | Keep regular-season rows from the selected NBA season starts. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `repair_team_game_type_from_games` | `repair_team_game_type_from_games(team)` | Fill missing team-game gameType values from Games.csv before regular-season filtering. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_player_name` | `add_player_name(df)` | Add readable player_name. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `clean_player_profile` | `clean_player_profile(df)` | Clean static player profile features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_primary_position_group` | `add_primary_position_group(df)` | Add a compact G/F/C position group from starter slot and profile flags. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_starter_minute_sources` | `add_starter_minute_sources(df)` | Add same-game role-minute raw material used only through shifted rolling features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_fantasy_points` | `add_fantasy_points(df)` | Calculate the fantasy-points target from player-game box-score columns. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `load_player_base_last4_regular` | `load_player_base_last4_regular(season_starts=None)` | Load the player-game base from PlayerStatisticsExtended. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `load_team_games_last4_regular` | `load_team_games_last4_regular(season_starts=None)` | Load team-game rows from TeamStatisticsExtended. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `add_player_profile` | `add_player_profile(base)` | Merge static player profile data. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `_prefixed_team_source` | `_prefixed_team_source(team_games, prefix)` | Prefix team-context columns so own-team and opponent-team joins stay distinct. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `build_player_game_source_last4_regular` | `build_player_game_source_last4_regular(season_starts=None)` | Build one row per player-game with selected player, team, and opponent raw material. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `build_team_model_source` | `build_team_model_source(team_games)` | Create team-game modeling source columns for own-team rolling features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `build_opponent_model_source` | `build_opponent_model_source(team_games)` | Create team-game modeling source columns for opponent defensive rolling features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `build_opponent_position_model_source` | `build_opponent_position_model_source(source)` | Create opponent-position game rows used for shifted matchup rolling features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `create_column_decisions_v1` | `create_column_decisions_v1(output_path=None)` | Create the first explicit keep/drop/rolling decision table. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `write_source_csv` | `write_source_csv(df, output_path=None)` | Write the merged source table as CSV. | Path | 4. Data Preparation And Table Unification |
| `add_rest_features` | `add_rest_features(df)` | Add player rest features known before the game. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `build_modeling_table_last4_regular` | `build_modeling_table_last4_regular(source=None, season_starts=None)` | Build the leakage-safe modeling table with shifted rolling features. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `write_modeling_csv` | `write_modeling_csv(df, output_path=None)` | Write the modeling table as CSV. | Path | 4. Data Preparation And Table Unification |
| `has_target_components` | `has_target_components(columns)` | Return True when a table has all columns needed to create fantasy_points. | bool | 4. Data Preparation And Table Unification |
| `candidate_base_tables` | `candidate_base_tables(column_summary)` | Find raw tables that could support one row per player-game target creation. | pd.DataFrame | 4. Data Preparation And Table Unification |
| `classify_raw_columns` | `classify_raw_columns(column_summary)` | Create an initial leakage-focused raw-column classification table. | pd.DataFrame | 4. Data Preparation And Table Unification |
### Functions In 6. Exploratory Data Analysis

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `load_eda_data` | `load_eda_data(modeling_path=None, source_path=None)` | Load modeling rows plus current-game columns used only for EDA segmentation. | pd.DataFrame | 6. Exploratory Data Analysis |
| `add_eda_segments` | `add_eda_segments(df)` | Add position, minutes, target, and role buckets for EDA. | pd.DataFrame | 6. Exploratory Data Analysis |
| `dataset_summary` | `dataset_summary(df)` | Return high-level dataset health metrics. | pd.DataFrame | 6. Exploratory Data Analysis |
| `missingness_summary` | `missingness_summary(df)` | Return missingness and constant-column diagnostics. | pd.DataFrame | 6. Exploratory Data Analysis |
| `numeric_feature_columns` | `numeric_feature_columns(df)` | Numeric modeling features excluding IDs and target. | list[str] | 6. Exploratory Data Analysis |
| `current_game_diagnostic_correlations` | `current_game_diagnostic_correlations(df)` | Correlations for current-game EDA-only columns that are not model features. | pd.DataFrame | 6. Exploratory Data Analysis |
| `feature_correlations` | `feature_correlations(df)` | Correlation of numeric features with fantasy points. | pd.DataFrame | 6. Exploratory Data Analysis |
| `highly_correlated_pairs` | `highly_correlated_pairs(df, feature_rank, top_n_features=150, threshold=0.95)` | Find highly redundant numeric feature pairs among the strongest features. | pd.DataFrame | 6. Exploratory Data Analysis |
| `target_summary_by` | `target_summary_by(df, group_col)` | Summarize fantasy points by one segment. | pd.DataFrame | 6. Exploratory Data Analysis |
| `player_volume_summary` | `player_volume_summary(df)` | Player-level volume and target summary. | pd.DataFrame | 6. Exploratory Data Analysis |
| `outlier_tables` | `outlier_tables(df)` | Create outlier and special-case EDA tables. | dict[str, pd.DataFrame] | 6. Exploratory Data Analysis |
| `leakage_checks` | `leakage_checks(df)` | Return explicit leakage sanity checks. | pd.DataFrame | 6. Exploratory Data Analysis |
| `_savefig` | `_savefig(path)` | Save the current Matplotlib figure to disk and close it. | None | 6. Exploratory Data Analysis |
| `plot_target_distribution` | `plot_target_distribution(df, figures_dir=FIGURES_DIR)` | Save histograms that describe the fantasy-points target distribution. | None | 6. Exploratory Data Analysis |
| `plot_group_boxplots` | `plot_group_boxplots(df, figures_dir=FIGURES_DIR)` | Save boxplots comparing fantasy points across key player/game groups. | None | 6. Exploratory Data Analysis |
| `plot_feature_relationships` | `plot_feature_relationships(df, figures_dir=FIGURES_DIR, sample_size=50000)` | Save scatter plots for selected feature-target relationships. | None | 6. Exploratory Data Analysis |
| `plot_correlations` | `plot_correlations(feature_rank, figures_dir=FIGURES_DIR)` | Save a bar chart of the strongest feature-target correlations. | None | 6. Exploratory Data Analysis |
| `plot_missingness` | `plot_missingness(missing, figures_dir=FIGURES_DIR)` | Save a bar chart of the columns with the most missing values. | None | 6. Exploratory Data Analysis |
| `plot_correlation_heatmap` | `plot_correlation_heatmap(df, feature_rank, figures_dir=FIGURES_DIR)` | Save a heatmap for highly correlated feature groups. | None | 6. Exploratory Data Analysis |
| `run_full_eda` | `run_full_eda(modeling_path=None, source_path=None, tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR)` | Run full EDA and write report tables/figures. | dict[str, pd.DataFrame] | 6. Exploratory Data Analysis |
### Functions In 7. Feature Engineering

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `_has_any_prefix` | `_has_any_prefix(column, prefixes)` | Return whether a column starts with any prefix in a prefix list. | bool | 7. Feature Engineering |
| `_is_roll_family` | `_is_roll_family(column, base)` | Return whether a column belongs to a rolling feature family for a base stat. | bool | 7. Feature Engineering |
| `drop_reason` | `drop_reason(column)` | Return the first model-drop reason for a column, or None if it is eligible. | str \| None | 7. Feature Engineering |
| `eligible_model_features` | `eligible_model_features(columns)` | Return numeric model-eligible columns after first-pass drop policy. | list[str] | 7. Feature Engineering |
| `feature_drop_decisions` | `feature_drop_decisions(columns, eda_correlations=None)` | Create detailed keep/drop decisions for every modeling-table column. | pd.DataFrame | 7. Feature Engineering |
| `_family_features` | `_family_features(columns, bases)` | Collect eligible rolling-family features for selected base stats. | list[str] | 7. Feature Engineering |
| `_ordered_existing` | `_ordered_existing(columns, selected)` | Return selected features in modeling-table order while dropping missing columns. | list[str] | 7. Feature Engineering |
| `compact_feature_report` | `compact_feature_report(columns)` | Return the compact feature plan with availability and drop-policy details. | pd.DataFrame | 7. Feature Engineering |
| `build_compact_feature_sets` | `build_compact_feature_sets(columns)` | Build compact, lecturer-explainable feature sets around 40 columns. | dict[str, list[str]] | 7. Feature Engineering |
| `build_lineup_aware_feature_sets` | `build_lineup_aware_feature_sets(columns)` | Build named lineup-aware feature sets from available modeling columns. | dict[str, list[str]] | 7. Feature Engineering |
| `build_pre_lineup_feature_sets` | `build_pre_lineup_feature_sets(columns)` | Build feature sets for pre-lineup prediction. | dict[str, list[str]] | 7. Feature Engineering |
| `build_all_feature_sets` | `build_all_feature_sets(columns)` | Build both lineup-aware and pre-lineup feature sets. | dict[str, list[str]] | 7. Feature Engineering |
| `feature_set_summary` | `feature_set_summary(feature_sets, eda_correlations=None)` | Summarize feature-set sizes and correlation coverage. | pd.DataFrame | 7. Feature Engineering |
| `feature_set_membership_table` | `feature_set_membership_table(feature_sets)` | Return one row per feature per feature set. | pd.DataFrame | 7. Feature Engineering |
| `load_feature_set_inputs` | `load_feature_set_inputs(modeling_path=None, correlations_path=None)` | Load modeling schema and EDA correlations for feature-set generation. | tuple[list[str], pd.DataFrame] | 7. Feature Engineering |
| `write_feature_set_reports` | `write_feature_set_reports(modeling_path=None, correlations_path=None, output_dir=TABLES_DIR)` | Build and write feature-set and drop-decision reports. | dict[str, pd.DataFrame] | 7. Feature Engineering |
### Functions In 8. Chronological Train/Validation/Test Split

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `half_season_cutoff_date` | `half_season_cutoff_date(df, season_start=2024, date_col='game_date')` | Return the midpoint date by unique game dates for one season. | pd.Timestamp | 8. Chronological Train/Validation/Test Split |
| `assign_split_v1` | `assign_split_v1(df, date_col='game_date')` | Assign full-season chronological train/validation/test labels. | pd.Series | 8. Chronological Train/Validation/Test Split |
| `split_summary` | `split_summary(df, split_col=SPLIT_COL)` | Summarize rows, dates, seasons, games, and players by split. | pd.DataFrame | 8. Chronological Train/Validation/Test Split |
| `write_split_reports` | `write_split_reports(modeling_path=None, output_dir=TABLES_DIR, processed_dir=PROCESSED_DIR)` | Write split assignment and summary reports. | dict[str, pd.DataFrame] | 8. Chronological Train/Validation/Test Split |
### Functions In 10. Main Model Comparison

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `time_train_validation_test_split` | `time_train_validation_test_split(df, date_col='game_date', train_size=0.7, validation_size=0.15)` | Split rows chronologically into train, validation, and test sets. | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | 10. Main Model Comparison |
| `expanding_time_cv_splits` | `expanding_time_cv_splits(df, date_col='game_date', n_splits=4)` | Create expanding-window cross-validation splits over chronological rows. | list[tuple[np.ndarray, np.ndarray]] | 10. Main Model Comparison |
| `mean_baseline_prediction` | `mean_baseline_prediction(y_train, n_predictions)` | Predict the training target mean for each validation/test row. | np.ndarray | 10. Main Model Comparison |
| `regression_metrics` | `regression_metrics(y_true, y_pred)` | Return standard regression metrics for model comparison. | dict[str, float] | 10. Main Model Comparison |
| `load_feature_sets` | `load_feature_sets(path=None)` | Load feature-set membership report as an ordered dictionary. | dict[str, list[str]] | 10. Main Model Comparison |
| `load_validation_data` | `load_validation_data(modeling_path=None, split_path=None, feature_set_path=None, source_path=None)` | Load train/validation rows, feature columns, split labels, and EDA-only segment columns. | ValidationData | 10. Main Model Comparison |
| `add_validation_segments` | `add_validation_segments(df)` | Add validation error-analysis segments. | pd.DataFrame | 10. Main Model Comparison |
| `make_model` | `make_model(model_name)` | Return a model pipeline for numeric tabular features. | derived value | 10. Main Model Comparison |
| `_baseline_predictions` | `_baseline_predictions(model_name, y_train, validation)` | Generate validation predictions for the simple baseline models. | np.ndarray | 10. Main Model Comparison |
| `_validate_feature_frame` | `_validate_feature_frame(df, features)` | Check that all requested features exist and coerce them to numeric values. | pd.DataFrame | 10. Main Model Comparison |
| `run_one_experiment` | `run_one_experiment(data, feature_set, features, model_name)` | Fit/evaluate one model on train and validation only. | tuple[dict[str, object], pd.DataFrame] | 10. Main Model Comparison |
| `error_by_segment` | `error_by_segment(predictions)` | Summarize validation error by key segments. | pd.DataFrame | 10. Main Model Comparison |
| `feature_set_inputs` | `feature_set_inputs(feature_sets)` | Return one row per feature used in each model feature set. | pd.DataFrame | 10. Main Model Comparison |
| `_savefig` | `_savefig(path)` | Save the current Matplotlib figure to disk and close it. | None | 10. Main Model Comparison |
| `write_validation_plots` | `write_validation_plots(predictions, results, figures_dir=FIGURES_DIR)` | Write actual-vs-predicted and residual plots for top validation models. | None | 10. Main Model Comparison |
| `run_validation_experiments` | `run_validation_experiments(model_names=None, tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR)` | Run validation-only experiments and write report tables/figures. | dict[str, pd.DataFrame] | 10. Main Model Comparison |
### Functions In 10c. Controlled Model Comparison With Alternative Modeling Ideas

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `make_controlled_model` | `make_controlled_model(model_name)` | Create the final controlled comparison models with fixed seeds and bounded runtime. | derived value | 10c. Controlled Model Comparison With Alternative Modeling Ideas |
| `time_decay_weights` | `time_decay_weights(train_frame, half_life_days=180.0)` | Create train-only exponential time-decay weights relative to the latest train date. | np.ndarray | 10c. Controlled Model Comparison With Alternative Modeling Ideas |
| `fit_predict_controlled_model` | `fit_predict_controlled_model(model_name, x_train, y_train, x_validation, sample_weight=None)` | Fit one controlled model and return fitted model plus predictions. | derived value | 10c. Controlled Model Comparison With Alternative Modeling Ideas |
| `run_controlled_model_experiment` | `run_controlled_model_experiment(data, feature_sets, feature_set, model_name, use_time_decay=False)` | Evaluate one controlled model with train and validation overfitting diagnostics. | result tables / metrics | 10c. Controlled Model Comparison With Alternative Modeling Ideas |
### Functions In 12. Expanding Time Cross-Validation

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `make_expanding_time_cv_folds` | `make_expanding_time_cv_folds(data, date_col='game_date', n_splits=4, min_train_fraction=0.4)` | Create expanding chronological CV folds using contiguous validation date blocks. | pd.DataFrame | 12. Expanding Time Cross-Validation |
| `run_one_time_cv_experiment` | `run_one_time_cv_experiment(data, fold, feature_set, features, model_name, date_col='game_date')` | Runs one time cv experiment and returns/writes results. | dict[str, object] | 12. Expanding Time Cross-Validation |
| `summarize_time_cv_results` | `summarize_time_cv_results(results)` | Summarize fold-level CV metrics by feature set and model. | pd.DataFrame | 12. Expanding Time Cross-Validation |
| `write_time_cv_plot` | `write_time_cv_plot(results, figures_dir=FIGURES_DIR)` | Save a fold-by-fold MAE plot for the time CV experiment. | Path | 12. Expanding Time Cross-Validation |
| `write_time_cv_step_summary` | `write_time_cv_step_summary(path, fold_plan, summary)` | Write a short Markdown explanation of the time CV setup and top results. | None | 12. Expanding Time Cross-Validation |
### Functions In 13. PCA And Dimensionality Reduction

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `_numeric_feature_columns` | `_numeric_feature_columns(data, features)` | Return feature columns that can be converted to numeric values. | list[str] | 13. PCA And Dimensionality Reduction |
| `rank_pca_candidates` | `rank_pca_candidates(data, candidate_features, target_col=TARGET_COL)` | Rank candidate features by train-only absolute correlation with the target. | pd.DataFrame | 13. PCA And Dimensionality Reduction |
| `top_interfeature_correlations` | `top_interfeature_correlations(train, features, top_n=50)` | Report the highest absolute train-only correlations among PCA source features. | pd.DataFrame | 13. PCA And Dimensionality Reduction |
| `train_validation_metrics` | `train_validation_metrics(model, train_x, train_y, validation_x, validation_y)` | Fit a model and return train/validation diagnostics for overfitting checks. | dict[str, float] | 13. PCA And Dimensionality Reduction |
### Functions In 13b. Starter PCA Comparison

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `run_starter_pca_comparison` | `run_starter_pca_comparison(data, feature_sets, tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR)` | Compare valid starter feature sets with train-only PCA components. | pd.DataFrame | 13b. Starter PCA Comparison |
### Functions In 13c. Isolation Forest Anomaly Features

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `legal_isolation_features` | `legal_isolation_features(data, features)` | Return numeric legal features for anomaly detection, excluding direct current-game columns. | list[str] | 13c. Isolation Forest Anomaly Features |
| `assign_deciles_from_train` | `assign_deciles_from_train(train_scores, scores, n_bins=10)` | Assign score deciles using train-score thresholds only; higher decile means more anomalous. | np.ndarray | 13c. Isolation Forest Anomaly Features |
| `fit_isolation_variant` | `fit_isolation_variant(data, variant_name, source_feature_set)` | Fit Isolation Forest on train rows only and transform train/validation rows. | tuple[pd.DataFrame, dict[str, object]] | 13c. Isolation Forest Anomaly Features |
| `fit_predict_with_train_validation_metrics` | `fit_predict_with_train_validation_metrics(model, train_x, train_y, validation_x, validation_y)` | Fit a model and return train/validation metrics plus validation predictions. | derived value | 13c. Isolation Forest Anomaly Features |
| `run_isolation_forest_experiment` | `run_isolation_forest_experiment(data, feature_sets)` | Run Isolation Forest feature experiments without using test rows or current-game outcome columns. | dict[str, pd.DataFrame] | 13c. Isolation Forest Anomaly Features |
### Functions In 14. HGB Tuning

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `_dedupe` | `_dedupe(features)` | Return values in their first-seen order with duplicates removed. | list[str] | 14. HGB Tuning |
| `compact_base_features` | `compact_base_features()` | Return the planned pre-lineup compact feature list. | list[str] | 14. HGB Tuning |
| `build_refinement_feature_sets` | `build_refinement_feature_sets(available_features)` | Build compact refinement variants without changing the global feature-set reports. | dict[str, list[str]] | 14. HGB Tuning |
| `refinement_feature_plan` | `refinement_feature_plan(feature_sets)` | Return a long table describing which refinement features are in each variant. | pd.DataFrame | 14. HGB Tuning |
| `_refinement_group_for_feature` | `_refinement_group_for_feature(feature)` | Label each refinement feature by the candidate group it came from. | str | 14. HGB Tuning |
| `focus_segment_errors` | `focus_segment_errors(segment_errors)` | Keep the high-value error slices used to judge compact refinement. | pd.DataFrame | 14. HGB Tuning |
| `compact_refinement_summary` | `compact_refinement_summary(results, focus_errors)` | Create one row per fitted model with overall and focus-segment MAE. | pd.DataFrame | 14. HGB Tuning |
| `write_step_summary` | `write_step_summary(path, summary)` | Write a lecturer-friendly Markdown summary for compact refinement. | None | 14. HGB Tuning |
| `run_compact_refinement` | `run_compact_refinement(tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR)` | Run compact refinement experiments and write report artifacts. | dict[str, pd.DataFrame] | 14. HGB Tuning |
| `tuning_config_table` | `tuning_config_table(configs=None)` | Return the HGB tuning config table. | pd.DataFrame | 14. HGB Tuning |
| `make_hgb_pipeline` | `make_hgb_pipeline(config)` | Build a HistGradientBoosting pipeline from one config row. | Pipeline | 14. HGB Tuning |
| `run_tuning_config` | `run_tuning_config(data, features, config)` | Fit one HGB config on train and evaluate on validation only. | tuple[dict[str, object], pd.DataFrame] | 14. HGB Tuning |
| `tuning_summary` | `tuning_summary(results, focus_errors)` | Attach focus-segment MAE and deltas to tuning results. | pd.DataFrame | 14. HGB Tuning |
| `tuning_error_by_segment` | `tuning_error_by_segment(predictions)` | Summarize validation errors by segment while preserving tuning config. | pd.DataFrame | 14. HGB Tuning |
| `write_step_summary` | `write_step_summary(path, summary)` | Write a lecturer-friendly Markdown summary for HGB tuning. | None | 14. HGB Tuning |
| `run_hgb_tuning` | `run_hgb_tuning(tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR, configs=None)` | Run focused HGB tuning on the compact scoring feature set. | dict[str, pd.DataFrame] | 14. HGB Tuning |
### Functions In 14b. Valid 11-Feature Replication

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `replication_feature_plan` | `replication_feature_plan(available_columns)` | Return the 11-feature replication plan with leakage labels. | pd.DataFrame | 14b. Valid 11-Feature Replication |
| `_leakage_note` | `_leakage_note(feature_set, feature)` | Describe whether a replication feature is shifted history or pre-game context. | str | 14b. Valid 11-Feature Replication |
| `_source_concept` | `_source_concept(feature)` | Map a replication feature name to its basketball concept. | str | 14b. Valid 11-Feature Replication |
| `validate_replication_features` | `validate_replication_features(columns)` | Raise a clear error if any planned feature is missing. | None | 14b. Valid 11-Feature Replication |
| `add_missing_replication_features` | `add_missing_replication_features(data, modeling_path=None)` | Load planned replication-only columns that are excluded from normal feature membership. | pd.DataFrame | 14b. Valid 11-Feature Replication |
| `write_step_summary` | `write_step_summary(path, results, focus_errors)` | Write the lecturer-friendly summary for the valid 11-feature replication. | None | 14b. Valid 11-Feature Replication |
| `run_replication_11` | `run_replication_11(tables_dir=TABLES_DIR, figures_dir=FIGURES_DIR, modeling_path=None)` | Run the 11-feature replication models and write reports. | dict[str, pd.DataFrame] | 14b. Valid 11-Feature Replication |
### Functions In 16. Feature Importance And Interpretability

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `_experiment_normalize_id` | `_experiment_normalize_id(series)` | Normalize ids in this standalone notebook section before merges. | pd.Series | 16. Feature Importance And Interpretability |
| `_experiment_add_game_date_and_season` | `_experiment_add_game_date_and_season(df, date_col='gameDateTimeEst')` | Add parsed game dates and NBA season starts for the profile experiment. | pd.DataFrame | 16. Feature Importance And Interpretability |
| `_load_experiment_player_profile` | `_load_experiment_player_profile(raw_dir=RAW_DIR)` | Load static profile fields and create age/draft/size features. | pd.DataFrame | 16. Feature Importance And Interpretability |
| `_is_direct_player_fp_roll` | `_is_direct_player_fp_roll(feature)` | Return whether a feature is a player fantasy-points shortcut that this experiment excludes. | bool | 16. Feature Importance And Interpretability |
| `_is_target_derived_context` | `_is_target_derived_context(feature)` | Return whether a feature is derived from target components that this experiment excludes. | bool | 16. Feature Importance And Interpretability |
| `_is_allowed_existing_context_feature` | `_is_allowed_existing_context_feature(feature)` | Keep static/profile fields and rolling-10 non-target context; exclude FP shortcuts. | bool | 16. Feature Importance And Interpretability |
| `_load_extended_player_numeric_rolls` | `_load_extended_player_numeric_rolls(raw_dir=RAW_DIR)` | Build shifted rolling-10 features from numeric PlayerStatisticsExtended columns. | tuple[pd.DataFrame, pd.DataFrame] | 16. Feature Importance And Interpretability |
| `build_profile_extended_driver_10_table` | `build_profile_extended_driver_10_table(modeling_path=PROCESSED_DIR / 'modeling_table_last4_regular.csv', split_path=TABLES_DIR / 'modeling_rows_with_split_v1.csv')` | Create the no-player-FP-components rolling-10 interpretability table and feature plan. | tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame] | 16. Feature Importance And Interpretability |
| `run_profile_extended_driver_10_experiment` | `run_profile_extended_driver_10_experiment()` | Run the profile/extended-driver experiment and write its tables, predictions, and summary. | dict[str, pd.DataFrame] | 16. Feature Importance And Interpretability |
| `run_profile_extended_driver_10_importance` | `run_profile_extended_driver_10_importance(experiment_table=None, features=None, max_validation_rows=5000)` | Compute Ridge, random-forest, and permutation importance for the profile/extended-driver experiment. | dict[str, pd.DataFrame] | 16. Feature Importance And Interpretability |
### Functions In 17. Artifact Inventory And Final Report Links

| Function | Receives | Purpose | Output | Used In |
| --- | --- | --- | --- | --- |
| `_file_inventory` | `_file_inventory(folder, pattern)` | List generated files in one report folder with relative paths and file sizes. | pd.DataFrame | 17. Artifact Inventory And Final Report Links |
| `_markdown_table` | `_markdown_table(df)` | Render a small Markdown table without requiring the optional tabulate package. | str | 17. Artifact Inventory And Final Report Links |

## Important Output Files

- `runs/final/notebooks/NBA_ML_project.ipynb`: Final notebook.
- `runs/final/README.md`: GitHub README.
- `runs/final/requirements.txt`: Environment requirements.
- `runs/final/reports/tables/final_validation_ranking.csv`: Final validation ranking.
- `runs/final/reports/tables/final_test_evaluation.csv`: One-time 2025 test result.
- `runs/final/reports/tables/time_cv_summary.csv`: Time CV summary.
- `runs/final/reports/tables/leakage_diagnostic_actual_minutes.csv`: Invalid same-game minutes diagnostic.
- `runs/final/reports/step_summaries/notebooklm_project_brief.md`: Presentation/project brief.

## Main Lessons

1. Same-game actual minutes are powerful but invalid for pre-game prediction.
2. Shifted starter history and confirmed starter status are the strongest valid role signals.
3. Broad features help, but compact and starter-focused sets are easier to explain.
4. PCA is useful mainly when combined with the lineup-aware starter setup.
5. Isolation Forest anomaly features gave the best validation row, but the gain over starter PCA was small.
6. CatBoost on the clean lineup-aware starter feature set was selected for the controlled one-time test evaluation.

## Cleanup Notes Before Submission

- The notebook has historical numbering such as `17b`; execution order is correct, but labels can be cleaned.
- The cross-validation summary filename still says `step_15b_time_cross_validation.md` even though the notebook section is now Section 12.
- `profile_extended_driver_10` is technically correct but could be renamed to `profile_extended_stats_roll10_interpretability` for clarity.
- Do not upload Kaggle cache, processed modeling tables, or large prediction dumps to GitHub; the notebook recreates them.