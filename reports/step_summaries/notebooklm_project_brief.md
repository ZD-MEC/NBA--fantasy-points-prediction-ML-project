# NotebookLM Project Brief

## Project Goal
Predict NBA player-game fantasy points using a clean time-aware supervised-learning workflow.

## Key Story
The project started with broad data exploration, leakage checks, rolling feature engineering, and several model families. An alternative notebook showed below-7 MAE, but the improvement came from direct current-game `starter_minutes`, which is not available before prediction time. The final project keeps that result as a leakage diagnostic and evaluates the valid ideas under the clean split.

## Data And Split
Train seasons are 2021-2023, validation is 2024, and test is 2025. The test season is used once after model selection.

## Main Experiments
- Baselines and broad feature models
- Starter-role models
- Expanding time cross-validation
- PCA feature compression
- Isolation Forest anomaly features
- XGBoost, LightGBM, CatBoost, ExtraTrees, and RandomForest controlled comparisons

## Best Validation Result
Best Isolation Forest anomaly model using `lineup_aware_starter_flag_roll10_best_pca_plus_isolation` with MAE 7.5797.

## Final Test Result
Selected model `catboost` on `lineup_aware_starter_flag_roll10` reached test MAE 7.6712, RMSE 9.7909, and R2 0.5440.

## Presentation Message
The most important lesson is that sports prediction is highly sensitive to role/minutes information. Actual minutes create excellent but invalid results; shifted starter history and lineup-aware starter flags provide valid alternatives.
