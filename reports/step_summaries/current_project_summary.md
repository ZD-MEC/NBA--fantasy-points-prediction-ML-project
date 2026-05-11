# Current Project Summary

## Goal

Predict player-game fantasy basketball points with a time-aware supervised learning workflow.

Target formula:

`fantasy_points = points + 1.2 * reboundsTotal + 1.5 * assists + 3 * steals + 3 * blocks - turnovers`

The prediction row grain is one player in one NBA regular-season game. Current-game box-score outcomes are not valid model features. Outcome fields can only enter through shifted historical rolling features.

## Course-Aligned Workflow

This notebook follows the course workflow with data collection, preparation, cleansing checks, EDA, feature engineering, train/validation/test split, modeling, validation comparison, dimensionality reduction, tuning, interpretability, and final recommendation.

Intentional deviations from a generic supervised-learning template:

- Chronological split replaces random split because future games must not train models for earlier games.
- Expanding time cross-validation replaces random CV for the same reason.
- The 2025 test season is held back until the model choice is frozen.
- Deployment is not included because this is an academic modeling notebook, not a production app.

## Data Scope And Split

This final run uses NBA seasons `2021`, `2022`, `2023`, `2024`, and `2025`.

Split column: `split_2021_train_2024_validation_2025_test`

| split | rows | unique_games | unique_players | date_min | date_max | seasons | target_mean | target_median | target_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 76937 | 3624 | 818 | 2021-10-19 19:30:00 | 2024-04-14 15:30:00 | 2021, 2022, 2023 | 21.458608991772486 | 19.4 | 14.866903934325746 |
| validation | 26162 | 1223 | 569 | 2024-10-22 19:30:00 | 2025-04-13 15:30:00 | 2024 | 21.722498279948017 | 20.0 | 14.911719753737543 |
| test | 26651 | 1230 | 582 | 2025-10-21 19:30:00 | 2026-04-12 20:30:00 | 2025 | 21.62010431128288 | 20.0 | 14.4995619628602 |

The 2021 team-context issue was repaired by filling missing `TeamStatisticsExtended.gameType` values from `Games.csv` by `gameId` before regular-season filtering.

## Main Outputs

Modeling table rows: `129750`  
Modeling table columns: `576`

## Final Model Comparison

| model_label | feature_set | validity | n_features | mae | rmse | r2 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best Isolation Forest anomaly model | lineup_aware_starter_flag_roll10_best_pca_plus_isolation | lineup-aware | 44 | 7.579672486347945 | 9.694710672391565 | 0.5773013707223377 | Isolation Forest anomaly features; variant lineup_aware_anomaly. |
| Best valid starter PCA model | lineup_aware_starter_flag_roll10_plus_pca_30 | lineup-aware | 41 | 7.581493274826162 | 9.69404735724241 | 0.5773592110894976 | Starter model plus 30 PCA components. |
| Best valid starter-focused model | lineup_aware_starter_flag_roll10 | lineup-aware | 11 | 7.655891244591493 | 9.78251609850373 | 0.5696098961274687 | Starter model family included as a main comparison group. |
| Lineup-aware starter-flag HGB | lineup_aware_starter_flag_roll10 | lineup-aware | 11 | 7.655891244591493 | 9.78251609850373 | 0.5696098961274687 | Uses current_is_starter plus starter_minutes_roll_10, not actual current-game minutes. |
| Broad valid HGB reference | pre_lineup_player_team_opp | pre-lineup | 486 | 7.731508771004425 | 9.90076184766369 | 0.5591423689619857 | Broad benchmark with player/team/opponent shifted rolling features. |
| Best valid PCA compact model | compact_scoring_plus_pca_30 | pre-lineup | 75 | 7.7479845764132085 | 9.90753503243398 | 0.5585389746749083 | 30 PCA components from unused broad features. |
| Best valid tuned compact model | pre_lineup_compact_40_plus_scoring | pre-lineup | 45 | 7.769961117050048 | 9.933004652325202 | 0.5562663010115096 | Tuned HGB config: no_early_stop_more_trees |
| Compact scoring before tuning | pre_lineup_compact_40_plus_scoring | pre-lineup | 45 | 7.781769330266231 | 9.947804761804573 | 0.5549429954908913 | Compact feature-refinement candidate before tuning. |
| Best valid 11-feature replication | replication_11_safe_recent_starter | pre-lineup | 11 | 7.817988642824836 | 9.975004516803118 | 0.5525058766678708 | Uses shifted starter-minutes history. |
| Pre-lineup starter-history HGB | pre_lineup_starter_history_roll10 | pre-lineup | 10 | 7.909683099771941 | 10.08782620329792 | 0.542325919972285 | Uses shifted starter history including starter_minutes_roll_10. |

## PCA Compression Results

PCA candidates came from broad legal pre-lineup features that were not already in the compact scoring feature set. Candidate ranking, missingness filtering, imputation, scaling, and PCA fitting used train rows only.

| experiment | model | feature_set | n_features | n_pca_source_features | n_pca_components | pca_explained_variance_ratio_sum | train_mae | validation_mae | mae_gap | train_rmse | validation_rmse | rmse_gap | train_r2 | validation_r2 | r2_gap | overfitting_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_hgb_reference | hist_gradient_boosting | reference | 484 | 0 | 0 |  | 7.354056684040197 | 7.731508771004425 | 0.3774520869642277 | 9.36811243445491 | 9.90076184766369 | 0.5326494132087802 | 0.6029284571174601 | 0.5591423689619857 | 0.04378608815547447 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_30 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 75 | 100 | 30 | 0.9079994369538437 | 7.469243809604122 | 7.7479845764132085 | 0.27874076680908644 | 9.520423649268581 | 9.90753503243398 | 0.38711138316539895 | 0.5899119411128868 | 0.5585389746749083 | 0.03137296643797849 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_41 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 86 | 100 | 41 | 0.9519369260011116 | 7.4527995388218296 | 7.749075544425801 | 0.29627600560397127 | 9.49948552094648 | 9.903476470259719 | 0.4039909493132381 | 0.5917137589017605 | 0.5589005843051491 | 0.03281317459661137 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_30 | ridge | pre_lineup_compact_40_plus_scoring | 75 | 100 | 30 | 0.9079994369538437 | 7.712555665864955 | 7.751898844148483 | 0.03934317828352807 | 9.858114542658827 | 9.900597752263597 | 0.04248320960476981 | 0.5603042230342953 | 0.5591569824050635 | 0.0011472406292317716 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_20 | ridge | pre_lineup_compact_40_plus_scoring | 65 | 100 | 20 | 0.8322079469857311 | 7.715314622457488 | 7.752297510132782 | 0.03698288767529423 | 9.861522027432825 | 9.903931241489992 | 0.04240921405716769 | 0.5600002063590956 | 0.5588600724848105 | 0.001140133874285132 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_10 | ridge | pre_lineup_compact_40_plus_scoring | 55 | 100 | 10 | 0.6679441695299557 | 7.723319356233508 | 7.754614491818273 | 0.03129513558476571 | 9.87128984992491 | 9.90828820019102 | 0.03699835026610998 | 0.5591281364350428 | 0.5584718526617678 | 0.0006562837732749793 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_41 | ridge | pre_lineup_compact_40_plus_scoring | 86 | 100 | 41 | 0.9519369260011116 | 7.709114018568696 | 7.7563515401846965 | 0.047237521616000144 | 9.854068120405795 | 9.902713867732782 | 0.048645747326986566 | 0.5606651094125519 | 0.5589685141030725 | 0.0016965953094794095 | Small gap with improved validation MAE = useful compression. |
| compact_scoring_plus_pca_20 | hist_gradient_boosting | pre_lineup_compact_40_plus_scoring | 65 | 100 | 20 | 0.8322079469857311 | 7.513088592794321 | 7.762020431927739 | 0.24893183913341765 | 9.574866020647757 | 9.922236342527315 | 0.3473703218795574 | 0.5852083684326335 | 0.5572278775058952 | 0.027980490926738266 | Small gap with improved validation MAE = useful compression. |

## Starter-Focused Models

Starter models are now a main comparison family:

- `pre_lineup_starter_history_roll10`: valid before lineup news because it uses shifted starter history, including `starter_minutes_roll_10`.
- `lineup_aware_starter_flag_roll10`: valid once starting lineup is known because it uses `current_is_starter` and shifted `starter_minutes_roll_10`.
- Direct current-game `starter_minutes` is not used in model comparisons.

| feature_set | model | n_features | train_rows | validation_rows | fit_predict_seconds | mae | rmse | r2 | mean_error | median_absolute_error | validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 11 | 76937 | 26162 | 1.252 | 7.655891244591493 | 9.78251609850373 | 0.5696098961274687 | 0.14553330634638179 | 6.297060858119103 | lineup-aware |
| lineup_aware_starter_flag_roll10 | ridge | 11 | 76937 | 26162 | 0.122 | 7.721617482900054 | 9.865872158289582 | 0.562244005886912 | -0.08559834754068661 | 6.378065935346564 | lineup-aware |
| pre_lineup_starter_history_roll10 | hist_gradient_boosting | 10 | 76937 | 26162 | 1.035 | 7.909683099771941 | 10.08782620329792 | 0.542325919972285 | 0.10157575682773268 | 6.558809891794746 | pre-lineup |
| pre_lineup_starter_history_roll10 | ridge | 10 | 76937 | 26162 | 0.136 | 7.952467684422498 | 10.150775276804861 | 0.5365962319497654 | -0.07954774282324902 | 6.600303136221067 | pre-lineup |

## Starter PCA Results

| experiment | base_feature_set | model | n_features | n_pca_source_features | n_pca_components | pca_explained_variance_ratio_sum | train_mae | validation_mae | mae_gap | train_rmse | validation_rmse | rmse_gap | train_r2 | validation_r2 | r2_gap | overfitting_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10_plus_pca_30 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 41 | 100 | 30 | 0.927620804405783 | 7.409026122774365 | 7.581493274826162 | 0.17246715205179708 | 9.45261955956563 | 9.69404735724241 | 0.24142779767677958 | 0.5957324031379916 | 0.5773592110894976 | 0.01837319204849397 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_37 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 48 | 100 | 37 | 0.9528697012504477 | 7.349468007307149 | 7.581563107492612 | 0.23209510018546275 | 9.374060950534165 | 9.68915997920353 | 0.31509902866936557 | 0.602424036109489 | 0.57778526318792 | 0.02463877292156902 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_20 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 31 | 100 | 20 | 0.8606097578558353 | 7.427530168043712 | 7.584784803444978 | 0.15725463540126583 | 9.475665518019406 | 9.698203961640502 | 0.22253844362109554 | 0.5937587509637259 | 0.5769966943560505 | 0.01676205660767538 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_30 | lineup_aware_starter_flag_roll10 | ridge | 41 | 100 | 30 | 0.927620804405783 | 7.637865405504943 | 7.589702912183666 | -0.048162493321276756 | 9.766861469274561 | 9.718277069597349 | -0.048584399677212176 | 0.5684067635143725 | 0.5752438382029276 | -0.0068370746885551 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_37 | lineup_aware_starter_flag_roll10 | ridge | 48 | 100 | 37 | 0.9528697012504477 | 7.631899772370708 | 7.591356684616971 | -0.04054308775373627 | 9.758038931804252 | 9.712857212536985 | -0.04518171926726744 | 0.5691861393083275 | 0.5757174768373352 | -0.006531337529007697 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_10 | lineup_aware_starter_flag_roll10 | hist_gradient_boosting | 21 | 100 | 10 | 0.7216487766689973 | 7.43555056061255 | 7.595782658736675 | 0.16023209812412453 | 9.489775006285074 | 9.709210111483932 | 0.21943510519885834 | 0.5925480447479925 | 0.5760360464785237 | 0.01651199826946881 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_20 | lineup_aware_starter_flag_roll10 | ridge | 31 | 100 | 20 | 0.8606097578558353 | 7.6454014984669945 | 7.5976050718016115 | -0.047796426665382974 | 9.77665716502994 | 9.725850885091512 | -0.05080627993842768 | 0.5675405945476624 | 0.5745815236074349 | -0.007040929059772494 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_plus_pca_10 | lineup_aware_starter_flag_roll10 | ridge | 21 | 100 | 10 | 0.7216487766689973 | 7.66912052013937 | 7.615704981525715 | -0.05341553861365522 | 9.806692890418828 | 9.751323156278769 | -0.055369734140059634 | 0.5648793199620066 | 0.5723502401323168 | -0.007470920170310258 | Small gap with improved validation MAE = useful compression. |

## Isolation Forest Anomaly Results

| experiment | base_feature_set | anomaly_variant | model | n_base_features | n_features | train_rows | validation_rows | train_mae | validation_mae | mae_gap | train_rmse | validation_rmse | rmse_gap | train_r2 | validation_r2 | r2_gap | overfitting_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineup_aware_starter_flag_roll10_best_pca_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | hist_gradient_boosting | 11 | 44 | 76937 | 26162 | 7.359589118096379 | 7.579672486347945 | 0.22008336825156682 | 9.389131533462812 | 9.694710672391565 | 0.30557913892875277 | 0.6011446508004211 | 0.5773013707223377 | 0.02384328007808345 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_best_pca_reference | lineup_aware_starter_flag_roll10 | none | hist_gradient_boosting | 11 | 41 | 76937 | 26162 | 7.409026122774365 | 7.581493274826162 | 0.17246715205179708 | 9.45261955956563 | 9.69404735724241 | 0.24142779767677958 | 0.5957324031379916 | 0.5773592110894976 | 0.01837319204849397 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_best_pca_reference | lineup_aware_starter_flag_roll10 | none | ridge | 11 | 41 | 76937 | 26162 | 7.637865405504943 | 7.589702912183666 | -0.048162493321276756 | 9.766861469274561 | 9.718277069597349 | -0.048584399677212176 | 0.5684067635143725 | 0.5752438382029276 | -0.0068370746885551 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_best_pca_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | ridge | 11 | 44 | 76937 | 26162 | 7.637594371998586 | 7.5900752752505625 | -0.04751909674802324 | 9.766735759921426 | 9.718879206102484 | -0.04785655381894216 | 0.5684178735229636 | 0.5751912014829936 | -0.006773327960030051 | Small gap with improved validation MAE = useful compression. |
| lineup_aware_starter_flag_roll10_reference | lineup_aware_starter_flag_roll10 | none | hist_gradient_boosting | 11 | 11 | 76937 | 26162 | 7.621969636346413 | 7.655891244591493 | 0.03392160824508039 | 9.724880243968183 | 9.78251609850373 | 0.05763585453554754 | 0.5721090526500594 | 0.5696098961274687 | 0.0024991565225906953 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| lineup_aware_starter_flag_roll10_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | hist_gradient_boosting | 11 | 14 | 76937 | 26162 | 7.634611930545788 | 7.662170433138331 | 0.027558502592543554 | 9.7383019079664 | 9.781375630274027 | 0.04307372230762674 | 0.5709271416244401 | 0.5697102420145412 | 0.0012168996098989027 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| lineup_aware_starter_flag_roll10_reference | lineup_aware_starter_flag_roll10 | none | ridge | 11 | 11 | 76937 | 26162 | 7.794193485773586 | 7.721617482900054 | -0.07257600287353139 | 9.961254193949047 | 9.865872158289582 | -0.09538203565946546 | 0.5510555366945682 | 0.562244005886912 | -0.011188469192343842 | Small gap with worse validation MAE = underpowered or unhelpful features. |
| lineup_aware_starter_flag_roll10_plus_isolation | lineup_aware_starter_flag_roll10 | lineup_aware_anomaly | ridge | 11 | 14 | 76937 | 26162 | 7.7929543967049355 | 7.723145275641869 | -0.06980912106306647 | 9.95997584835813 | 9.868341395920451 | -0.09163445243767931 | 0.551170756994915 | 0.5620248546910267 | -0.010854097696111675 | Small gap with worse validation MAE = underpowered or unhelpful features. |

## Libraries Used

- `pandas`: DataFrame loading, joins, groupby summaries, CSV outputs.
- `numpy`: Numeric arrays, missing values, vectorized target and metric calculations.
- `matplotlib`: Static EDA and model diagnostic figures.
- `scikit-learn`: Ridge, HistGradientBoostingRegressor, RandomForestRegressor, IsolationForest, SimpleImputer, StandardScaler, PCA, permutation importance.
- `pathlib`: Stable filesystem paths.
- `dataclasses`: Small ValidationData container.

## Important Parameters

- Seasons: `[2021, 2022, 2023, 2024, 2025]`
- Target column: `fantasy_points`
- Split column: `split_2021_train_2024_validation_2025_test`
- PCA top candidate cap: `100`
- PCA missingness cap: `40%`
- PCA component grid: `[10, 20, 30]` plus the component count needed for 95% train explained variance
- Isolation Forest contamination: `5%`
- Isolation Forest estimators: `200`
- Random state: `42`
- Main metrics: MAE, RMSE, R2

## Thinking Process

The notebook first builds a legal modeling table with shifted rolling features because the core risk is same-game leakage. It then compares broad and compact feature sets to check whether many team/opponent features help beyond player history. Starter uncertainty is handled as its own model family because role and minutes are the main basketball-specific uncertainty. PCA is added as a controlled compression experiment for many legal unused features. Isolation Forest is added as a train-only anomaly/reliability experiment. HGB tuning is applied only after the feature families are compared. Direct current-game `starter_minutes` is excluded from model comparisons and is used only to create shifted rolling history.

## Artifact Inventory

- Processed datasets: `3`
- Report tables: `89`
- Report figures: `68`
- Step summaries: `16`

See `reports/tables/artifact_inventory_final.csv` for the full file list.
