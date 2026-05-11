# Step 13c Isolation Forest Anomaly Features

Isolation Forest was fit on train rows only and transformed validation rows only. The 2025 test split was not used.

Direct current-game `starter_minutes`, actual current-game `numMinutes`, and the target were excluded from anomaly inputs.

Best reference validation MAE in this section: `7.5815` from `lineup_aware_starter_flag_roll10_best_pca_reference` with `hist_gradient_boosting`.

Best anomaly-feature validation MAE: `7.5797` from `lineup_aware_starter_flag_roll10_best_pca_plus_isolation` with `hist_gradient_boosting`.

If anomaly deciles show higher MAE in high-score groups, the feature can be useful as a prediction reliability indicator even when it does not improve the main validation MAE.
