# Step 13b Starter PCA Comparison

This section compares two valid starter model families with and without train-only PCA features:

- `pre_lineup_starter_history_roll10`
- `lineup_aware_starter_flag_roll10`

Direct current-game `starter_minutes` is not used as a model feature.

Best starter PCA/reference validation MAE: `7.5815` from `lineup_aware_starter_flag_roll10_plus_pca_30` with `hist_gradient_boosting`.

PCA ranking, imputation, scaling, and fitting used train rows only. Validation rows were transformed only. Test rows were not used.
