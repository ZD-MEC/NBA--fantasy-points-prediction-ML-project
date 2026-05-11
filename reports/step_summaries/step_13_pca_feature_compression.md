# Step 13 PCA Feature Compression

PCA used `100` train-selected source features from `pre_lineup_player_team_opp` after removing `pre_lineup_compact_40_plus_scoring` features and filtering train missingness above `40%`.

Best PCA validation MAE: `7.7480` from `compact_scoring_plus_pca_30` with `hist_gradient_boosting`.

All ranking, missingness filtering, imputation, scaling, and PCA fitting used train rows only. Validation rows were transformed only. Test rows were not used.
