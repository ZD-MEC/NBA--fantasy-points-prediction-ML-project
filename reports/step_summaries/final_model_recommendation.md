# Final Model Recommendation

Use `pre_lineup_player_team_opp` as the main pre-lineup benchmark.

If confirmed starting lineup is available, `lineup_aware_starter_flag_roll10_best_pca_plus_isolation` is the strongest valid lineup-aware benchmark.

Keep the tuned compact scoring HistGradientBoosting model as the strongest compact explanatory candidate.

Do not touch the 2025 test split until the model choice is frozen.

Best lineup-aware validation MAE: 7.5797.

Best pre-lineup validation MAE: 7.7315.

Tuned compact validation MAE: 7.7700.

Best Isolation Forest anomaly MAE: 7.5797.
