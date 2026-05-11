# End-To-End Summary

This final run uses NBA seasons `2021` through `2025`.

Split policy:

- Train: full seasons `2021`, `2022`, `2023`
- Validation: full season `2024`
- Test: full season `2025`

Lowest validation MAE among valid lineup-aware or pre-lineup models: `Best Isolation Forest anomaly model` / `lineup_aware_starter_flag_roll10_best_pca_plus_isolation`.

Best lineup-aware validation MAE: 7.5797 from `lineup_aware_starter_flag_roll10_best_pca_plus_isolation`.

Best pre-lineup validation MAE: 7.7315 from `pre_lineup_player_team_opp`.

Best compact explainable candidate: `pre_lineup_compact_40_plus_scoring` with tuned HGB, MAE 7.7700.

Best Isolation Forest anomaly model: `lineup_aware_starter_flag_roll10_best_pca_plus_isolation` with MAE 7.5797.

The 2025 test split remains untouched for model choice.
