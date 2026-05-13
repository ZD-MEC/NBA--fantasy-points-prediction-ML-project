# NBA Fantasy Points Prediction: Stakeholder Report Source

This project predicts one NBA player-game fantasy score before the game outcome is known.

Final selected model: `catboost` with `lineup_aware_starter_flag_roll10`.

- Validation MAE: `7.644`
- 2025 test MAE: `7.671`
- Current-game minutes correlation with fantasy points: `0.804`
- Rolling last-10 fantasy baseline validation MAE: `7.986`
- Best valid validation experiment: `Best Isolation Forest anomaly model`, MAE `7.580`

Conclusion: use the selected lineup-aware CatBoost model when confirmed starters are available. Keep below-7 MAE models labeled as leakage diagnostics because they use same-game minutes.
