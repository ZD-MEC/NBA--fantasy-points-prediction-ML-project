# Presentation Slide Outline

1. Project objective and fantasy-points target
2. Dataset and row grain: player-game regular-season rows
3. Course workflow and chronological split
4. Data preparation and 2021 team-context repair
5. EDA: target distribution, positions, starters, minutes, outliers
6. Leakage rules and why current-game minutes are invalid
7. Feature engineering: shifted rolling stats, starter history, opponent context, momentum
8. Baselines and first valid model comparisons
9. Alternative modeling approach: strong below-7 result, but leakage diagnostic only
10. Valid corrected model ideas: XGBoost, LightGBM, CatBoost, ExtraTrees, RandomForest
11. PCA and Isolation Forest experiments
12. Final validation ranking
13. One-time 2025 test result: `catboost` / `lineup_aware_starter_flag_roll10` MAE 7.6712
14. Feature importance and interpretation
15. Main conclusion and next improvements
