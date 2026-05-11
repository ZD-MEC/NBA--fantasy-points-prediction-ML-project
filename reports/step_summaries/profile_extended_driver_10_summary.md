# Profile Extended Driver 10 Summary

Purpose: interpret non-FP drivers of future fantasy points, not chase the best validation MAE.

Feature set: `profile_extended_driver_10`

Rows: 129,750

Features kept: 144

Rules applied:

- One rolling window only: 10 games.
- Player history is shifted by one game before rolling.
- Excluded player-level rolling fantasy points and direct FP components: assists, blocks, fantasy_points, points, reboundsTotal, steals, turnovers.
- Added profile fields: age at game date, height, weight, BMI, draft year, draft round, draft number, years since draft, and missingness flags.
- Kept extended player-stat rolling features and valid team/opponent/role context.

Best model in this interpretability experiment: `hist_gradient_boosting`

Validation MAE: 7.8685

Interpretation note: if this model trails the FP-rolling models, that is expected. The next analytical step is to inspect Ridge/RF importance and permutation importance to identify which non-target drivers carry the most signal.
