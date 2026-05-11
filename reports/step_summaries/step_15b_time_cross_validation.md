# Step 15b: Expanding Time Cross-Validation

## Goal

This section checks whether selected pre-game models are stable across multiple earlier chronological validation blocks.

It uses only rows from the existing `train` split. The official validation split and the held-out 2025 test split are not used here.

## Fold Design

- Number of folds: 4
- First train date: 2021-10-19 19:30:00
- Last CV validation date: 2024-04-14 15:30:00
- Fold type: expanding train window with contiguous future validation blocks

## Best Mean CV Result

```text
feature_set = pre_lineup_player_team_opp
model       = hist_gradient_boosting
features    = 486
mean MAE    = 7.7063
std MAE     = 0.0896
mean RMSE   = 9.8657
mean R2     = 0.5716
```

## Interpretation Rule

This result is supporting evidence only. The final model recommendation should still be judged against the official chronological validation split unless the project explicitly changes its model-selection protocol.
