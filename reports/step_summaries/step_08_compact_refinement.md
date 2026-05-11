# Step 08: Compact Feature Refinement

## Goal

This step refined the compact 40-feature model with small, basketball-explainable candidate groups.

The goal was to improve the compact model without returning to a broad 400+ feature approach.

## Variants Tested

- `pre_lineup_compact_40`: original compact baseline.
- `pre_lineup_compact_40_plus_role`: adds role and minute-momentum features.
- `pre_lineup_compact_40_plus_scoring`: adds scoring and usage-ceiling features.
- `pre_lineup_compact_40_plus_ceiling`: adds all-around ceiling features.
- `pre_lineup_compact_40_plus_matchup`: adds matchup refinements.
- `pre_lineup_compact_40_plus_all_candidates`: adds every candidate group.
- `pre_lineup_player_team_opp`: broad 471-feature reference.

## Why This And Not More Broad Features

The previous step showed that 40 features nearly matched 471 features. Therefore, the next rational test is controlled feature swaps, not adding every available column.

Each candidate group is interpretable:

- role: likely minutes and starter-like usage
- scoring: scoring ceiling and usage movement
- ceiling: rebounds, assists, impact, efficiency
- matchup: position and opponent context

## Best Result

```text
feature_set = pre_lineup_player_team_opp
model       = hist_gradient_boosting
features    = 486
MAE         = 7.7315
RMSE        = 9.9008
R2          = 0.5591
```

## Key Comparison

```text
original compact HGB MAE = 7.7832
broad reference HGB MAE  = 7.7315
best refinement MAE      = 7.7315
```

## Interpretation

The winning variant should be judged against two goals:

1. Does it improve overall MAE?
2. Does it help the hard ceiling segments such as 35+ minutes and 50+ fantasy points?

The full numeric comparison is saved in:

```text
reports/tables/compact_refinement_summary.csv
reports/tables/compact_refinement_error_focus_segments.csv
```

## Next Step

If a small variant beats the compact baseline, promote it as the new compact candidate.

If no variant improves overall MAE, keep `pre_lineup_compact_40` as the main explainable model and move to model tuning or a cleaner target-specific ceiling approach.
