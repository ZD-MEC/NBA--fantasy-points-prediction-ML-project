# Step 10: 11-Feature Replication

## Goal

This step tests whether an 11-feature model can perform well using only valid pre-game features.

## Feature Sets

- `replication_11_safe`: uses shifted historical `starter_minutes_roll_10`.
- `replication_11_safe_recent_starter`: uses shifted historical `starter_minutes_roll_3`.

Direct current-game `starter_minutes` is not included in this comparison.

## Best Result

```text
feature_set = replication_11_safe_recent_starter
model       = hist_gradient_boosting
features    = 11
MAE         = 7.8180
RMSE        = 9.9750
R2          = 0.5525
```

## Best Safe HGB

```text
best safe HGB MAE = 7.8180
```

## Outputs

```text
reports/tables/replication_11_feature_plan.csv
reports/tables/replication_11_model_results.csv
reports/tables/replication_11_focus_segments.csv
```
