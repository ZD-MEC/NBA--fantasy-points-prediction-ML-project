# Step 09: Focused HGB Tuning

## Goal

This step tuned only the best current feature set:

```text
pre_lineup_compact_40_plus_scoring
```

The goal was to improve validation MAE without adding more features or using the untouched 2025 test set.

## Why Focused Tuning

The previous step found that the 45-feature scoring compact model beat the 471-feature broad model. Therefore, the next useful question is whether HistGradientBoosting settings can extract more signal from the same 45 features.

We did not tune every model and every feature set because that would increase overfitting risk and make the experiment harder to explain.

## Search Space

The tuning varied:

- learning rate
- number of boosting iterations
- tree size through `max_leaf_nodes`
- L2 regularization
- minimum samples per leaf
- internal early stopping

The current HGB settings were included as `reference_current_hgb`.

## Best Result

```text
config_name = no_early_stop_more_trees
features    = 45
MAE         = 7.7700
RMSE        = 9.9330
R2          = 0.5563
```

## Reference Comparison

```text
reference MAE = 7.7818
best MAE      = 7.7700
improvement   = 0.0118
```

## Best Parameters

```text
learning_rate       = 0.03
max_iter            = 260
max_leaf_nodes      = 31
l2_regularization   = 0.3
min_samples_leaf    = 40
early_stopping      = False
```

## Interpretation

This step uses validation only. The 2025 test period remains untouched.

The full tuning table and focus-segment errors are saved in:

```text
reports/tables/hgb_tuning_summary.csv
reports/tables/hgb_tuning_focus_segments.csv
```

## Next Step

If the best tuned config improves validation meaningfully, promote it as the current final validation candidate.

Then run model interpretation on the tuned 45-feature model before touching the test set.
