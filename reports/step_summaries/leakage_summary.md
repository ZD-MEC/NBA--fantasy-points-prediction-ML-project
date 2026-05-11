# Leakage Summary

`starter_minutes = current_is_starter * numMinutes`. Because `numMinutes` is actual current-game minutes, direct `starter_minutes` is same-game information and is excluded from model comparisons.

Valid alternatives are shifted historical features such as `starter_minutes_roll_3`, `starter_minutes_roll_10`, and `starter_minutes_share_roll_10`.
