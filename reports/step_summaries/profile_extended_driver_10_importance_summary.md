# Profile Extended Driver 10 Importance Summary

Permutation importance was computed on 5,000 validation rows with 3 repeats.

Top 10 permutation features:

                        feature  importance  importance_std
     ext_fieldGoalsMade_roll_10    1.071791        0.049314
        ext_possessions_roll_10    0.460222        0.024145
             numMinutes_roll_10    0.220860        0.006194
ext_fieldGoalsAttempted_roll_10    0.106316        0.013820
  ext_reboundsDefensive_roll_10    0.103037        0.009454
           days_since_last_game    0.085099        0.008783
       ext_doubleDouble_roll_10    0.038262        0.000582
   playerImpactEstimate_roll_10    0.036424        0.003627
   ext_assistPercentage_roll_10    0.031612        0.007376
       ext_foulsAgainst_roll_10    0.031159        0.001054

Use permutation importance as the main interpretation table because it measures validation MAE impact directly. Ridge and random forest importance are supporting views.
