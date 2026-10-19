Read model outputs
============================================================

Main artifacts
------------------------------------------------------------

- ``Training_Set_Detailed_*.csv``: modeling-pool features and SHAP.
- ``Feature_<name>``: descriptor value.
- ``SHAP_<name>`` / ``SHAP_Std_<name>``: attribution mean/dispersion across available validation appearances.
- ``Test_Set_Detailed_*.csv``: explicit held-out samples, if provided.
- ``Realtest_Pred`` / ``Realtest_Pred_Std``: original ensemble prediction and between-model dispersion.
- ``Target``: supplied target, not a new measurement.

The engine uses repeated random splits. With too few repeats some samples may
never appear in validation; legacy aggregation can then contain zero-filled
values. Check validation coverage before reporting metrics. Ensemble spread is
not a calibrated confidence interval.

Automatic feature selection is exploratory, not a nested-CV implementation of
the separately frozen manuscript benchmark. Predefine features, or evaluate the
entire selection procedure inside training folds for scientific comparisons.
Keep permanent tests out of selection, explanation ranking and parameter choice.

SHAP values are model attributions, not measured effects or candidate scores.
Open actual files with :doc:`../examples/interpretation`.
