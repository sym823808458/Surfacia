Machine Learning API
====================

The machine-learning layer handles compact model construction, feature selection, validation, and interpretability analysis.

What Belongs Here
-----------------

- feature selection
- model training and evaluation
- cross-validation logic
- SHAP-based interpretation

Typical Responsibilities
------------------------

- reduce large feature matrices to compact subsets
- fit predictive models
- compare model behavior across splits
- generate outputs that remain chemically interpretable

When to Use the ML API
----------------------

Use this layer when you want to:

- train models directly from feature tables
- test alternative feature-selection settings
- integrate Surfacia features into custom ML code
- inspect prediction and explanation outputs programmatically

SPES API
--------

Use ``surfacia.ml.spes`` when you want to generate SPES outputs from existing detailed training and test CSV files.

Primary functions:

- ``resolve_spes_parameters(...)``: choose default SPES parameters for Mode 1, Mode 2, or Mode 3.
- ``build_spes_overlay(...)``: return an in-memory SPES dataframe and metadata dictionary.
- ``write_spes_artifacts(...)``: write ``SPES_Test_Set_Detailed_*.csv`` and ``SPES_Metadata_*.json``.

Minimal example:

.. code-block:: python

   import pandas as pd
   from surfacia.ml.spes import write_spes_artifacts

   training_df = pd.read_csv("Training_Set_Detailed.csv")
   test_df = pd.read_csv("Test_Set_Detailed.csv")

   write_spes_artifacts(
       training_df=training_df,
       test_df=test_df,
       output_dir="spes_out",
       base_name="manual_run",
       mode_hint="Mode3",
   )

For user-facing interpretation, see :doc:`../user_guide/spes`.
