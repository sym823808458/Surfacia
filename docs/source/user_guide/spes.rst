SPES Candidate Prioritization
=============================

SPES is a post-processing layer for ranking external or held-out candidates after Surfacia has already selected and interpreted a regression model. It is designed for discovery workflows where the main question is which test-set samples deserve attention first.

What SPES Is
------------

SPES uses the training-set SHAP landscape to estimate whether a test-set sample sits in a high-potential region of feature space. It then adds a conservative ranking score on top of the selected model prediction.

SPES is not a new target value, and it is not a replacement model. Treat it as a candidate-prioritization score.

When It Is Generated
--------------------

When ML analysis includes a test set, Surfacia writes SPES outputs automatically:

.. code-block:: bash

   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv --test-samples "1,2,3"

Typical output files:

- ``Training_Set_Detailed_*.csv``
- ``Test_Set_Detailed_*.csv``
- ``SPES_Test_Set_Detailed_*.csv``
- ``SPES_Metadata_*.json``

How To View It
--------------

Open the SHAP dashboard with both the raw test set and SPES layer:

.. code-block:: bash

   surfacia shap-viz \
     -i Training_Set_Detailed.csv \
     -x ./xyz_files \
     --test-csv Test_Set_Detailed.csv \
     --spes-csv SPES_Test_Set_Detailed.csv

In the dashboard, use the external overlay control to switch between no overlay, raw test-set points, and the SPES layer.

Important Columns
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Meaning
   * - ``SPES_Score``
     - Ranking score used to prioritize candidates.
   * - ``SPES_Percentile``
     - Percentile of the test sample in the training-derived SPES landscape.
   * - ``SPES_Delta``
     - Difference between ``SPES_Score`` and the original prediction.
   * - ``SPES_Rank``
     - Rank after sorting ``SPES_Score`` from high to low.

How To Interpret It
-------------------

Start from the selected model prediction, then use SPES to sort candidates. A high ``SPES_Score`` with a high ``SPES_Percentile`` means the sample is predicted well and lies in a SHAP landscape region that Surfacia considers promising.

Use SPES cautiously when the test sample is far outside the training chemistry space. In that case, inspect the SHAP plot and descriptor values before treating the rank as chemically meaningful.

Python Usage
------------

For custom scripts, call ``write_spes_artifacts`` directly:

.. code-block:: python

   import pandas as pd
   from surfacia.ml.spes import write_spes_artifacts

   training_df = pd.read_csv("Training_Set_Detailed.csv")
   test_df = pd.read_csv("Test_Set_Detailed.csv")

   paths = write_spes_artifacts(
       training_df=training_df,
       test_df=test_df,
       output_dir="spes_out",
       base_name="mode3_demo",
       mode_hint="Mode3",
   )

   print(paths["csv"])
   print(paths["json"])

Required Input Columns
----------------------

The training table must contain ``Feature_*`` columns, matching ``SHAP_*`` columns, and ``Target``.

The test table must contain the same ``Feature_*`` and ``SHAP_*`` columns. For the original prediction, Surfacia looks for ``Realtest_Pred`` first, then ``Predicted``, then ``Target``.
