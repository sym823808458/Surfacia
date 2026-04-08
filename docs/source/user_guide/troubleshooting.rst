Troubleshooting
===============

This page summarizes common issues that users may encounter while running Surfacia.

Common Problems
---------------

**Missing external software**
  Check that Gaussian, Multiwfn, and any required geometry-optimization tools are installed and callable.

**Interrupted calculations**
  Prefer resume-friendly workflows and verify that intermediate files were created correctly.

**Unexpectedly weak model performance**
  Check sample size, descriptor quality, target consistency, and whether the chosen analysis mode matches the chemistry.

**Interpretation feels too vague**
  Inspect compact retained descriptors first, then revisit whether Mode 1 or Mode 2 would better reflect the problem structure.

Machine Learning Compatibility (Important)
------------------------------------------

If you see errors similar to:

.. code-block:: text

   could not convert string to float: '[-3.1971428E0]'

this is typically a version compatibility issue between ``xgboost`` and ``shap`` in your environment, not a problem with your ``FinalFull`` CSV.

Verified working combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xgboost==2.1.4
   shap==0.48.0

Quick check:

.. code-block:: bash

   conda activate surfacia
   python -c "import xgboost, shap; print('xgboost', xgboost.__version__, 'shap', shap.__version__)"

If your versions do not match, fix with:

.. code-block:: bash

   pip install --force-reinstall "xgboost==2.1.4" "shap==0.48.0"

CLI Input Path Pitfalls
-----------------------

**Error**: ``Input file '' not found!``

Cause: ``$finalfull`` was not defined in the current shell session.

Use explicit file names or define the variable first:

.. code-block:: bash

   finalfull=$(ls -1t FinalFull*.csv | head -n 1)
   surfacia ml-analysis -i "$finalfull" --test-samples "1,2,3"

**Error**: ``surfacia: command not found``

Cause: Surfacia is not installed in the currently active conda environment.

Fix:

.. code-block:: bash

   conda activate <your_env>
   pip install surfacia

Remote Linux (HPC) Step-7 Re-run Pattern
-----------------------------------------

When Step 1-6 have completed and only ML analysis needs re-running:

.. code-block:: bash

   conda activate surfacia
   cd /home/<user>/Surfacia_runs/<run_id>/Surfacia_3.0_<timestamp>
   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv \
     --max-features 5 --stepreg-runs 3 \
     --train-test-split 0.85 --epoch 64 --cores 8 \
     --test-samples "1,2,3"

For a faster smoke test:

.. code-block:: bash

   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv \
     --max-features 1 --stepreg-runs 1 --epoch 8 --cores 4 \
     --train-test-split 0.85 --test-samples "1,2,3"

Related Example
---------------

For a full real-world replay of the same troubleshooting sequence, see:

- :doc:`../examples/mode3_top20_remote_debug`
