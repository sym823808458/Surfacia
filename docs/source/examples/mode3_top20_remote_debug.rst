Mode3 Top-20 Remote Debug (Linux, No SLURM)
===========================================

This example records a real end-to-end validation run on ``2026-04-08`` for Surfacia Mode 3 with 20 molecules, focusing on reproducible debugging.

Goal
----

- Verify that the PyPI package can be installed and used directly in a fresh conda environment.
- Reproduce and fix the main runtime errors observed during ML analysis.

Run Context
-----------

- Platform: remote Linux (SSH session)
- Workflow scope: Step-7 (``ml-analysis``) validation on ``FinalFull_Mode3_20_168.csv``
- Execution mode: direct command line (no SLURM)

Environment Setup
-----------------

.. code-block:: bash

   conda create -n surfacia_mode3_test_20260408 python=3.10 -y
   conda activate surfacia_mode3_test_20260408
   pip install surfacia

Version sanity check:

.. code-block:: bash

   python -c "import xgboost, shap; print('xgboost', xgboost.__version__, 'shap', shap.__version__)"

If needed, force compatible versions:

.. code-block:: bash

   pip install --force-reinstall "xgboost==2.1.4" "shap==0.48.0"

Step-7 Command (Validated)
--------------------------

.. code-block:: bash

   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv \
     --max-features 1 --stepreg-runs 1 \
     --train-test-split 0.85 --epoch 32 --cores 8 \
     --test-samples "1,2,3"

Observed Errors and Fixes
-------------------------

**Error 1**: ``Input file '' not found!``

- Cause: shell variable (for example ``$finalfull``) was not defined.
- Fix: use an explicit filename or define variable first.

**Error 2**: ``Input file 'FinalFull_Mode3_20_168' not found!``

- Cause: missing file extension.
- Fix: use ``FinalFull_Mode3_20_168.csv``.

**Error 3**: ``surfacia: command not found``

- Cause: package not installed in current conda environment.
- Fix: activate target env and run ``pip install surfacia``.

**Error 4**: ``could not convert string to float: '[-3.2828572E0]'`` (or similar ``[-3.xxxE0]``)

- Cause: environment dependency mismatch (``xgboost`` / ``shap`` combination).
- Fix: pin to validated versions:

  .. code-block:: bash

     pip install --force-reinstall "xgboost==2.1.4" "shap==0.48.0"

Results Checklist
-----------------

After fixes, the run should proceed through:

1. baseline analysis
2. stepwise regression
3. output generation under ``Workflow_Analysis_<timestamp>/``

This example is a practical deployment validation template for new users and new environments.
