Commands Reference
==================

This section keeps detailed command syntax and parameters.

.. toctree::
   :maxdepth: 2

   workflow
   ml_analysis
   shap_viz
   utilities

Quick Use Patterns
------------------

.. code-block:: bash

   # End-to-end run
   surfacia workflow -i molecules.csv --test-samples "1,3"

   # Resume from interrupted workflow
   surfacia workflow -i molecules.csv --resume --test-samples "1,3"

   # ML only
   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv --test-samples "1,2,3"

   # SHAP visualization
   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_files/

Command Selection
-----------------

- ``workflow``: full pipeline and resume control
- ``ml_analysis``: model training, feature selection, and evaluation
- ``shap_viz``: interpretation dashboard
- ``utilities``: helper tools (draw/view/rerun/info)
