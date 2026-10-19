Case 3 · Read an explanation
============================================================

Download both detailed CSVs from :doc:`descriptor_table`. Create an empty
``structures`` directory for the descriptor-only demo, or use matching numbered
structure/PDB files. 2D drawings use SMILES; 3D surfaces need actual surface files.

.. code-block:: bash

   surfacia shap-viz -i Training_Set_Detailed_demo.csv -x ./structures --test-csv Test_Set_Detailed_demo.csv --skip-surface-gen --host 127.0.0.1 --port 8052

Open http://127.0.0.1:8052 . Select a feature and inspect its value, SHAP
contribution, molecular structure and local region. Held-out points are the
**original test set**, without ranking correction.

No API key is needed for plots. The optional AI assistant is separate from local
SHAP and is not called by this tutorial. Review data sent to third-party services
before enabling it. Over SSH, bind to localhost and forward port 8052.
High attribution means model influence, not causality or guaranteed discovery.
