Python interfaces
============================================================

Import from the actual modules:

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import ChemMLWorkflow

   result = ChemMLWorkflow.run_analysis(
       mode="manual", data_file="descriptors.csv",
       test_sample_names=["6", "12", "18", "24", "30", "36"],
       features="Full", epoch=32, core_num=1,
       train_test_split=0.85, generate_fitting=False,
   )

The API uses ``core_num``; the CLI uses ``--cores``. Replace ``Full`` with a
predefined feature list when appropriate. Evaluation limits: :doc:`../user_guide/results`.
The visualizer entry is
``surfacia.visualization.interactive_shap_viz.interactive_shap_viz_main`` with
``csv_path``, ``xyz_path``, optional ``test_csv_path``, ``api_key``,
``skip_surface_gen``, ``port`` and ``host``. The former ranking API was removed:
:doc:`../migration`.
