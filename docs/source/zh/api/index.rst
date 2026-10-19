Python 接口
============================================================

从实际模块导入：

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import ChemMLWorkflow

   result = ChemMLWorkflow.run_analysis(
       mode="manual", data_file="descriptors.csv",
       test_sample_names=["6", "12", "18", "24", "30", "36"],
       features="Full", epoch=32, core_num=1,
       train_test_split=0.85, generate_fitting=False,
   )

API 并行参数为 ``core_num``，CLI 为 ``--cores``；可将 ``Full`` 换成预先固定特征列表。
评价边界见 :doc:`../user_guide/results`。交互解释入口为
``surfacia.visualization.interactive_shap_viz.interactive_shap_viz_main``，参数是
``csv_path``、``xyz_path``、可选 ``test_csv_path``、``api_key``、
``skip_surface_gen``、``port`` 与 ``host``。旧排序 API 已移除：:doc:`../migration`。
