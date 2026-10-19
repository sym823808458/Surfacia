升级到 4.0.0
===============

安装后运行 ``python -c "import surfacia; print(surfacia.__version__)"`` 核对版本。

4.0.0 聚焦分子表面描述符、预测与解释。旧脚本如依赖已移除的候选排序接口，需删去相关参数。
留出样本通过 ``--test-csv`` 读取原始 ``Test_Set_Detailed_*.csv``。
普通 SHAP、原始预测与表面分析仍可使用。

具体安装见 :doc:`getting_started/installation`，输出见 :doc:`user_guide/results`。
旧发行包及完整变更记录保留在 Git 历史中。
