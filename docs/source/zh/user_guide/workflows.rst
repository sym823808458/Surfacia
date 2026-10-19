完整量子化学流程
============================================================

配置好的本地/Linux 流程包括：

1. SMILES 转三维。
2. xTB 优化。
3. Gaussian 输入与波函数。
4. Multiwfn 性质与分子表面。
5. 全分子/局部描述符。
6. 固定特征建模或探索筛选。
7. 原始预测与 SHAP 检查。

完成 :doc:`../getting_started/installation` 并使用独立目录；监督流程需要已知目标。

.. code-block:: bash

   surfacia workflow -i molecules.csv --mode 1 --element S --test-samples "6,12" --epoch 64 --cores 4

这是 **配置示例**，不是本手册执行过的量化任务。长任务前检查电荷、自旋、方法、
基组与溶剂。实际参数见 :doc:`../commands/workflow`。``--resume`` 根据文件检测继续点，
不证明科学设置正确或已收敛。

结构输入见 :doc:`../examples/surface_workflow`；已有描述符从
:doc:`../getting_started/quick_start` 开始。
