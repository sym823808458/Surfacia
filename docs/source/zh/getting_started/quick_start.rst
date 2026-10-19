从现有描述符开始
============================================================

从 :doc:`../examples/descriptor_table` 下载完整 CSV 到新目录。
特征和目标由 RDKit 计算，属于操作教程，**不是 Surfacia 量子表面 benchmark**。

输入格式
------------------------------------------------------------

列顺序：``Sample Name``、数值描述符、``smiles``、``target``。
列名大小写一致，样本名唯一。此监督回归必须有目标列；编号/结果不要混入描述符。

.. code-block:: text

   Sample Name,MolWt,TPSA,HBD,HBA,RotB,Rings,smiles,target

.. code-block:: bash

   surfacia ml-analysis -i descriptors.csv --manual --manual-features Full --test-samples "6,12,18,24,30,36" --epoch 32 --cores 1 --train-test-split 0.85 --no-generate-fitting

``--manual --manual-features Full`` 在验证前固定特征。
``--test-samples`` 匹配 **Sample Name 的值**，不是行号或范围。
``--epoch`` 是重复随机划分次数；``--train-test-split`` 是建模池内训练比例，
不是永久测试集比例。

输入文件旁的 ``Full_Feature_Analysis_*`` 包含明细 CSV 和预测图。
解释 R² 前先读 :doc:`../user_guide/results`。
