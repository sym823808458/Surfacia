案例一：描述符建模与 SHAP
============================================================

本例列出 36 个常见分子的 SMILES。六个 RDKit 特征是分子量、TPSA、氢键供受体数、
可旋转键数与环数；目标为 **RDKit 计算的 Crippen logP**。
不含实验或尚未发表的标签。

这是计算目标的操作教程，**不是 Surfacia 量子表面 benchmark，也不是实验 logP 精度声明**。

下载与复现
------------------------------------------------------------

- :download:`描述符表 </_static/examples/descriptors.csv>`
- :download:`SMILES </_static/examples/molecules.csv>`
- :download:`生成脚本 </_static/examples/make_data.py>`
- :download:`运行环境与指标 </_static/examples/run_summary.json>`

按 :doc:`../getting_started/installation` 安装，CSV 保存为 ``descriptors.csv``：

.. code-block:: bash

   surfacia ml-analysis -i descriptors.csv --manual --manual-features Full --test-samples "6,12,18,24,30,36" --epoch 32 --cores 1 --train-test-split 0.85 --no-generate-fitting

六个特征固定，显式留出六个样本。32 次重复用于操作演示，不是论文冻结协议。
记录的输出由当前源码使用该命令实际运行生成。

实际运行：建模池验证 R² = 0.766、MAE = 0.267；留出 R² = 0.076、MAE = 0.307。六样本的测试 R² 不稳定且较弱，本例不构成泛化证明。每个建模样本至少进入验证集两次。

.. figure:: /_static/images/demo_predictions.png
   :alt: RDKit 计算 logP 的实际验证与留出预测

   实际运行输出，目标为 RDKit 计算性质。

.. figure:: /_static/images/demo_shap.png
   :alt: 六个 RDKit 特征的平均绝对 SHAP

   模型归因，不是已验证的化学机理。

- :download:`建模池明细 </_static/examples/Training_Set_Detailed_demo.csv>`
- :download:`留出明细 </_static/examples/Test_Set_Detailed_demo.csv>`

评价边界见 :doc:`../user_guide/results`，交互查看见 :doc:`interpretation`。
