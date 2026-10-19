案例三：将解释关联到结构
============================================================

从 :doc:`descriptor_table` 下载两个明细 CSV。只有表时创建空目录 ``structures``；
已有结构时使用相应编号的结构/PDB 文件。二维图由 SMILES 生成，三维表面需要真实文件。

.. code-block:: bash

   surfacia shap-viz -i Training_Set_Detailed_demo.csv -x ./structures --test-csv Test_Set_Detailed_demo.csv --skip-surface-gen --host 127.0.0.1 --port 8052

打开 http://127.0.0.1:8052 ，选择特征，检查其数值、SHAP 贡献、结构和局部区域。
留出点显示为 **原始测试集**，不附加候选排序修正。

查看图不需要密钥。可选 AI 助手与本地 SHAP 分开，本教程不会调用；启用前确认
发送给第三方的数据。SSH 时绑定 localhost 并转发 8052 端口。
高归因表示模型影响，不是因果或发现保证。
