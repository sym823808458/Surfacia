工作流：始终围绕化学问题
====================================

先确定局部区域的化学含义。一个描述符的 **区域、物理性质和聚合方式** 都明确，归因才有可读的对象。

1. 选择区域
-----------

- **元素模式**：有硫位点先验的添加剂系列，选硫作为局部锚点。
- **片段模式**：有共同催化核心的有机催化剂系列，预先定义片段。
- **自动局部模式**：没有明确局部先验时，探索局部环境。

先确定区域，再查看测试结果。局部变量与全分子变量一起考虑，保留尺寸、形状和暴露程度的信息。
三种模式见 :doc:`../descriptors/index`。

2. 结构 → 波函数 → 表面 → 描述符
----------------------------------

从 SMILES 准备结构，xTB 优化，Gaussian 生成波函数，Multiwfn 提取 ESP/ALIE/LEAE，
再整理全分子与选定区域的描述符。所有阶段沿用同一组样本名。

.. code-block:: bash

   surfacia smi2xyz -i sulfur_molecules.csv
   # 完成量化和 Multiwfn 阶段后，使用实际生成的 FullOption.csv：
   surfacia extract-features -i FullOption.csv --mode 1 --element S

六个公开硫分子输入见 :doc:`../examples/surface_workflow`。
记录电荷、自旋、构象、方法、溶剂、密度等值面与单位；计算失败不能默认为零。

文章中可追溯的 Mode 1/2 生产设置为 GFN2-xTB/ALPB 水，随后
PBE1PBE/6-311G*、GD3、SMD 水的单点计算，再在 0.010 a.u. 密度等值面进行 Multiwfn 分析。
这是 **文章设置**，不表示 CLI 默认参数自动复现该协议。运行前检查实际输入。

3. 描述符 → 模型 → 解释
-----------------------

具备可追溯目标和固定留出集后，对预先指定的特征块建模。
测试目标不参与特征筛选或解释排名。已配置环境中的整合入口：

.. code-block:: bash

   surfacia workflow -i molecules.csv --mode 1 --element S --test-samples "6,12" --epoch 64 --cores 4

此命令需要外部量化软件与已知目标，本手册没有重新执行量化链。
新体系建议先逐阶段检查。``--resume`` 只检测文件存在，不验证科学设置。
实际选项见 :doc:`../commands/workflow`。

4. 归因 → 化学证据
-------------------

把描述符定义、跨分子趋势和原始预测一起阅读，再提出可以检验这种关系的结构改动。
具体示范见 :doc:`../examples/interpretation`。

公开 CLI 是 XGBoost 与重复随机划分流程。文章中的共享 128 次划分、双模型
OOF TreeSHAP/DeepSHAP、扰动检验和盲化多 LLM 评分属于另行冻结的研究分析。
``workflow`` 不会自动执行整套评分，也不能直接等同于论文 benchmark 复现。
