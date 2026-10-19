案例二：含硫分子表面流程
============================================================

公开结构集包括甲硫醇、乙硫醇、二甲硫醚、二甲基二硫、噻吩、苯甲硫醚。
它示范元素中心区域选择，不预测催化性能；不编造 FE、产率、吸附能或活性标签。

- :download:`仅含结构的 CSV </_static/examples/sulfur_molecules.csv>`

.. figure:: /_static/images/sulfur_molecules.png
   :alt: 根据 SMILES 绘制的六个含硫结构

   真实分子结构，不是计算表面。

核对分子状态/构象、硫区域可比性和 ESP/ALIE/LEAE 设置；检查局部趋势是否可能
由全分子变化解释。先准备结构：

.. code-block:: bash

   surfacia smi2xyz -i sulfur_molecules.csv

再按 :doc:`../user_guide/workflows` 与 :doc:`../commands/index` 完成量化阶段。
Multiwfn 产出所需性质表后，提取命令形式为：

.. code-block:: bash

   surfacia extract-features -i FullOption.csv --mode 1 --element S

文件名替换为 **实际输出**。结构案例不继续监督建模；正式研究需真实可追溯目标
和冻结验证设计。本手册没有重跑量化，也不将虚构表面/性能图当作真实结果。
