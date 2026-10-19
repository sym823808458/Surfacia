:orphan:

Surfacia 使用手册
============================================================

分子表面描述符、预测建模与化学解释。

Surfacia 将结构、量子计算、全分子/表面/局部描述符连接到紧凑的 XGBoost 模型与
SHAP 分析。这里是 **分子表面**，不是固体催化界面的直接模拟。

从你的起点开始
------------------------------------------------------------

- 已有描述符表：:doc:`getting_started/quick_start`，下载案例直接跑，不需要量化软件。
- 只有结构：:doc:`user_guide/workflows`，先配置外部程序并确认局部区域。
- 已有输出：:doc:`examples/interpretation`，联合查看预测、SHAP 和结构。
- 正在升级：:doc:`migration`，核对版本和已移除的旧候选排序扩展。

.. image:: /_static/images/workflow_zh.svg
   :alt: 结构、分子表面、描述符、建模与解释
   :class: workflow-figure

本手册对应 **4.0.0 源码修订**；网页更新不会自动改变本机 pip 包。

`GitHub <https://github.com/sym823808458/Surfacia>`_ ·
`PyPI <https://pypi.org/project/surfacia/>`_ ·
`苏禹铭主页 <http://yumingsu.top/>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/installation
   getting_started/quick_start
   getting_started/basic_concepts
   user_guide/workflows
   descriptors/index
   user_guide/results
   examples/index
   commands/index
   api/index
   integrations/index
   user_guide/best_practices
   user_guide/troubleshooting
   migration
   citation
   changelog
   contributing
   license
