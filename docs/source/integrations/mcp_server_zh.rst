Surfacia MCP Server 从零启动手册
================================

这份手册面向第一次接触 Surfacia MCP server 的用户，目标不是解释 MCP 的全部背景，而是让你尽快把服务跑起来、知道先测什么、出问题先查哪里。

它对应的正式英文说明见 :doc:`mcp_server`。

一、这是什么
-------------

``surfacia-mcp`` 是 Surfacia 的 stdio MCP server。它把 Surfacia 的核心流程拆成一组可调用工具，让支持 MCP 的客户端或 Agent 可以直接调用：

- 环境检查
- 工作目录状态识别
- 基准描述符矩阵生成
- Surfacia 特征提取
- 机器学习分析
- XTB / Gaussian / Multiwfn 重计算阶段
- SHAP 可视化启动
- 全流程总编排

你可以把它理解成：

“把原本面向命令行用户的 Surfacia，额外包装成一组给 Agent 使用的工具接口。”

二、代码现在放在哪里
--------------------

当前实现位于：

- ``surfacia/mcp/server.py``
- ``surfacia/mcp/tooling.py``
- ``surfacia/mcp/utils.py``
- ``surfacia/mcp/shap_launcher.py``

另外还有一份设计说明：

- ``SURFACIA_MCP_DESIGN.md``

三、启动前你需要准备什么
------------------------

至少要准备两层环境。

**Python 层**

- 你当前用于 Surfacia 的 Python 环境
- 已安装 Surfacia 本体
- 能导入 ``rdkit``、``shap``、``dash``、``xgboost`` 等依赖

**外部程序层**

- ``xtb``
- ``g16``
- ``formchk``
- ``Multiwfn_noGUI`` 或你实际使用的 Multiwfn 命令

如果这些外部程序没有在 PATH 里，MCP server 本身可能能启动，但对应工具会在真正执行时失败。

四、如何安装
------------

建议在 Surfacia 源码目录下做可编辑安装：

.. code-block:: powershell

   cd D:\YumingSu\Papers\Surfacia\code251222\Surfacia
   pip install -e .

安装完成后，会得到一个命令：

.. code-block:: powershell

   surfacia-mcp

如果你的环境里没有正确暴露这个命令，也可以直接用模块方式启动：

.. code-block:: powershell

   python -m surfacia.mcp.server --log-level INFO

五、怎么启动
------------

最直接的启动命令：

.. code-block:: powershell

   surfacia-mcp --log-level INFO

或者：

.. code-block:: powershell

   python -m surfacia.mcp.server --log-level INFO

.. admonition:: 注意
   :class: note

   这个命令不是给人手工一条条输入工具调用参数的。它是标准的 MCP stdio server，正常情况下应该由 MCP 客户端来启动并连接。

六、第一次不要直接跑全流程
--------------------------

最稳的做法不是一上来就调用 ``surfacia_run_full_pipeline``，而是按下面顺序一步一步测：

1. ``surfacia_check_environment``
2. ``surfacia_detect_workflow_state``
3. ``surfacia_generate_benchmark_matrices`` 或 ``surfacia_extract_features``
4. ``surfacia_run_ml_analysis``
5. 再去测 ``xtb / Gaussian / Multiwfn`` 这些重工具
6. 最后再测 ``surfacia_launch_shap_visualizer`` 或 ``surfacia_run_full_pipeline``

这样你能很快区分：

- 是 Python 依赖问题
- 是外部程序没装好
- 是路径给错了
- 还是化学计算本身真的失败了

七、最先应该调试什么
--------------------

**第一步：环境检查**

你首先应该确认 ``surfacia_check_environment`` 的结果。

它会告诉你两件事：

- Python 模块是否可用
- 外部命令是否可用

如果这里不过，后面大概率都不稳。

**第二步：状态识别**

然后跑 ``surfacia_detect_workflow_state``，看某个工作目录现在停在哪一阶段。

这个工具特别适合下面几种情况：

- 你有一个已经算到一半的目录
- 你不确定目录里是 ``xyz``、``com``、``fchk`` 还是 ``FinalFull``
- 你想知道下一步该调哪个 MCP tool

八、现在有哪些工具
------------------

当前版本已经实现的工具有：

- ``surfacia_check_environment``
- ``surfacia_detect_workflow_state``
- ``surfacia_generate_benchmark_matrices``
- ``surfacia_smi_to_xyz``
- ``surfacia_run_xtb_opt``
- ``surfacia_generate_gaussian_inputs``
- ``surfacia_run_gaussian_jobs``
- ``surfacia_rerun_failed_gaussian_jobs``
- ``surfacia_run_multiwfn_analysis``
- ``surfacia_extract_features``
- ``surfacia_run_ml_analysis``
- ``surfacia_launch_shap_visualizer``
- ``surfacia_run_full_pipeline``

九、怎么理解返回结果
--------------------

每个工具返回的最外层结构都是统一的，大致长这样：

.. code-block:: json

   {
     "ok": true,
     "tool": "surfacia_extract_features",
     "summary": "Completed Surfacia Mode 3 feature extraction.",
     "working_dir": "D:/.../case_dir",
     "artifacts": {},
     "metrics": {},
     "logs": {}
   }

可以这样理解：

- ``ok``：工具成功还是失败
- ``summary``：一句话结果摘要
- ``working_dir``：这次执行主要关联的目录
- ``artifacts``：生成的关键文件
- ``metrics``：结构化结果指标
- ``logs``：调试时需要看的标准输出或错误输出

十、SHAP 可视化怎么用
----------------------

``surfacia_launch_shap_visualizer`` 不会阻塞 MCP server 本身，它会后台启动一个 Dash 应用。

它适合在下面这种状态使用：

- 你已经有 ``Training_Set_Detailed*.csv``
- 工作目录里有对应的 ``xyz`` 或 ``fchk`` 文件
- 你希望保留可视化能力，但不让 MCP server 因为 Dash 进程卡住

启动后，日志会写到：

.. code-block:: text

   surfacia_mcp_logs/
   ├── shap_visualizer.stdout.log
   └── shap_visualizer.stderr.log

如果仪表板起不来，先看这两个文件。

十一、现在还没完全做完的地方
----------------------------

这版已经能用，但还不是“最终生产版”。

目前还建议继续做的事情有：

1. 给 Gaussian / Multiwfn 这类长任务加 ``job_id`` 和状态查询
2. 把部分同步重任务改成异步可恢复
3. 给每个 MCP tool 增加自动化测试
4. 把 SHAP / SLPS 的结构化结果再细化
5. 继续减少对“当前工作目录”的隐式依赖

十二、如果你现在就要开始联调
----------------------------

建议你按下面这个顺序实际测试：

.. code-block:: text

   A. 启动 surfacia-mcp
   B. 用客户端调用 surfacia_check_environment
   C. 选择一个已有案例目录，调用 surfacia_detect_workflow_state
   D. 如果已有 FullOption，调用 surfacia_extract_features
   E. 如果已有 FinalFull，调用 surfacia_run_ml_analysis
   F. 最后再测 Gaussian/Multiwfn 和 full pipeline

这样效率最高，也最容易定位问题。
