排错与运行环境
============================================================


.. code-block:: bash

   python -c "import sys,surfacia,xgboost,shap; print(sys.executable); print(surfacia.__version__, xgboost.__version__, shap.__version__)"
   python -m pip check
   surfacia ml-analysis --help

- 找不到命令：激活安装它的环境。
- 缺 openbabel/RDKit：在相同环境安装声明的依赖。
- ``could not convert string to float: '[-3.xxxE0]'``：核对 :doc:`../getting_started/installation` 中的兼容范围。
- 特征不是数值：``Sample Name`` 与 ``smiles,target`` 之间只放数值描述符。
- 测试样本未找到：使用真实样本名，不是行号/范围。
- 缺 xTB/Gaussian/Multiwfn：单独安装外部程序。
- 没有三维表面：提供对应真实结构/PDB。
- SSH 界面：绑定 localhost 并转发端口。
- 旧参数被拒绝：以 :doc:`../commands/index` 为准；旧的泛化算法/分类参数并未实现。

反馈问题时提供版本、命令、脱敏表头和报错。
