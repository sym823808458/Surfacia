安装与版本核对
============================================================

使用独立环境，优先选择 Python 3.10–3.12；包声明最低 3.9。
新 Python 不一定已有全部二进制依赖。完整量化流程推荐 Linux。

.. code-block:: bash

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install "surfacia[viz]"
   python -c "import surfacia; print(surfacia.__version__)"
   surfacia --help

PyPI 可能落后于源码手册。若输出 3.0.3，阅读 :doc:`../migration` 或安装源码：

.. code-block:: bash

   git clone https://github.com/sym823808458/Surfacia.git
   cd Surfacia
   python -m pip install -e ".[viz]"

ML 依赖范围为 ``xgboost>=2.1.4,<3`` 与 ``shap>=0.48,<0.49``。
完整流程另需 xTB、Gaussian（g16、formchk）与 Multiwfn，单独安装并遵守许可。
它们不包含在 pip 包里，已有描述符案例不需要它们。交互解释用 ``viz`` 扩展。
下一步：:doc:`quick_start`。
