Installation and versions
============================================================

Use an isolated environment. Python 3.10–3.12 is a practical choice for the pinned
ML stack; the declared minimum is 3.9. New Python releases may lack binary wheels.
Linux is recommended for the full quantum workflow.

.. code-block:: bash

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install "surfacia[viz] @ git+https://github.com/sym823808458/Surfacia.git"
   python -c "import surfacia; print(surfacia.__version__)"
   surfacia --help

**PyPI publication of 4.0.0 is pending; the current PyPI release is 3.0.3.**
The command above installs current GitHub source and requires Git. A plain
``pip install surfacia`` still installs the former version. Alternatively clone:

.. code-block:: bash

   git clone https://github.com/sym823808458/Surfacia.git
   cd Surfacia
   python -m pip install -e ".[viz]"

ML constraints remain ``xgboost>=2.1.4,<3`` and ``shap>=0.48,<0.49``.
The full quantum workflow additionally needs xTB, Gaussian (g16, formchk) and
Multiwfn, separately installed and licensed. They are not in the pip wheel.
The descriptor-table case needs none of them. Install ``viz`` for py3Dmol and
interactive visualization. Next: :doc:`quick_start`.
