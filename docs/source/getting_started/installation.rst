Installation
============

This page keeps only the minimum steps needed to start using Surfacia.

Requirements
------------

- Python ``>=3.9``
- Linux/macOS/Windows (Linux recommended for long calculations)
- External tools in ``PATH``:
  - ``xtb``
  - ``g16`` and ``formchk``
  - ``Multiwfn`` or ``Multiwfn_noGUI``

Install Surfacia
----------------

.. code-block:: bash

   # optional but recommended
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   pip install --upgrade pip
   pip install surfacia

For development from source:

.. code-block:: bash

   git clone https://github.com/sym823808458/Surfacia.git
   cd Surfacia
   pip install -e .

Compatibility Note (Important)
------------------------------

If you see errors like:

.. code-block:: text

   could not convert string to float: '[-3.xxxE0]'

pin ML dependencies to the validated combination:

.. code-block:: bash

   pip install --force-reinstall "xgboost==2.1.4" "shap==0.48.0"

Quick Verification
------------------

.. code-block:: bash

   surfacia --help
   python -c "import surfacia, xgboost, shap; print(surfacia.__version__, xgboost.__version__, shap.__version__)"
   which xtb
   which g16
   which formchk
   which Multiwfn

Next
----

- First run: :doc:`quick_start`
- Error handling: :doc:`../user_guide/troubleshooting`
- Parameter details: :doc:`../commands/index`
