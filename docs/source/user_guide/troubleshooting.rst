Troubleshooting
============================================================


.. code-block:: bash

   python -c "import sys,surfacia,xgboost,shap; print(sys.executable); print(surfacia.__version__, xgboost.__version__, shap.__version__)"
   python -m pip check
   surfacia ml-analysis --help

- Command missing: activate the environment used for installation.
- Missing openbabel/RDKit: install declared dependencies in that environment.
- ``could not convert string to float: '[-3.xxxE0]'``: check the pinned
  XGBoost/SHAP ranges in :doc:`../getting_started/installation`.
- Non-numeric features: only numeric descriptors belong between ``Sample Name`` and ``smiles,target``.
- Test sample not found: use the exact name, not row positions or ranges.
- Missing xTB/Gaussian/Multiwfn: install external executables separately.
- No 3D surface: supply matching real structure/PDB files.
- SSH dashboard: bind localhost and forward the port.
- Rejected old tutorial option: use :doc:`../commands/index`; the older generic algorithm/classification options were not implemented.

File issues with versions, command, sanitized header and traceback.
