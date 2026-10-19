Migration and scope alignment
============================================================

**4.0.0 is a source revision for scope alignment.** Check
`PyPI <https://pypi.org/project/surfacia/>`_ for uploaded versions. An installed
3.0.3 retains its old behavior until upgraded or replaced with current source.
Historical wheels and research archives are not overwritten.

Removed from the current package
----------------------------------------------------------------

- Automatic ``SPES_Test_Set_Detailed_*`` / metadata outputs.
- SPES overlay and workflow auto-discovery.
- CLI/MCP ``--spes-csv`` and Python ``spes_csv_path`` arguments.
- The ``surfacia.ml.spes`` ranking module.

Remove these arguments from scripts. Use original ``Test_Set_Detailed`` via
``--test-csv`` for held-out prediction/SHAP points. Ordinary SHAP, original
predictions and surface descriptors remain. The major version records removed
interfaces, not a new quantum method.

The revised research scope excludes former Go Beyond, frontier acquisition and
SHAP-cliff claims. Historical manuscripts remain historical records, not silently
rewritten evidence. No unpublished manuscript data are distributed in the cases.
