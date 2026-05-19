Command Reference
=================

This section is the CLI lookup table for Surfacia. Use it when you already know what stage you want to run and need the exact command, input file, or option name.

.. toctree::
   :maxdepth: 2

   workflow
   ml_analysis
   shap_viz
   utilities

Core Commands
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Command
     - Main input
     - Purpose
   * - ``surfacia workflow``
     - Molecule table with SMILES
     - Run the full workflow from structures to ML and SHAP outputs.
   * - ``surfacia ml-analysis``
     - Descriptor table
     - Run model selection, prediction, SHAP analysis, and optional test-set outputs.
   * - ``surfacia shap-viz``
     - ``Training_Set_Detailed*.csv``
     - Launch the interactive SHAP dashboard.
   * - ``surfacia-mcp``
     - MCP client launch command
     - Start the stdio MCP server for agent-based workflows.

Common Examples
---------------

.. code-block:: bash

   surfacia workflow -i molecules.csv --test-samples "1,3"
   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv --test-samples "1,2,3"
   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_files --test-csv Test_Set_Detailed.csv
   surfacia-mcp --log-level INFO

Shared Options
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Meaning
   * - ``-i, --input``
     - Main input file for the command.
   * - ``-o, --output``
     - Output directory, when supported.
   * - ``--test-samples``
     - Comma-separated sample indices used as an external test set.
   * - ``--resume``
     - Continue a partially completed workflow.
   * - ``--api-key``
     - Optional ZhipuAI key for AI-assisted SHAP interpretation.
   * - ``--host`` / ``--port``
     - Host and port for local web interfaces.

Where To Go Next
----------------

- Full pipeline options: :doc:`workflow`
- ML-only options: :doc:`ml_analysis`
- SHAP dashboard options: :doc:`shap_viz`
- Utility commands: :doc:`utilities`
- MCP setup: :doc:`../integrations/mcp_server`
