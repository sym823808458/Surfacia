Surfacia MCP Server
===================

The ``surfacia-mcp`` command starts a stdio MCP server that exposes Surfacia workflow stages as structured tools. This makes Surfacia easier to connect to agent systems, orchestration layers, or other MCP-compatible clients without rewriting the chemistry codebase from scratch.

Why This Exists
---------------

The Surfacia codebase already organizes the core scientific workflow into separable stages:

1. SMILES to XYZ conversion
2. XTB optimization
3. Gaussian input generation
4. Gaussian execution
5. Multiwfn processing
6. Feature extraction
7. Machine-learning analysis
8. SHAP-based interpretation

The MCP server wraps these stages as tool calls with explicit inputs and structured outputs so an agent can:

- inspect the current state of a working directory
- run one stage at a time
- keep intermediate files visible and auditable
- launch the SHAP dashboard without blocking the server process

What Ships in the Current Version
---------------------------------

The current implementation lives under ``surfacia/mcp/`` and includes the following tools:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Tool name
     - Purpose
   * - ``surfacia_check_environment``
     - Check Python dependencies and external chemistry executables such as ``xtb``, ``g16``, ``formchk``, and ``Multiwfn_noGUI``.
   * - ``surfacia_detect_workflow_state``
     - Inspect a working directory and report which Surfacia stage is ready next.
   * - ``surfacia_generate_benchmark_matrices``
     - Build RDKit and fingerprint-based benchmark descriptor matrices from ``smiles`` and ``target`` columns.
   * - ``surfacia_smi_to_xyz``
     - Convert a molecular CSV into xyz structures and Surfacia mapping files.
   * - ``surfacia_run_xtb_opt``
     - Run XTB geometry optimization on xyz files.
   * - ``surfacia_generate_gaussian_inputs``
     - Generate Gaussian ``.com`` files from xyz structures.
   * - ``surfacia_run_gaussian_jobs``
     - Run Gaussian and ``formchk`` over prepared input files.
   * - ``surfacia_rerun_failed_gaussian_jobs``
     - Detect and rerun missing or empty ``.fchk`` jobs.
   * - ``surfacia_run_multiwfn_analysis``
     - Produce ``RawFull`` and ``FullOption`` outputs from completed wavefunction files.
   * - ``surfacia_extract_features``
     - Run Surfacia Mode 1, 2, or 3 feature extraction.
   * - ``surfacia_run_ml_analysis``
     - Run workflow-mode or manual-mode compact model analysis.
   * - ``surfacia_launch_shap_visualizer``
     - Launch the Dash SHAP application as a detached subprocess.
   * - ``surfacia_run_full_pipeline``
     - Execute the full staged workflow from input CSV to ML outputs, with optional SHAP launch.

Installation
------------

Install Surfacia from source in editable mode so the ``surfacia-mcp`` console script becomes available:

.. code-block:: bash

   git clone https://github.com/sym823808458/Surfacia.git
   cd Surfacia
   pip install -e .

You can confirm that the MCP entrypoint is installed:

.. code-block:: bash

   surfacia-mcp --log-level INFO

You can also launch the server as a Python module:

.. code-block:: bash

   python -m surfacia.mcp.server --log-level INFO

.. admonition:: Important
   :class: note

   ``surfacia-mcp`` is a stdio MCP server. It is meant to be started by an MCP-compatible client, not used as a human-interactive shell command.

Recommended First-Run Checks
----------------------------

Before running chemistry jobs through MCP, start with the two lightweight tools below:

1. ``surfacia_check_environment``
2. ``surfacia_detect_workflow_state``

This usually tells you whether the problem is:

- a missing executable
- an import problem
- a partially completed workflow directory
- or a real chemistry/runtime failure in one of the heavy stages

Typical Startup Sequence
------------------------

The most reliable first-run sequence is:

.. code-block:: text

   1. surfacia_check_environment
   2. surfacia_detect_workflow_state
   3. surfacia_generate_benchmark_matrices   (optional)
   4. surfacia_extract_features              (if FullOption already exists)
   5. surfacia_run_ml_analysis               (if FinalFull already exists)
   6. surfacia_run_xtb_opt / gaussian / multiwfn tools
   7. surfacia_launch_shap_visualizer

This staged approach makes debugging much easier than immediately calling the full end-to-end tool.

Structured Output Contract
--------------------------

Every tool returns the same top-level envelope:

.. code-block:: json

   {
     "ok": true,
     "tool": "surfacia_run_ml_analysis",
     "summary": "Completed workflow-mode ML analysis on 42 samples.",
     "working_dir": "D:/.../case1",
     "artifacts": {},
     "metrics": {},
     "logs": {}
   }

This design keeps the server friendly to agents because they can:

- read a short natural-language summary
- inspect metrics separately from file artifacts
- examine logs only when needed

Client Configuration
--------------------

Any MCP-compatible client should be able to launch the server with a command resembling:

.. code-block:: json

   {
     "command": "surfacia-mcp",
     "args": ["--log-level", "INFO"]
   }

If your environment does not expose the console script, use:

.. code-block:: json

   {
     "command": "python",
     "args": ["-m", "surfacia.mcp.server", "--log-level", "INFO"]
   }

If the SHAP dashboard should use AI-assisted analysis, provide the API key in the client environment:

.. code-block:: text

   SURFACIA_ZHIPUAI_API_KEY=your_api_key_here

Debugging Checklist
-------------------

If the MCP server starts but a tool fails, check the following in order:

1. Is the Surfacia Python environment the one that contains ``rdkit``, ``shap``, ``dash``, and related dependencies?
2. Does ``surfacia_check_environment`` report that ``xtb``, ``g16``, ``formchk``, and ``Multiwfn_noGUI`` are available?
3. Are you pointing the tool to the correct working directory or input file?
4. Does the directory contain the expected intermediate files for the requested stage?
5. For SHAP dashboard startup, did the server produce the detached log files under ``surfacia_mcp_logs/``?

Current Limitations
-------------------

The current MCP implementation is intentionally practical rather than fully productionized.

What still needs improvement:

- persistent ``job_id`` tracking for long Gaussian and Multiwfn jobs
- resumable asynchronous state instead of fully synchronous heavy-tool execution
- richer structured SHAP summaries
- dedicated regression tests for every MCP wrapper
- more decoupling from filesystem-oriented legacy entrypoints

Chinese Manual
--------------

For a step-by-step Chinese startup guide, see :doc:`mcp_server_zh`.
