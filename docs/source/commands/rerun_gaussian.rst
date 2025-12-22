Gaussian Rerun Utility (rerun-gaussian)
=======================================

Intelligent Gaussian calculation restart and recovery tool for handling failed or incomplete quantum chemistry calculations.

Overview
--------

The [`rerun-gaussian`](../commands/rerun_gaussian.rst:1) command provides automated recovery and restart capabilities for Gaussian quantum chemistry calculations. This tool analyzes failed calculations, diagnoses common issues, and automatically applies appropriate fixes to resume computations.

Command Syntax
---------------

.. code-block:: bash

   surfacia rerun-gaussian [OPTIONS] INPUT_FILE

Basic Usage Examples
--------------------

Simple Restart
~~~~~~~~~~~~~~

.. code-block:: bash

   # Restart a failed Gaussian calculation
   surfacia rerun-gaussian molecule.gjf
   
   # Restart with automatic error detection
   surfacia rerun-gaussian molecule.gjf --auto-fix
   
   # Restart from checkpoint file
   surfacia rerun-gaussian molecule.gjf --from-checkpoint

Command-Line Options
---------------------

Input Options
~~~~~~~~~~~~~

``INPUT_FILE``
  **Required.** Gaussian input file to restart
  
``--from-checkpoint, -c``
  Restart from existing checkpoint file
  
``--auto-fix``
  Enable automatic error detection and fixing

Recovery Options
~~~~~~~~~~~~~~~~

``--max-cycles N``
  Set maximum SCF or optimization cycles
  
``--memory MEMORY``
  Set memory allocation (e.g., 8GB)
  
``--processors N``
  Set number of processors

Common Error Types
------------------

**SCF Convergence Failures**
  - Auto-fix: Increase cycles, change algorithm
  
**Geometry Optimization Problems**
  - Auto-fix: Looser convergence, different optimizer
  
**Memory Issues**
  - Auto-fix: Increase memory, reduce basis set

Integration with Workflow
-------------------------

.. code-block:: bash

   # Used automatically in workflow when calculations fail
   surfacia workflow --input molecule.xyz --output results/ --auto-restart

See Also
--------

- :doc:`workflow` - Complete analysis pipeline
- :doc:`../getting_started/installation` - Gaussian setup requirements