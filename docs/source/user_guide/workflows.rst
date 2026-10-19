From structures to interpretation
==================================================================

The configured local/Linux workflow is:

1. SMILES to 3D.
2. xTB optimization.
3. Gaussian inputs and wavefunctions.
4. Multiwfn properties and molecular surfaces.
5. Whole/local descriptor extraction.
6. Fixed-feature modeling or exploratory selection.
7. Original prediction and SHAP inspection.

Check :doc:`../getting_started/installation` and work in a dedicated directory.
The supervised full workflow requires known targets.

.. code-block:: bash

   surfacia workflow -i molecules.csv --mode 1 --element S --test-samples "6,12" --epoch 64 --cores 4

This is a **configured recipe**, not a quantum run executed by this manual.
Inspect charge, spin, method, basis and solvent before long jobs.
Actual parameters: :doc:`../commands/workflow`. ``--resume`` detects files; it
does not certify their scientific settings or convergence.

Structure-only inputs: :doc:`../examples/surface_workflow`.
Existing descriptors: :doc:`../getting_started/quick_start`.
