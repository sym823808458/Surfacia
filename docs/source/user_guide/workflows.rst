The workflow · Keep the chemical question in view
==================================================

Start with the chemical region. An interpretable descriptor states its
**region, physical property and aggregation rule** explicitly.

1. Choose the region
--------------------

- **Element mode:** sulfur in an additive series with a sulfur-based prior.
- **Fragment mode:** a conserved catalytic core in an organocatalyst series.
- **Automated local mode:** local environments without a predefined region.

Choose before inspecting test outcomes. Keep whole-molecule descriptors alongside
local ones so that size, shape and exposure remain visible.
See :doc:`../descriptors/index` for the three modes.

2. Structure → wavefunction → surface → descriptors
---------------------------------------------------

Prepare structures from SMILES, optimize with xTB, generate the Gaussian
wavefunction, and obtain ESP/ALIE/LEAE with Multiwfn. Extract whole-molecule
and selected-region descriptors. Persistent sample names connect every stage.

.. code-block:: bash

   surfacia smi2xyz -i sulfur_molecules.csv
   # After quantum/Multiwfn stages produce your actual FullOption.csv:
   surfacia extract-features -i FullOption.csv --mode 1 --element S

Download :doc:`../examples/surface_workflow` for six public sulfur structures.
Record charge, multiplicity, conformer, method, solvent, density isovalue and units.
Do not replace failed calculations with an unexplained zero.

The article's recovered Mode 1/2 production settings were GFN2-xTB with ALPB water,
then a PBE1PBE/6-311G* single point with GD3 and SMD water, then Multiwfn at a
0.010 a.u. density isovalue. These are **article settings**, not a guarantee that
CLI defaults reproduce the protocol. Review generated inputs before running.

3. Descriptors → model → explanation
------------------------------------

With traceable targets and a fixed held-out set, fit a predefined feature block.
Keep held-out outcomes out of feature selection and interpretation ranking.
The configured end-to-end shortcut is:

.. code-block:: bash

   surfacia workflow -i molecules.csv --mode 1 --element S --test-samples "6,12" --epoch 64 --cores 4

This requires installed external chemistry programs and known targets;
it was not quantum-rerun for this manual. Inspect stages individually for a new system.
``--resume`` detects files; it does not validate scientific settings.
:doc:`../commands/workflow` lists the implemented options.

4. Attribution → chemical evidence
----------------------------------

Read the descriptor definition, the trend across molecules and original predictions
together. Ask which molecular change would test the suggested relationship.
See :doc:`../examples/interpretation` for a worked reading.

The public CLI provides an XGBoost/repeated-split workflow. The article's shared
128-split, dual-model OOF TreeSHAP/DeepSHAP, perturbation and blinded multi-LLM
assessment are a separately frozen research analysis. ``workflow`` does not
automatically run that assessment or reproduce the article benchmark.
