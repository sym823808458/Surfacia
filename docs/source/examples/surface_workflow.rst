Case 2 · Sulfur-centered surfaces
==================================================================

A public structure-only set contains methanethiol, ethanethiol, dimethyl sulfide,
dimethyl disulfide, thiophene and thioanisole. It illustrates an element-centered
region choice, not catalyst-performance prediction. No FE, yield, adsorption
energy or activity labels are fabricated.

- :download:`Structure-only CSV </_static/examples/sulfur_molecules.csv>`

.. figure:: /_static/images/sulfur_molecules.png
   :alt: Six sulfur-containing structures drawn from the supplied SMILES

   Real structures; no computed surface is shown.

Check molecular state/conformers, comparable sulfur regions and consistent
ESP/ALIE/LEAE settings. Ask whether a local trend is instead explained by a
whole-molecule change. Prepare structures:

.. code-block:: bash

   surfacia smi2xyz -i sulfur_molecules.csv

Continue with :doc:`../user_guide/workflows` and :doc:`../commands/index`.
After Multiwfn produces the required table, the extraction pattern is:

.. code-block:: bash

   surfacia extract-features -i FullOption.csv --mode 1 --element S

Replace the filename with your **actual output**. This case stops before
supervised ML: real traceable targets and a frozen evaluation design are needed.
Quantum calculations were not rerun for this manual; no quantum surface or
reaction-performance plot is presented as an observed result.
