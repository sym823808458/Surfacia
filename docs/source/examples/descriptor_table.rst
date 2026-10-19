Case 1 · Descriptor table to predictions and SHAP
==================================================================================================

This runnable case lists 36 ordinary molecules. Six RDKit features are molecular
weight, TPSA, H-bond donors/acceptors, rotatable bonds and rings. The target is
**RDKit-calculated Crippen logP**. No experimental or unpublished labels are used.

This is an ML/SHAP tutorial with a computed target, **not a Surfacia quantum-surface
benchmark or a claim of experimental logP accuracy**.

Download and run
------------------------------------------------------------

- :download:`Descriptor CSV </_static/examples/descriptors.csv>`
- :download:`Source SMILES </_static/examples/molecules.csv>`
- :download:`Generation script </_static/examples/make_data.py>`
- :download:`Recorded run and metrics </_static/examples/run_summary.json>`

Install :doc:`../getting_started/installation`, save the CSV as ``descriptors.csv``:

.. code-block:: bash

   surfacia ml-analysis -i descriptors.csv --manual --manual-features Full --test-samples "6,12,18,24,30,36" --epoch 32 --cores 1 --train-test-split 0.85 --no-generate-fitting

All six features are fixed, with six named held-out samples. Thirty-two repeats
are a short operation demo, not the frozen manuscript protocol. Recorded outputs
were generated using this command and current source.

Observed run: modeling-pool validation R² = 0.766, MAE = 0.267; held-out R² = 0.076, MAE = 0.307. The six-sample test R² is unstable and weak; this example does not establish generalization. Each modeling sample appeared in validation at least twice.

.. figure:: /_static/images/demo_predictions.png
   :alt: Actual validation and held-out predictions of RDKit-calculated logP

   Observed example outputs; the target is a computed RDKit property.

.. figure:: /_static/images/demo_shap.png
   :alt: Mean absolute SHAP for the six RDKit features

   Model attribution, not a validated chemical mechanism.

- :download:`Detailed modeling-pool CSV </_static/examples/Training_Set_Detailed_demo.csv>`
- :download:`Detailed held-out CSV </_static/examples/Test_Set_Detailed_demo.csv>`

Read :doc:`../user_guide/results` and open outputs with :doc:`interpretation`.
