Start from a descriptor table
============================================================

Start with :doc:`../examples/descriptor_table`. Download the complete CSV into a
new directory. Its target and features are computed by RDKit: this is an operation
tutorial, **not a Surfacia quantum-surface benchmark**.

Input contract
------------------------------------------------------------

Column order: ``Sample Name``, numeric descriptors, ``smiles``, ``target``.
Use exact spelling and unique sample names. This supervised command requires a
target. Do not include outcome/identifier columns among descriptors.

.. code-block:: text

   Sample Name,MolWt,TPSA,HBD,HBA,RotB,Rings,smiles,target

.. code-block:: bash

   surfacia ml-analysis -i descriptors.csv --manual --manual-features Full --test-samples "6,12,18,24,30,36" --epoch 32 --cores 1 --train-test-split 0.85 --no-generate-fitting

``--manual --manual-features Full`` fixes features before validation.
``--test-samples`` matches **Sample Name values**, not row positions or ranges.
``--epoch`` counts repeated random splits; ``--train-test-split`` is the training
fraction inside the modeling pool, not the permanent test fraction.

Find ``Full_Feature_Analysis_*`` beside the input. Inspect detailed CSVs and prediction
plots; read :doc:`../user_guide/results` before interpreting R².
