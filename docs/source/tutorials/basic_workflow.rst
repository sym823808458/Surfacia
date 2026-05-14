Basic Workflow Tutorial
=======================

This tutorial shows the simplest end-to-end Surfacia workflow from input structures to interpretable output.

Goal
----

Learn how to:

- prepare a minimal input table
- run the main workflow command
- locate the most important output files
- understand what each stage produced

Before You Start
----------------

You should already have:

- Surfacia installed
- Gaussian and Multiwfn available in your environment
- a CSV file with at least ``Sample Name`` and ``SMILES`` columns

Minimal Input Example
---------------------

.. code-block:: text

   Sample Name,SMILES
   caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
   ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O

Recommended Path
----------------

1. Prepare your molecule table and target values.
2. Run the standard workflow command.
3. Inspect generated descriptor tables.
4. Review the compact model and SHAP outputs.

Run the Workflow
----------------

For a first pass, use the full workflow with resume enabled:

.. code-block:: bash

   surfacia workflow -i molecules.csv --resume --test-samples "1,2"

What to Inspect First
---------------------

After the workflow finishes, focus on these outputs first:

- the complete descriptor table
- the training or modeling table used downstream
- the compact retained feature list
- the SHAP visualization outputs

Practical Reading Order
-----------------------

If you are new to Surfacia, this order usually works best:

1. confirm the workflow finished cleanly
2. inspect whether descriptor columns look reasonable
3. check which features survived compact modeling
4. inspect SHAP outputs only after the retained features make chemical sense

What a Good First Result Looks Like
-----------------------------------

A good first tutorial run usually gives you:

- a clean output directory structure
- descriptors spanning size, shape, electronics, and surface analysis
- a compact feature subset that is smaller than the raw matrix
- SHAP results that can be connected back to recognizable chemistry

Common Beginner Mistakes
------------------------

- starting with too many molecules before checking one small test run
- trusting prediction metrics without checking the retained descriptors
- treating all features as equally meaningful instead of focusing on the compact model
- using SHAP before confirming the feature table itself is sensible

What to Do Next
---------------

After one successful run, choose a direction:

- go to :doc:`advanced_analysis` if you want to compare Surfacia modes
- go to :doc:`machine_learning` if you want to focus on compact modeling
- go to :doc:`../examples/index` if you want problem-oriented usage patterns

See Also
--------

- :doc:`../getting_started/quick_start`
- :doc:`../commands/workflow`
