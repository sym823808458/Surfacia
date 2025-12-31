Basic Workflow Tutorial
=======================

This tutorial walks you through the complete Surfacia workflow from molecular input to interpretable machine learning results.

Objective
---------

In this tutorial, you will learn how to:

* Prepare molecular input data
* Run the complete 8-step Surfacia workflow
* Interpret the results
* Generate visualizations

Prerequisites
-------------

Before starting this tutorial, ensure you have:

* Surfacia installed (see :doc:`../getting_started/installation`)
* Basic knowledge of SMILES notation
* Gaussian and Multiwfn installed (for quantum calculations)
* A sample CSV file with molecular structures

Step 1: Prepare Input Data
--------------------------

Create a CSV file with your molecules. The file should contain at least a SMILES column:

.. code-block:: csv

   name,smiles,activity
   benzene,c1ccccc1,1.2
   toluene,c1ccccc1C,1.5
   phenol,c1ccccc1O,2.1

Save this as ``molecules.csv`` in your working directory.

Step 2: Run Complete Workflow
------------------------------

Use the workflow command to process your molecules:

.. code-block:: bash

   surfacia workflow -i molecules.csv --resume --test-samples "1,2,3"

This command will:

1. Convert SMILES to 3D structures
2. Optimize geometries using XTB
3. Run quantum chemical calculations with Gaussian
4. Analyze wavefunctions with Multiwfn
5. Extract surface descriptors
6. Perform feature engineering
7. Train machine learning models
8. Generate SHAP explanations

Step 3: Monitor Progress
------------------------

The workflow provides detailed progress updates:

.. code-block:: text

   [INFO] Starting Surfacia workflow...
   [INFO] Processing 3 molecules...
   [INFO] Step 1: Converting SMILES to XYZ...
   [INFO] Step 2: Geometry optimization with XTB...
   [INFO] Step 3: Quantum calculations with Gaussian...
   ...

Step 4: Examine Results
-----------------------

After completion, you'll find several output files:

* ``results.csv`` - Final predictions and descriptors
* ``model_results.pkl`` - Trained machine learning model
* ``shap_analysis.html`` - Interactive SHAP visualization
* ``molecular_structures/`` - 3D molecular structures

Step 5: Visualize Results
------------------------

Open the SHAP visualization:

.. code-block:: bash

   # Open in browser
   surfacia shap-viz -i results.csv --api-key YOUR_API_KEY
   
   # Or open the HTML file directly
   open shap_analysis.html

Expected Results
-----------------

You should see:

* Molecular structures displayed in 3D
* SHAP value plots showing feature importance
* Predicted activities with confidence intervals
* Surface property visualizations

Troubleshooting
----------------

Common issues and solutions:

**Gaussian not found**
   Ensure Gaussian is properly installed and in your PATH. Test with:

   .. code-block:: bash

      g16 --version

**Memory issues**
   Reduce memory allocation for Gaussian:

   .. code-block:: bash

      surfacia workflow -i molecules.csv --memory 8GB

**Convergence problems**
   Skip XTB optimization step:

   .. code-block:: bash

      surfacia workflow -i molecules.csv --skip-xtb

Next Steps
----------

After completing this tutorial:

* Try the :doc:`advanced_analysis` tutorial for more complex analyses
* Learn about :doc:`custom_descriptors` for specialized applications
* Explore the :doc:`machine_learning` tutorial for advanced ML techniques

Additional Resources
--------------------

* :doc:`../commands/workflow` - Detailed workflow command options
* :doc:`../commands/mol_viewer` - Molecular visualization tools
* :doc:`../commands/shap_viz` - SHAP analysis and visualization
