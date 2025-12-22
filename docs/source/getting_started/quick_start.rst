Quick Start
===========

This guide will get you up and running with Surfacia in just a few minutes. We'll walk through a complete analysis from SMILES input to interpretable predictions.

Prerequisites
-------------

Before starting, ensure you have:

- Surfacia installed (see :doc:`installation`)
- Gaussian 16 and Multiwfn properly configured
- A CSV file with molecular SMILES

5-Minute Tutorial
-----------------

**Step 1: Prepare Your Data**

Create a CSV file with your molecules:

.. code-block:: bash

   # Create example dataset
   cat > molecules.csv << EOF
   Sample Name,SMILES
   caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
   ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
   EOF

**Step 2: Run Complete Workflow**

Execute the full analysis pipeline:

.. code-block:: bash

   # Complete workflow with intelligent resume
   surfacia workflow -i molecules.csv --resume --test-samples "1,2"

.. admonition:: Expected Output
   :class: tip

   .. code-block:: text

      🚀 Starting Surfacia Workflow Analysis
      📁 Input: molecules.csv (3 molecules)
      🎯 Test samples: caffeine, aspirin
      
      ✓ Step 1: SMILES Processing - Completed
      ✓ Step 2: 3D Structure Generation - Completed  
      ⚡ Step 3: Gaussian Calculations - Running...
      📊 Progress: [██████████] 100% (3/3 molecules)
      
      ✓ Step 4: Multiwfn Analysis - Completed
      ✓ Step 5: Surface Property Mapping - Completed
      ✓ Step 6: Feature Extraction - Completed (78 descriptors)
      ✓ Step 7: Machine Learning Analysis - Completed
      ✓ Step 8: SHAP Visualization - Completed
      
      🎉 Analysis Complete! Results saved to Surfacia_3.0_[timestamp]/

**Step 3: Explore Results**

The workflow generates several output files:

.. code-block:: text

   Surfacia_3.0_20241201_143022/
   ├── FinalFull_data.csv           # Complete descriptor dataset
   ├── Training_Set_Detailed.csv    # ML training data
   ├── SHAP_analysis_results.html   # Interactive SHAP visualization
   ├── model_performance.png        # Model evaluation plots
   └── feature_importance.csv       # Feature ranking

**Step 4: View Interactive Results**

Open the SHAP visualization in your browser:

.. code-block:: bash

   # Launch interactive SHAP analysis
   surfacia shap-viz -i Surfacia_3.0_*/Training_Set_Detailed*.csv --api-key YOUR_API_KEY

This opens an interactive dashboard where you can:

- Explore SHAP values for each molecule
- Get AI-powered explanations of results
- Identify key molecular features
- Generate design hypotheses

Understanding the Output
------------------------

**Descriptor Categories**

Surfacia generates three types of descriptors:

.. tabs::

   .. tab:: Size & Shape (22 features)

      Basic molecular properties:
      
      - ``Atom Number``: Total atom count
      - ``Molecule Weight``: Molecular weight (Da)
      - ``Sphericity``: Shape compactness measure
      - ``Molecular Size Long``: Longest dimension (Å)

   .. tab:: Electronic Properties (28 features)

      Quantum mechanical descriptors:
      
      - ``HOMO``: Highest occupied molecular orbital energy
      - ``LUMO``: Lowest unoccupied molecular orbital energy
      - ``ALIE_min``: Most nucleophilic site
      - ``ESP_max``: Most electrophilic site

   .. tab:: Surface Analysis (32 features)

      Multi-scale surface properties:
      
      - ``Atom_ALIE_min``: Global most nucleophilic atom
      - ``Fun_ESP_delta``: Functional group polarity range
      - ``Atom_area_mean``: Average atomic surface area
      - ``Fun_LEAE_max``: Strongest electron-accepting group

**SHAP Interpretation**

SHAP values show how each feature contributes to predictions:

- **Positive values**: Feature increases predicted property
- **Negative values**: Feature decreases predicted property  
- **Magnitude**: Strength of the contribution
- **Color coding**: Red (increase) vs Blue (decrease)

Common Workflows
----------------

**Workflow 1: Property Prediction**

For predicting molecular properties:

.. code-block:: bash

   # Full workflow for property prediction
   surfacia workflow -i molecules.csv --target-property "LogP" --resume

**Workflow 2: Activity Classification**

For binary classification tasks:

.. code-block:: bash

   # Classification workflow
   surfacia workflow -i molecules.csv --target-property "Active" --classification --resume

**Workflow 3: Batch Processing**

For large datasets:

.. code-block:: bash

   # Process in batches with parallel execution
   surfacia workflow -i large_dataset.csv --batch-size 50 --parallel 4 --resume

**Workflow 4: Custom Analysis**

For specific molecular fragments:

.. code-block:: bash

   # Fragment-specific analysis
   surfacia workflow -i molecules.csv --fragment-file benzene.xyz --resume

Individual Commands
-------------------

You can also run individual steps:

**Molecular Visualization**

.. code-block:: bash

   # Generate 2D molecular structures
   surfacia mol-drawer -i molecules.csv -o molecular_structures/
   
   # View 3D structures interactively
   surfacia mol-viewer -i molecule.xyz

**Machine Learning Only**

.. code-block:: bash

   # Run ML analysis on existing descriptors
   surfacia ml-analysis -i processed_data.csv --test-samples "1,2,3" --cv-folds 5

**SHAP Analysis Only**

.. code-block:: bash

   # Generate SHAP explanations
   surfacia shap-viz -i training_data.csv --api-key YOUR_API_KEY --port 8050

**Error Recovery**

.. code-block:: bash

   # Rerun failed Gaussian calculations
   surfacia rerun-gaussian -i failed_molecules.csv

Performance Tips
----------------

**Optimize Calculations**

.. code-block:: bash

   # Use multiple CPU cores
   export OMP_NUM_THREADS=8
   
   # Enable intelligent resume to skip completed steps
   surfacia workflow -i molecules.csv --resume
   
   # Process in parallel batches
   surfacia workflow -i molecules.csv --batch-size 20 --parallel 4

**Memory Management**

For large datasets:

.. code-block:: bash

   # Reduce memory usage
   surfacia workflow -i molecules.csv --low-memory --batch-size 10

**Speed Up Development**

During model development:

.. code-block:: bash

   # Skip expensive QM calculations for testing
   surfacia ml-analysis -i existing_descriptors.csv --quick-test

Troubleshooting
---------------

**Common Issues**

.. admonition:: Gaussian calculation fails
   :class: warning

   **Symptoms**: ``Error in Gaussian calculation for molecule X``
   
   **Solutions**:
   
   - Check molecular structure validity
   - Use ``surfacia rerun-gaussian`` to retry failed calculations
   - Adjust Gaussian parameters in configuration

.. admonition:: Out of memory
   :class: warning

   **Symptoms**: ``MemoryError`` during large calculations
   
   **Solutions**:
   
   - Reduce batch size: ``--batch-size 10``
   - Enable low-memory mode: ``--low-memory``
   - Process subsets of data separately

.. admonition:: SHAP visualization not loading
   :class: warning

   **Symptoms**: Browser shows blank page
   
   **Solutions**:
   
   - Check if port is available: ``--port 8051``
   - Verify API key is set correctly
   - Try different browser or disable ad blockers

**Getting Help**

.. code-block:: bash

   # Get help for any command
   surfacia workflow --help
   surfacia ml-analysis --help
   surfacia shap-viz --help

Next Steps
----------

Now that you've completed your first analysis:

1. **Understand the theory**: Read :doc:`basic_concepts`
2. **Explore commands**: Check the :doc:`../commands/index` reference
3. **Learn advanced techniques**: Follow :doc:`../tutorials/index`
4. **Understand descriptors**: Study :doc:`../descriptors/index`
5. **See real examples**: Browse :doc:`../examples/index`

**Advanced Features to Explore**

- Custom descriptor selection
- Fragment-specific analysis
- Batch processing optimization
- Integration with Jupyter notebooks
- API development for custom applications