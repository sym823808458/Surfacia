workflow
========

The ``workflow`` command runs the complete Surfacia analysis pipeline from SMILES input to interpretable predictions. This is the primary command for most users, providing an automated 8-step process with intelligent resume capabilities.

Synopsis
--------

.. code-block:: bash

   surfacia workflow [OPTIONS] -i INPUT_FILE

Description
-----------

The workflow command orchestrates the complete Surfacia analysis pipeline:

1. **SMILES Processing**: Validates and processes molecular structures
2. **3D Generation**: Creates optimized 3D conformers
3. **Gaussian Calculations**: Performs quantum mechanical calculations
4. **Multiwfn Analysis**: Analyzes wavefunctions and calculates properties
5. **Surface Mapping**: Maps electronic properties onto molecular surfaces
6. **Feature Extraction**: Generates comprehensive descriptor sets
7. **Machine Learning**: Trains predictive models with cross-validation
8. **SHAP Visualization**: Creates interpretable explanations with AI assistance

.. mermaid::

   graph TD
       A[SMILES Input] --> B[3D Generation]
       B --> C[Gaussian QM]
       C --> D[Multiwfn Analysis]
       D --> E[Surface Mapping]
       E --> F[Feature Extraction]
       F --> G[ML Training]
       G --> H[SHAP Analysis]
       
       style A fill:#e1f5fe
       style H fill:#f3e5f5

Options
-------

**Required Parameters**

.. option:: -i, --input PATH

   Input CSV file containing molecular data. Must include columns:
   
   - ``Sample Name``: Unique identifier for each molecule
   - ``SMILES``: Valid SMILES string representation
   - Optional: Target property columns for supervised learning

**Analysis Configuration**

.. option:: --test-samples TEXT

   Comma-separated list of sample indices or names to use as test set.
   
   **Examples**:
   
   - ``"1,2,3"`` - Use samples 1, 2, and 3 as test set
   - ``"caffeine,aspirin"`` - Use named samples as test set
   - ``"1-5,10,15-20"`` - Range notation supported

.. option:: --resume

   Enable intelligent resume functionality. The system automatically detects completed steps and continues from the last incomplete stage, potentially saving hours of computation time.

.. option:: --target-property TEXT

   Specify target property column name for supervised learning. If not provided, unsupervised analysis is performed.

.. option:: --classification

   Treat the target property as a classification problem (binary or multi-class) rather than regression.

**Performance Options**

.. option:: --batch-size INTEGER

   Number of molecules to process in each batch. Larger batches use more memory but may be more efficient.
   
   **Default**: 20
   
   **Recommendations**:
   
   - Small systems (< 8GB RAM): 5-10
   - Medium systems (8-16GB RAM): 10-20
   - Large systems (> 16GB RAM): 20-50

.. option:: --parallel INTEGER

   Number of parallel processes for calculations. Should not exceed the number of CPU cores.
   
   **Default**: Number of CPU cores

.. option:: --low-memory

   Enable low-memory mode for processing large datasets. Reduces memory usage at the cost of some performance.

.. option:: --timeout INTEGER

   Timeout in seconds for individual Gaussian calculations.
   
   **Default**: 3600 (1 hour)

**Output Options**

.. option:: -o, --output PATH

   Output directory for results. If not specified, creates a timestamped directory.
   
   **Default**: ``Surfacia_3.0_YYYYMMDD_HHMMSS/``

.. option:: --verbose

   Enable detailed logging output for debugging and monitoring progress.

**AI Integration**

.. option:: --api-key TEXT

   ZhipuAI API key for AI-powered explanations in SHAP visualization. Can also be set via ``ZHIPUAI_API_KEY`` environment variable.

.. option:: --host TEXT

   Host address for the SHAP visualization server.
   
   **Default**: ``localhost``

.. option:: --port INTEGER

   Port number for the SHAP visualization server.
   
   **Default**: 8050

Examples
--------

**Basic Usage**

.. code-block:: bash

   # Simple workflow with default settings
   surfacia workflow -i molecules.csv

**With Test Set and Resume**

.. code-block:: bash

   # Specify test samples and enable resume
   surfacia workflow -i molecules.csv --test-samples "1,2,3" --resume

**Supervised Learning**

.. code-block:: bash

   # Regression analysis
   surfacia workflow -i molecules.csv --target-property "LogP" --test-samples "1-5"
   
   # Classification analysis
   surfacia workflow -i molecules.csv --target-property "Active" --classification --test-samples "10,20,30"

**Performance Optimization**

.. code-block:: bash

   # Large dataset processing
   surfacia workflow -i large_dataset.csv --batch-size 50 --parallel 8 --low-memory --resume

**With AI Assistant**

.. code-block:: bash

   # Enable AI-powered explanations
   export ZHIPUAI_API_KEY="your_api_key_here"
   surfacia workflow -i molecules.csv --test-samples "1,2,3" --resume

**Custom Output Directory**

.. code-block:: bash

   # Specify custom output location
   surfacia workflow -i molecules.csv -o my_analysis_results/ --resume

Input File Format
-----------------

The input CSV file must contain the following columns:

**Required Columns**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "Sample Name", "Unique identifier", "caffeine"
   "SMILES", "Valid SMILES string", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

**Optional Columns**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "LogP", "Target property (regression)", "1.23"
   "Active", "Target property (classification)", "1"
   "MW", "Additional molecular properties", "194.19"

**Example Input File**

.. code-block:: text

   Sample Name,SMILES,LogP,Active
   caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,-0.07,1
   aspirin,CC(=O)OC1=CC=CC=C1C(=O)O,1.19,1
   ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,3.97,1
   glucose,C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O,-3.24,0

Output Files
------------

The workflow generates a comprehensive set of output files:

**Primary Results**

.. code-block:: text

   Surfacia_3.0_20241201_143022/
   ├── FinalFull_data.csv              # Complete descriptor dataset
   ├── Training_Set_Detailed.csv       # ML training data with SHAP values
   ├── model_performance.png           # Model evaluation plots
   ├── feature_importance.csv          # Ranked feature importance
   └── SHAP_analysis_results.html      # Interactive SHAP dashboard

**Intermediate Files**

.. code-block:: text

   ├── xyz_files/                      # 3D molecular structures
   ├── gaussian_outputs/               # Quantum calculation results
   ├── multiwfn_outputs/              # Wavefunction analysis results
   ├── surface_properties/            # Surface property maps
   └── logs/                          # Detailed execution logs

**File Descriptions**

.. glossary::

   FinalFull_data.csv
      Complete dataset with all calculated descriptors (typically 50-100 features)

   Training_Set_Detailed.csv
      Processed dataset ready for machine learning with selected features

   model_performance.png
      Visualization of model performance metrics (R², RMSE, confusion matrix)

   feature_importance.csv
      Ranked list of features by importance with SHAP values

   SHAP_analysis_results.html
      Interactive web dashboard for exploring SHAP explanations

Intelligent Resume Functionality
--------------------------------

The ``--resume`` flag enables sophisticated checkpoint-based resumption:

**Automatic Detection**

The system automatically detects completed steps by examining output files:

- **Steps 1-2**: Checks for XYZ coordinate files
- **Steps 3-4**: Verifies Gaussian output files and convergence
- **Step 5**: Confirms Multiwfn analysis completion
- **Step 6**: Validates descriptor extraction results
- **Step 7**: Checks ML model training completion
- **Step 8**: Verifies SHAP analysis results

**Time Savings**

Resume functionality can save significant computation time:

- **Small datasets (< 50 molecules)**: 30-60% time savings
- **Medium datasets (50-200 molecules)**: 50-75% time savings  
- **Large datasets (> 200 molecules)**: 60-80% time savings

**Example Resume Scenarios**

.. code-block:: bash

   # First run - interrupted after Step 4
   surfacia workflow -i molecules.csv --test-samples "1,2,3"
   # ... calculation interrupted ...
   
   # Resume from Step 5
   surfacia workflow -i molecules.csv --test-samples "1,2,3" --resume
   # ✓ Steps 1-4: Already completed, skipping...
   # ⚡ Starting from Step 5: Surface Mapping

Performance Considerations
-------------------------

**Memory Usage**

Typical memory requirements:

- **Small molecules (< 50 atoms)**: 2-4 GB per batch
- **Medium molecules (50-100 atoms)**: 4-8 GB per batch
- **Large molecules (> 100 atoms)**: 8-16 GB per batch

**Computation Time**

Approximate processing times per molecule:

- **Gaussian calculation**: 5-30 minutes (depends on molecule size)
- **Multiwfn analysis**: 1-5 minutes
- **Feature extraction**: < 1 minute
- **ML training**: Seconds to minutes (depends on dataset size)

**Optimization Strategies**

.. code-block:: bash

   # For memory-constrained systems
   surfacia workflow -i molecules.csv --batch-size 5 --low-memory
   
   # For time-critical analysis
   surfacia workflow -i molecules.csv --parallel 8 --resume
   
   # For large datasets
   surfacia workflow -i molecules.csv --batch-size 100 --parallel 16 --low-memory

Error Handling and Recovery
---------------------------

**Automatic Error Recovery**

- Failed Gaussian calculations are automatically retried
- Problematic molecules are isolated and reported
- Batch processing continues despite individual failures

**Manual Recovery**

.. code-block:: bash

   # Rerun failed calculations
   surfacia rerun-gaussian -i failed_molecules.csv
   
   # Then resume the workflow
   surfacia workflow -i molecules.csv --resume

**Common Issues and Solutions**

.. admonition:: Gaussian convergence failure
   :class: warning

   **Symptoms**: ``SCF convergence failure`` in logs
   
   **Solutions**:
   
   - Use smaller batch sizes
   - Check molecular structures for validity
   - Adjust Gaussian parameters in configuration

.. admonition:: Memory exhaustion
   :class: warning

   **Symptoms**: ``MemoryError`` or system becomes unresponsive
   
   **Solutions**:
   
   - Reduce ``--batch-size``
   - Enable ``--low-memory`` mode
   - Process dataset in smaller chunks

.. admonition:: Timeout errors
   :class: warning

   **Symptoms**: Calculations terminate after timeout period
   
   **Solutions**:
   
   - Increase ``--timeout`` value
   - Use more CPU cores with ``--parallel``
   - Check system load and available resources

Integration with Other Commands
-------------------------------

The workflow command integrates seamlessly with other Surfacia tools:

**Post-Analysis Visualization**

.. code-block:: bash

   # Run workflow first
   surfacia workflow -i molecules.csv --resume
   
   # Then explore results interactively
   surfacia shap-viz -i Surfacia_3.0_*/Training_Set_Detailed*.csv --api-key YOUR_KEY

**Molecular Structure Analysis**

.. code-block:: bash

   # Generate molecular visualizations
   surfacia mol-drawer -i molecules.csv -o structures/
   
   # View specific molecules
   surfacia mol-viewer -i Surfacia_3.0_*/xyz_files/caffeine.xyz

**Error Recovery Workflow**

.. code-block:: bash

   # Initial run with some failures
   surfacia workflow -i molecules.csv --resume
   
   # Fix failed calculations
   surfacia rerun-gaussian -i failed_molecules.csv
   
   # Complete the analysis
   surfacia workflow -i molecules.csv --resume

Best Practices
--------------

**Data Preparation**

1. **Validate SMILES**: Ensure all SMILES strings are chemically valid
2. **Unique Names**: Use descriptive, unique sample names
3. **Clean Data**: Remove duplicates and invalid entries
4. **Reasonable Size**: Start with smaller datasets (< 100 molecules) for testing

**Performance Optimization**

1. **Use Resume**: Always use ``--resume`` for interrupted calculations
2. **Batch Size**: Adjust based on available memory and molecule complexity
3. **Parallel Processing**: Use multiple cores but don't exceed system capacity
4. **Monitor Resources**: Watch memory and CPU usage during execution

**Result Interpretation**

1. **Check Logs**: Review execution logs for warnings or errors
2. **Validate Results**: Examine model performance metrics
3. **Explore SHAP**: Use interactive visualization for insights
4. **Domain Knowledge**: Interpret results in chemical context

**Workflow Management**

1. **Organized Directories**: Keep input and output files well-organized
2. **Version Control**: Track changes to input data and parameters
3. **Documentation**: Record analysis parameters and decisions
4. **Backup Results**: Save important results and intermediate files

See Also
--------

- :doc:`ml_analysis` - Machine learning analysis only
- :doc:`shap_viz` - Interactive SHAP visualization
- :doc:`utilities` - Supporting tools and utilities
- :doc:`../getting_started/quick_start` - Quick start tutorial