Commands Reference
==================

Surfacia provides a comprehensive command-line interface for all aspects of surface-based molecular analysis. This section documents all available commands with detailed examples and best practices.

.. toctree::
   :maxdepth: 2

   workflow
   ml_analysis
   shap_viz
   utilities

Command Overview
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔄 workflow
      :link: workflow
      :link-type: doc

      Complete 8-step analysis pipeline from SMILES to interpretable predictions

   .. grid-item-card:: 🤖 ml-analysis
      :link: ml_analysis
      :link-type: doc

      Machine learning model training and evaluation with cross-validation

   .. grid-item-card:: 📊 shap-viz
      :link: shap_viz
      :link-type: doc

      Interactive SHAP visualization with AI-powered explanations

   .. grid-item-card:: 🛠️ utilities
      :link: utilities
      :link-type: doc

      Molecular visualization, batch processing, and error recovery tools

Quick Command Reference
-----------------------

**Complete Workflow**

.. code-block:: bash

   # Full analysis with intelligent resume
   surfacia workflow -i molecules.csv --resume --test-samples "1,2,3"

**Individual Analysis Steps**

.. code-block:: bash

   # Machine learning analysis only
   surfacia ml-analysis -i processed_data.csv --test-samples "1,2,3"
   
   # SHAP visualization with AI assistant
   surfacia shap-viz -i training_data.csv --api-key YOUR_API_KEY
   
   # Molecular structure visualization
   surfacia mol-drawer -i molecules.csv -o output_dir/
   
   # 3D molecular viewer
   surfacia mol-viewer -i molecule.xyz
   
   # Rerun failed calculations
   surfacia rerun-gaussian -i failed_molecules.csv

Common Parameters
-----------------

Most commands share these common parameters:

**Input/Output**
   - ``-i, --input``: Input CSV file with molecular data
   - ``-o, --output``: Output directory (default: auto-generated)

**Analysis Options**
   - ``--test-samples``: Comma-separated list of test sample indices
   - ``--resume``: Enable intelligent resume functionality
   - ``--batch-size``: Number of molecules to process in each batch

**Performance**
   - ``--parallel``: Number of parallel processes
   - ``--low-memory``: Enable low-memory mode for large datasets
   - ``--verbose``: Enable detailed logging

**AI Integration**
   - ``--api-key``: ZhipuAI API key for AI assistant features
   - ``--host``: Server host for web interfaces (default: localhost)
   - ``--port``: Server port for web interfaces (default: 8050)

Getting Help
------------

Get detailed help for any command:

.. code-block:: bash

   # General help
   surfacia --help
   
   # Command-specific help
   surfacia workflow --help
   surfacia ml-analysis --help
   surfacia shap-viz --help

**Help Output Example**

.. code-block:: text

   Usage: surfacia workflow [OPTIONS]

   Run the complete Surfacia workflow from SMILES to interpretable predictions.

   Options:
     -i, --input PATH              Input CSV file with SMILES data [required]
     -o, --output PATH             Output directory
     --test-samples TEXT           Comma-separated test sample indices
     --resume                      Enable intelligent resume functionality
     --batch-size INTEGER          Batch size for processing (default: 20)
     --parallel INTEGER            Number of parallel processes
     --api-key TEXT                ZhipuAI API key for AI features
     --help                        Show this message and exit.

Error Handling
--------------

Surfacia provides comprehensive error handling and recovery:

**Automatic Recovery**
   - Failed calculations are automatically retried
   - Intelligent resume skips completed steps
   - Batch processing isolates failures

**Manual Recovery**
   - Use ``surfacia rerun-gaussian`` for failed QM calculations
   - Check log files for detailed error information
   - Use ``--verbose`` flag for debugging

**Common Error Solutions**

.. admonition:: Gaussian calculation timeout
   :class: tip

   **Solution**: Increase timeout or use smaller batch sizes
   
   .. code-block:: bash
   
      surfacia workflow -i molecules.csv --batch-size 10 --timeout 3600

.. admonition:: Memory issues with large datasets
   :class: tip

   **Solution**: Enable low-memory mode and reduce batch size
   
   .. code-block:: bash
   
      surfacia workflow -i molecules.csv --low-memory --batch-size 5

Best Practices
--------------

**Data Preparation**
   - Validate SMILES strings before analysis
   - Use descriptive sample names
   - Include target properties in the same CSV file

**Performance Optimization**
   - Use ``--resume`` to avoid recomputing completed steps
   - Adjust ``--batch-size`` based on available memory
   - Enable ``--parallel`` processing for large datasets

**Result Interpretation**
   - Always examine SHAP visualizations for insights
   - Use AI assistant for complex pattern interpretation
   - Validate important findings with domain knowledge

**Workflow Management**
   - Keep organized directory structures
   - Use version control for input data
   - Document analysis parameters and results

Next Steps
----------

- **Start with**: :doc:`workflow` for complete analysis pipelines
- **Focus on ML**: :doc:`ml_analysis` for model development
- **Visualize results**: :doc:`shap_viz` for interpretable explanations
- **Use tools**: :doc:`utilities` for specialized tasks