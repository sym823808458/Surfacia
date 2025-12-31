API Reference
=============

This section contains detailed API documentation for all Surfacia modules and commands.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   core

Machine Learning Modules
----------------------

.. toctree::
   :maxdepth: 2

   ml

Visualization Modules
--------------------

.. toctree::
   :maxdepth: 2

   visualization

Utility Modules
---------------

.. toctree::
   :maxdepth: 2

   utils

Command Reference
-----------------

.. toctree::
   :maxdepth: 2

   commands

Overview
--------

This section provides comprehensive API documentation for all Surfacia modules, classes, and functions. It's designed for developers who want to use Surfacia programmatically or extend its functionality.

About This Documentation
-----------------------

**Accuracy Commitment**: This API documentation is based on the actual implementation in the Surfacia codebase. All documented functions, classes, and parameters are verified against the source code to ensure accuracy and reliability.

**Documentation Coverage**:

- **Core Modules**: Quantum chemical calculations, molecular structure conversion, and workflow orchestration
- **Machine Learning**: XGBoost-based modeling, feature selection, and SHAP analysis
- **Visualization**: Interactive dashboards, 3D molecular viewers, and plotting tools
- **Utilities**: Molecular information calculation, file processing, and data utilities
- **Command Line Interface**: Complete CLI reference with all commands and parameters

Module Structure
-----------------

Surfacia's API is organized into several key modules:

Core Modules (``surfacia.core``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core modules provide fundamental functionality for quantum chemical calculations and workflow orchestration:

* **Gaussian Module**: Generate Gaussian input files and run quantum chemical calculations
* **SMILES to XYZ**: Convert SMILES strings from CSV files to 3D molecular structures
* **XTB Optimization**: Perform geometry optimizations using the xTB program
* **Multiwfn Module**: Run Multiwfn calculations for surface analysis
* **Workflow Module**: Complete workflow orchestration from SMILES to SHAP visualization
* **Rerun Gaussian**: Recover from failed calculations and system interruptions
* **Descriptors**: Calculate molecular shape and size descriptors from atomic coordinates

Machine Learning Module (``surfacia.ml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The machine learning module provides comprehensive tools for molecular descriptor analysis:

* **Chemical ML Analyzer**: XGBoost-based model training with feature selection
* **Manual Feature Analysis**: Analyze user-specified features
* **Workflow Analyzer**: Automatic feature selection and model optimization
* **SHAP Analysis**: Model interpretability with SHAP values
* **Feature Recommendations**: Intelligent feature selection based on multiple criteria

Visualization Module (``surfacia.visualization``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The visualization module handles all interactive and static visualizations:

* **Interactive SHAP Visualization**: Web-based dashboard with 3D molecular viewer and AI assistant
* **Molecular Drawer**: Generate 2D molecular structure images from SMILES
* **Molecular Viewer**: Interactive 3D molecular structure viewing
* **Surface Calculation**: Generate surface PDB files for visualization

Utility Modules (``surfacia.utils``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The utility modules provide helper functions for common tasks:

* **Molecular Information**: Calculate molecular properties (MW, LogP, HBD, HBA, TPSA, etc.)
* **Feature Extractor**: Extract molecular surface features using different extraction modes
* **File Utilities**: File I/O operations and directory management
* **Data Utilities**: Data validation, cleaning, and processing

Command Line Interface
----------------------

Surfacia provides a comprehensive command-line interface for accessing all functionality:

* **workflow**: Complete end-to-end analysis pipeline
* **smi2xyz**: Convert SMILES to XYZ format
* **xtb-opt**: Perform XTB geometry optimization
* **xyz2gaussian**: Generate Gaussian input files
* **run-gaussian**: Execute Gaussian calculations
* **multiwfn**: Run Multiwfn surface analysis
* **extract-features**: Extract molecular surface descriptors
* **ml-analysis**: Machine learning analysis with SHAP interpretation
* **shap-viz**: Interactive SHAP visualization with AI assistant
* **mol-draw**: Generate 2D molecular structure images
* **mol-info**: Calculate and display molecular properties
* **rerun-gaussian**: Rerun failed Gaussian calculations

Quick Start Guide
-----------------

**Using Core Workflow:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Run complete analysis pipeline
   workflow = SurfaciaWorkflow("molecules.csv")
   workflow.run_full_workflow(
       resume=True,
       max_features=5,
       test_samples="79,22,82,36,70,80"
   )

**Using Individual Core Modules:**

.. code-block:: python

   from surfacia.core.smi2xyz import smi2xyz_main
   from surfacia.core.xtb_opt import run_xtb_opt
   from surfacia.core.gaussian import process_xyz_files, run_gaussian
   
   # Step-by-step processing
   smi2xyz_main("molecules.csv")
   run_xtb_opt()
   process_xyz_files()
   run_gaussian()

**Using Machine Learning Analyzer:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   # Automatic workflow analysis
   analyzer = WorkflowAnalyzer(
       data_file="FinalFull.csv",
       max_features=5,
       n_runs=3
   )
   analyzer.run(epoch=64, core_num=32)

**Using Visualization Tools:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import run_interactive_shap_viz
   
   # Launch interactive SHAP visualization
   run_interactive_shap_viz(
       csv_path="Training_Set_Detailed.csv",
       xyz_path="./xyz_files",
       api_key="your_api_key",
       port=8052
   )

**Using Utility Functions:**

.. code-block:: python

   from surfacia.utils.mol_info import calculate_molecular_properties
   
   # Calculate molecular properties
   properties = calculate_molecular_properties("CCO")
   print(f"Molecular weight: {properties['mw']}")
   print(f"LogP: {properties['logp']}")

Function Reference
------------------

For detailed function signatures, parameters, and return values, see the individual module documentation pages linked above.

Command Line Reference
----------------------

For comprehensive command-line interface documentation, see the **Command Reference** section, which includes:

- Detailed parameter descriptions for all commands
- Usage examples for common workflows
- Parameter explanations and default values
- Best practices and troubleshooting guides

Development Guidelines
----------------------

**When extending Surfacia's API:**

1. **Code Structure**: Follow the existing modular structure and naming conventions
2. **Documentation**: Include comprehensive docstrings with parameter descriptions and examples
3. **Testing**: Add unit tests for new functionality to ensure reliability
4. **Documentation Updates**: Update this API reference when adding new modules or functions
5. **Backward Compatibility**: Ensure changes don't break existing user code

**Documentation Standards:**

- All public functions must have docstrings
- Docstrings should include: description, parameters, returns, examples
- Use numpydoc or Google style docstring formats
- Provide clear, runnable code examples
- Document error conditions and exception types

**Testing Requirements:**

- Unit tests for all public functions
- Integration tests for workflows
- Tests should cover normal operation and error cases
- Use pytest testing framework
- Maintain test coverage above 80%

Error Handling
--------------

Surfacia uses Python's standard exception handling:

- **FileNotFoundError**: Input files not found
- **ValueError**: Invalid parameter values or data formats
- **TypeError**: Incorrect data types
- **RuntimeError**: Calculation or processing errors
- **Exception**: General errors for unexpected issues

**Best Practices:**

1. Use try-except blocks for file operations
2. Validate input data before processing
3. Provide informative error messages
4. Log errors with context information
5. Implement graceful degradation when possible

**Example Error Handling:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   from surfacia.core.rerun_gaussian import rerun_failed_gaussian_calculations
   
   try:
       workflow = SurfaciaWorkflow("molecules.csv")
       workflow.run_full_workflow()
   except FileNotFoundError as e:
       print(f"Input file not found: {e}")
   except ValueError as e:
       print(f"Invalid data format: {e}")
   except Exception as e:
       print(f"Error during workflow: {e}")
       # Attempt recovery
       rerun_failed_gaussian_calculations()
       workflow.run_full_workflow(resume=True)

Contributing
------------

To contribute to Surfacia's API:

1. Fork the repository on GitHub
2. Create a feature branch for your changes
3. Follow the development guidelines above
4. Add tests and update documentation
5. Submit a pull request with a clear description

**Resources:**

- GitHub Repository: https://github.com/sym823808458/Surfacia
- Issue Tracker: Report bugs and request features
- Discussions: Ask questions and share ideas
