API Reference
=============

Complete API documentation for Surfacia's Python modules and functions.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   core
   descriptors
   ml
   visualization
   utils

Overview
--------

This section provides detailed API documentation for all Surfacia modules, classes, and functions. It's intended for developers who want to use Surfacia programmatically or extend its functionality.

Module Structure
----------------

Surfacia's API is organized into several key modules:

Core Module (``surfacia.core``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core module contains the fundamental workflow and processing functions:

* **Workflow Management**: Main workflow orchestration
* **File Processing**: Input/output handling and file management
* **Configuration**: Parameter management and validation
* **Logging**: Comprehensive logging and debugging support

Descriptors Module (``surfacia.descriptors``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The descriptors module implements molecular surface descriptors:

* **Size and Shape**: Geometric descriptors for molecular surfaces
* **Electronic Properties**: Quantum mechanical surface properties
* **Surface Analysis**: Advanced surface characterization methods
* **Custom Descriptors**: Framework for implementing new descriptors

Machine Learning Module (``surfacia.ml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ML module provides machine learning capabilities:

* **Feature Engineering**: Automated feature selection and transformation
* **Model Training**: Support for various ML algorithms
* **Cross-validation**: Robust model validation techniques
* **Performance Metrics**: Comprehensive evaluation metrics

Visualization Module (``surfacia.visualization``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The visualization module handles all plotting and interactive displays:

* **SHAP Visualizations**: Interpretable ML visualizations
* **Molecular Displays**: 3D molecular structure rendering
* **Statistical Plots**: Data analysis and results visualization
* **Interactive Dashboards**: Web-based interactive interfaces

Utilities Module (``surfacia.utils``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The utilities module contains helper functions and tools:

* **File Operations**: Advanced file handling utilities
* **Data Processing**: Data transformation and validation
* **System Integration**: External software integration
* **Debugging Tools**: Development and troubleshooting utilities

Usage Examples
--------------

Basic API Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   import surfacia
   from surfacia.core import workflow
   from surfacia.descriptors import calculate_descriptors
   
   # Run complete workflow
   results = workflow.run_workflow(
       input_file="molecule.xyz",
       output_dir="results/",
       config="config.yaml"
   )
   
   # Calculate specific descriptors
   descriptors = calculate_descriptors(
       surface_file="surface.wfn",
       descriptor_types=["size_shape", "electronic"]
   )

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from surfacia.ml import MLAnalyzer
   from surfacia.visualization import SHAPVisualizer
   
   # Machine learning analysis
   analyzer = MLAnalyzer(config="ml_config.yaml")
   model = analyzer.train_model(features, targets)
   predictions = analyzer.predict(test_features)
   
   # SHAP visualization
   visualizer = SHAPVisualizer(model=model)
   visualizer.create_dashboard(features, predictions)

Function Reference
------------------

For detailed function signatures, parameters, and return values, see the individual module documentation pages linked above.

Development Guidelines
----------------------

When extending Surfacia's API:

1. Follow the existing code structure and naming conventions
2. Include comprehensive docstrings with parameter descriptions
3. Add unit tests for new functionality
4. Update this documentation when adding new modules or functions
5. Ensure backward compatibility when modifying existing APIs

Error Handling
--------------

Surfacia uses a consistent error handling approach:

* **SurfaciaError**: Base exception class for all Surfacia-specific errors
* **ConfigurationError**: Configuration and parameter validation errors
* **ProcessingError**: Data processing and calculation errors
* **FileError**: File I/O and format errors
* **MLError**: Machine learning and model-related errors

See individual module documentation for specific exception types and handling strategies.