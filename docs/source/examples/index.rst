Examples
========

Practical examples demonstrating Surfacia's capabilities across various molecular systems and analysis scenarios.

.. toctree::
   :maxdepth: 2
   :caption: Example Categories

   basic_molecules
   complex_systems
   machine_learning
   custom_workflows

Overview
--------

This section provides comprehensive examples showcasing Surfacia's molecular surface analysis capabilities. Each example includes complete input files, command sequences, and detailed interpretation of results.

Example Categories
------------------

Basic Molecules
~~~~~~~~~~~~~~~

Simple organic compounds that demonstrate fundamental concepts:

* **Small Molecules**: Methane, water, ammonia - basic surface analysis
* **Aromatic Systems**: Benzene, naphthalene - π-system surface properties
* **Functional Groups**: Alcohols, amines, carbonyls - heteroatom effects
* **Conformational Analysis**: Butane, cyclohexane - conformational effects

Complex Systems
~~~~~~~~~~~~~~~

Larger molecular systems showcasing advanced features:

* **Biomolecules**: Amino acids, nucleotides, small peptides
* **Drug Molecules**: Pharmaceutical compounds and their analogs
* **Organometallics**: Transition metal complexes and catalysts
* **Supramolecular Systems**: Host-guest complexes and assemblies

Machine Learning Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

Applications of ML techniques to molecular surface data:

* **Property Prediction**: QSAR models for biological activity
* **Classification**: Molecular categorization based on surface features
* **Feature Importance**: SHAP analysis of descriptor contributions
* **Model Validation**: Cross-validation and performance assessment

Custom Workflows
~~~~~~~~~~~~~~~~

Specialized analysis pipelines for specific research needs:

* **High-throughput Screening**: Batch processing of molecular libraries
* **Parameter Optimization**: Systematic parameter space exploration
* **Comparative Studies**: Multi-molecule comparative analysis
* **Integration Examples**: Combining with external tools and databases

Example Structure
-----------------

Each example follows a consistent format:

1. **Objective**: Clear statement of what the example demonstrates
2. **Input Files**: All required input files with explanations
3. **Command Sequence**: Step-by-step commands with explanations
4. **Expected Output**: Description of expected results and files
5. **Interpretation**: Scientific interpretation of the results
6. **Variations**: Alternative approaches and parameter modifications
7. **Troubleshooting**: Common issues and solutions

Quick Start Examples
--------------------

Basic Workflow Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Complete workflow for a simple molecule
   surfacia workflow --input benzene.xyz --output results/ --config basic.yaml
   
   # View results
   surfacia mol-viewer --file results/benzene_surface.wfn

Machine Learning Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Feature extraction and ML analysis
   surfacia workflow --input molecules.sdf --output ml_results/ --steps 1-6
   surfacia ml-analysis --input ml_results/features.csv --target activity.csv
   surfacia shap-viz --model ml_results/model.pkl --features ml_results/features.csv

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Interactive SHAP visualization
   surfacia shap-viz --model model.pkl --features features.csv --interactive
   
   # Molecular structure viewer
   surfacia mol-viewer --file molecule.wfn --surface --properties

Data Files
----------

All example data files are available in the ``examples/data/`` directory:

* **Input Structures**: XYZ, SDF, and MOL files for various molecules
* **Configuration Files**: YAML configuration files for different analysis types
* **Reference Results**: Expected output files for validation
* **Datasets**: Curated datasets for machine learning examples

Download and Setup
------------------

To use the examples:

1. Download the example data package
2. Extract to your working directory
3. Follow the individual example instructions
4. Compare your results with the provided reference outputs

Advanced Examples
-----------------

For users seeking more sophisticated applications:

* **Custom Descriptor Development**: Implementing new surface descriptors
* **Integration with Quantum Chemistry**: Combining with Gaussian, ORCA, etc.
* **High-Performance Computing**: Parallel processing and cluster deployment
* **Web Interface Development**: Creating custom web-based analysis tools

Contributing Examples
---------------------

We welcome contributions of new examples:

1. Follow the standard example format
2. Include all necessary input files
3. Provide clear documentation and interpretation
4. Test thoroughly before submission
5. Submit via pull request with detailed description

Support
-------

For help with examples:

* Check the troubleshooting sections in individual examples
* Consult the :doc:`../user_guide/index` for general guidance
* Visit the project repository for community support
* Contact the development team for specific issues