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
   mode3_top20_remote_debug

Overview
--------

This section is organized around problem types rather than around software components. The goal is to help you recognize which style of Surfacia usage matches your chemistry.

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

Recommended Reading Order
-------------------------

For most users, this order works best:

1. start with :doc:`basic_molecules`
2. move to :doc:`complex_systems`
3. compare how the outputs feed into :doc:`machine_learning`
4. use :doc:`custom_workflows` when you want more flexible project structure

Quick Start Examples
--------------------

Basic Workflow Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run a first small workflow from a CSV input table
   surfacia workflow -i molecules.csv --resume --test-samples "1,2"

Machine Learning Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reuse a descriptor table for compact modeling
   surfacia ml-analysis -i descriptors.csv --target-property "LogP" --test-samples "1,2,3"

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Interpret retained features with SHAP outputs
   surfacia shap-viz -i training_data.csv --api-key YOUR_API_KEY

How to Use These Example Pages
------------------------------

These pages are intentionally written so you can adapt them to your own data instead of depending on a bundled example dataset.

- replace the sample input table with your own CSV
- keep the workflow pattern the same
- compare your outputs against the interpretation checklist in each page

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
