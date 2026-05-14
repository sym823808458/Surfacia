Tutorials
=========

Step-by-step tutorials for learning Surfacia's capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial Topics

   basic_workflow
   advanced_analysis
   custom_descriptors
   machine_learning

Overview
--------

These tutorials are arranged as a practical learning path rather than a purely technical reference. They are meant to help you go from "I can run Surfacia" to "I can choose the right analysis strategy and trust what I am reading."

Tutorial Structure
------------------

Each tutorial follows a consistent structure:

1. **Objective**: What you'll learn and accomplish
2. **Prerequisites**: Required knowledge and setup
3. **Step-by-step Instructions**: Detailed walkthrough
4. **Expected Results**: What outputs to expect
5. **Troubleshooting**: Common issues and solutions
6. **Next Steps**: Related tutorials and advanced topics

Recommended Learning Path
-------------------------

If you are new to Surfacia, this order usually works best:

1. start with :doc:`basic_workflow`
2. move to :doc:`machine_learning` once you have a descriptor table
3. read :doc:`advanced_analysis` when you want to compare analysis modes
4. use :doc:`custom_descriptors` only when you are ready to extend the system itself

Available Tutorials
-------------------

Basic Workflow Tutorial
~~~~~~~~~~~~~~~~~~~~~~~

Learn the fundamental Surfacia workflow from molecular input to final analysis results.

* Setting up input files
* Running the complete 8-step workflow
* Interpreting results
* Basic visualization

Advanced Analysis Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~

Explore advanced features for specialized molecular surface analysis.

* Custom parameter optimization
* Multi-scale descriptor analysis
* Advanced visualization techniques
* Performance optimization

Custom Descriptors Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and integrate custom molecular descriptors.

* Understanding the descriptor framework
* Implementing custom descriptors
* Integration with existing workflows
* Validation and testing

Machine Learning Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply machine learning techniques to molecular surface data.

* Feature selection and engineering
* Model training and validation
* SHAP-based interpretability
* Results interpretation

Getting Started
---------------

To begin with the tutorials:

1. Ensure Surfacia is properly installed (see :doc:`../getting_started/installation`)
2. Begin with a small input table before using large research datasets
3. Follow the tutorials in order for the smoothest onboarding
4. Refer to the :doc:`../commands/index` pages when you want option-level details

Pick a Tutorial by Goal
-----------------------

- If you want a first successful run, start with :doc:`basic_workflow`
- If you want better interpretation, go to :doc:`machine_learning`
- If you want to choose among Mode 1, 2, and 3, read :doc:`advanced_analysis`
- If you want to extend Surfacia itself, read :doc:`custom_descriptors`
