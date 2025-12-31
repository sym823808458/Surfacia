Surfacia: Surface-Based Feature Engineering and Interpretable Machine Learning
==============================================================================

.. image:: _static/images/surfacia_logo.png
   :alt: Surfacia Framework
   :align: center
   :width: 400px
   :class: main-logo

.. raw:: html

   <div style="text-align: center; margin: 2rem 0;">
   <h1 style="font-size: 3rem; color: #2563eb; margin: 1rem 0;">Surfacia</h1>
   <p style="font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">Surface-Based Feature Engineering and Interpretable Machine Learning</p>
   </div>

**Surfacia** is an automated framework for surface-based feature engineering and interpretable machine learning with reasoning language model integration. It addresses critical interpretability gap in structure-activity relationship analysis by systematically extracting quantitative descriptors across atomic, functional group, and molecular levels.

Key Features
------------

🔬 **Surface-Based Analysis**
   Systematic extraction of hierarchical surface descriptors from quantum mechanical calculations

🤖 **Interpretable Machine Learning**
   SHAP-based explainable AI with intelligent feature selection maintaining both predictive power and chemical interpretability

🧠 **AI Assistant Integration**
   Large language model integration for automated chemical interpretation and natural language explanations

⚡ **Automated Workflow**
   Complete 8-step pipeline from SMILES to interpretable predictions with intelligent resume capabilities

🔧 **Comprehensive Toolkit**
   Molecular visualization, batch processing, and error recovery tools

Installation
------------

.. code-block:: bash

   # Install from PyPI (recommended)
   pip install surfacia

   # Or install from source
   git clone https://github.com/sym823808458/Surfacia.git
   cd Surfacia
   pip install -e .

Quick Example
-------------

.. code-block:: bash

   # Complete workflow from SMILES to interpretable predictions
   surfacia workflow -i molecules.csv --resume --test-samples "1,2,3"

   # Individual analysis steps
   surfacia ml-analysis -i processed_data.csv --test-samples "1,2,3"
   surfacia shap-viz -i training_data.csv --api-key YOUR_API_KEY

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/basic_workflow

.. toctree::
   :maxdepth: 2
   :caption: Descriptors
   :hidden:

   descriptors/index
   descriptors/mqsa_modes

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   citation

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
