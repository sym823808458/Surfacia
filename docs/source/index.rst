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

**Surfacia** is an automated framework for surface-based feature engineering and interpretable machine learning with reasoning language model integration. It addresses the critical interpretability gap in structure-activity relationship analysis by systematically extracting quantitative descriptors across atomic, functional group, and molecular levels.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Quick Start
      :link: getting_started/index
      :link-type: doc

      Get up and running with Surfacia in minutes. Learn the basics and run your first analysis.

   .. grid-item-card:: 📖 User Guide
      :link: user_guide/index
      :link-type: doc

      Comprehensive guides for using all Surfacia features effectively.

   .. grid-item-card:: 💻 Commands Reference
      :link: commands/index
      :link-type: doc

      Detailed documentation for all CLI commands with examples and best practices.

   .. grid-item-card:: 🧪 Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step tutorials for common workflows and advanced techniques.

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

The Interpretability Challenge
------------------------------

Modern computational chemistry faces a fundamental paradox: while machine learning models achieve unprecedented predictive accuracy in drug discovery, they operate as black boxes providing little insight into chemical principles governing molecular behavior. This interpretability crisis is particularly problematic in structure-activity relationship analysis, where understanding *why* molecular modifications enhance or diminish properties is essential for rational design.

.. admonition:: The Solution
   :class: tip

   Surfacia bridges this gap by leveraging surface-based molecular properties such as local electron attachment energies, electrostatic potential distributions, and average local ionization energies. These properties directly encode electronic features responsible for molecular recognition and reactivity, providing a natural foundation for interpretable analysis.

Scientific Foundation
---------------------

Surface-based molecular interactions are fundamental to pharmaceutical activity. Our framework systematically quantifies these interactions through:

**Multi-Scale Descriptor Generation**
   - **Atomic Level**: Individual atom surface properties
   - **Functional Group Level**: Chemical fragment characteristics  
   - **Molecular Level**: Global molecular properties

**Quantum Mechanical Foundation**
   - Gaussian quantum chemistry calculations
   - Multiwfn wavefunction analysis
   - Surface-based property mapping

**Interpretable Feature Engineering**
   - Intelligent stepwise feature selection
   - Chemical interpretability preservation
   - Minimal feature sets with maintained predictive power

Beyond Prediction: Chemical Insight
------------------------------------

Surfacia addresses the critical "beyond-prediction" challenge by:

- **Identifying High-Potential Regions**: Systematic analysis of surface properties in top-performing molecules
- **Automated Chemical Interpretation**: Language model integration for natural language explanations
- **Extrapolative Design Guidance**: SHAP-based identification of molecular regions with exceptional potential

Installation
------------

.. code-block:: bash

   # Install from PyPI (recommended)
   pip install surfacia

   # Or install from source
   git clone https://github.com/surfacia/surfacia.git
   cd surfacia
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
   :caption: Contents
   :hidden:

   getting_started/index
   user_guide/index
   commands/index
   tutorials/index
   descriptors/index
   api/index
   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   citation
   contributing
   changelog
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`