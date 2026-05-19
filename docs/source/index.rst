Surfacia Documentation
======================

.. raw:: html

   <div class="surfacia-hero">
   <p class="surfacia-hero-kicker">Documentation</p>
   <img class="surfacia-hero-logo" src="_static/images/surfacia_logo.png" alt="Surfacia logo">
   <p class="surfacia-hero-subtitle">A practical manual for installing Surfacia, running the molecular surface workflow, and interpreting outputs.</p>
   <div class="surfacia-hero-actions">
   <a class="surfacia-hero-button surfacia-hero-button-primary" href="https://pypi.org/project/surfacia/3.0.3/">Install from PyPI</a>
   <a class="surfacia-hero-button surfacia-hero-button-secondary" href="getting_started/quick_start.html">Quick Start</a>
   <a class="surfacia-hero-button surfacia-hero-button-secondary" href="https://github.com/sym823808458/Surfacia">GitHub</a>
   </div>
   </div>

Choose Your Path
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install and Run
      :link: getting_started/index
      :link-type: doc
      :class-card: surfacia-home-card

      Install the package, check external programs, and run the first workflow.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc
      :class-card: surfacia-home-card

      Learn day-to-day workflows, result interpretation, SPES usage, and troubleshooting.

   .. grid-item-card:: Command Reference
      :link: commands/index
      :link-type: doc
      :class-card: surfacia-home-card

      Look up CLI commands, required inputs, common options, and examples.

   .. grid-item-card:: API and MCP
      :link: api/index
      :link-type: doc
      :class-card: surfacia-home-card

      Use Python APIs directly, or connect Surfacia through the MCP server.

Minimal Commands
----------------

.. code-block:: bash

   pip install surfacia==3.0.3
   surfacia workflow -i molecules.csv --test-samples "1,3"
   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_files --test-csv Test_Set_Detailed.csv

Manual Structure
----------------

- :doc:`getting_started/index`: installation, first run, and core concepts
- :doc:`user_guide/index`: practical workflows, SPES guide, best practices, and troubleshooting
- :doc:`commands/index`: command-line reference
- :doc:`api/index`: Python API reference
- :doc:`integrations/index`: MCP server and automation integration
- :doc:`descriptors/index`: descriptor naming and interpretation
- :doc:`examples/index`: worked examples and reusable templates

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/index
   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: References
   :hidden:

   commands/index
   api/index
   descriptors/index

.. toctree::
   :maxdepth: 2
   :caption: Workflows
   :hidden:

   tutorials/index
   examples/index
   integrations/index

.. toctree::
   :maxdepth: 1
   :caption: Project
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
