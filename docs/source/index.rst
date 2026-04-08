Surfacia Documentation
======================

.. raw:: html

   <div class="surfacia-hero">
   <p class="surfacia-hero-kicker">Documentation</p>
   <img class="surfacia-hero-logo" src="_static/images/surfacia_logo.png" alt="Surfacia logo">
   <p class="surfacia-hero-subtitle">Install quickly, run the workflow, and debug efficiently on local or remote Linux.</p>
   <div class="surfacia-hero-actions">
   <a class="surfacia-hero-button surfacia-hero-button-primary" href="getting_started/installation.html">Install</a>
   <a class="surfacia-hero-button surfacia-hero-button-secondary" href="https://github.com/sym823808458/Surfacia">GitHub</a>
   </div>
   </div>

Quick Navigation
----------------

- New users: :doc:`getting_started/index`
- Daily usage and troubleshooting: :doc:`user_guide/index`
- Command and parameter lookup: :doc:`commands/index`
- Reproducible workflows and templates: :doc:`examples/index`
- MCP server and agent integration: :doc:`integrations/index`

Project Links
-------------

- GitHub: https://github.com/sym823808458/Surfacia
- PyPI: https://pypi.org/project/surfacia/
- Academic homepage: https://sym823808458.github.io/yumingsu_homepage/

Essential Commands
------------------

.. code-block:: bash

   # Install
   pip install surfacia

   # Full workflow
   surfacia workflow -i molecules.csv --test-samples "1,3"

   # ML-only rerun
   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv --test-samples "1,2,3"

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting_started/index
   user_guide/index
   commands/index
   tutorials/index
   integrations/index
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
