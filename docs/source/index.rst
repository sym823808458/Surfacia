Surfacia documentation
============================================================

Molecular surface descriptors, predictive modeling, and chemical interpretation.

Surfacia connects molecular structures and quantum calculations to whole-molecule,
surface and local descriptors, compact XGBoost models and SHAP analysis.
The surface is a **molecular surface**, not a solid catalyst interface simulation.

Choose your starting point
------------------------------------------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: I have a descriptor table
      :link: getting_started/quick_start
      :link-type: doc

      Run a downloadable example, without Gaussian or Multiwfn.

   .. grid-item-card:: I have molecular structures
      :link: user_guide/workflows
      :link-type: doc

      Configure chemistry tools, choose a local region and inspect each stage.

   .. grid-item-card:: I want to read an explanation
      :link: examples/interpretation
      :link-type: doc

      Inspect original predictions, SHAP trends and structures together.

   .. grid-item-card:: I am upgrading
      :link: migration
      :link-type: doc

      Check versions and removal of the former candidate-ranking extension.

.. image:: /_static/images/workflow.svg
   :alt: Structures, molecular surfaces, descriptors, models and interpretation
   :class: workflow-figure

This manual describes the **4.0.0 source revision**. Check the installed version;
an old pip installation does not change when the website changes.

`GitHub <https://github.com/sym823808458/Surfacia>`_ ·
`PyPI <https://pypi.org/project/surfacia/>`_ ·
`Yuming Su <http://yumingsu.top/>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/installation
   getting_started/quick_start
   getting_started/basic_concepts
   user_guide/workflows
   descriptors/index
   user_guide/results
   examples/index
   commands/index
   api/index
   integrations/index
   user_guide/best_practices
   user_guide/troubleshooting
   migration
   citation
   changelog
   contributing
   license
