Examples
========

This section is a compact template library. Pick one example and run it directly.

Recommended Order
-----------------

1. :doc:`basic_molecules`
2. :doc:`machine_learning`
3. :doc:`mode3_top20_remote_debug`

Use Case Map
------------

- Learn the full pipeline quickly: :doc:`basic_molecules`
- Focus on modeling and SHAP: :doc:`machine_learning`
- Reproduce remote Linux debugging workflow: :doc:`mode3_top20_remote_debug`
- Need larger systems: :doc:`complex_systems`
- Need custom orchestration: :doc:`custom_workflows`

Fast Start Commands
-------------------

.. code-block:: bash

   # Full workflow
   surfacia workflow -i molecules.csv --test-samples "1,3"

   # ML only
   surfacia ml-analysis -i FinalFull_Mode3_20_168.csv --test-samples "1,2,3"

   # Visualization
   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_files/

.. toctree::
   :maxdepth: 2
   :caption: Example Pages

   basic_molecules
   complex_systems
   machine_learning
   custom_workflows
   mode3_top20_remote_debug
