shap-viz
========

``surfacia shap-viz`` launches the interactive SHAP dashboard from a detailed training CSV. It can also overlay external test-set points and SPES-ranked candidates.

Synopsis
--------

.. code-block:: bash

   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_files [OPTIONS]

Required Inputs
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Input
     - Description
   * - ``-i, --input``
     - Training detailed CSV containing ``Feature_*`` and matching ``SHAP_*`` columns.
   * - ``-x, --xyz-folder``
     - Folder containing molecular structure files used by the dashboard.

Common Options
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--test-csv``
     - Optional ``Test_Set_Detailed*.csv`` file for raw external test-set overlay.
   * - ``--spes-csv``
     - Optional ``SPES_Test_Set_Detailed*.csv`` file for SPES overlay.
   * - ``--api-key``
     - Optional ZhipuAI API key for the assistant panel.
   * - ``--host``
     - Web server host. Use ``0.0.0.0`` for remote access.
   * - ``--port``
     - Web server port.

Examples
--------

Training set only:

.. code-block:: bash

   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_files

With raw test-set overlay:

.. code-block:: bash

   surfacia shap-viz \
     -i Training_Set_Detailed.csv \
     -x ./xyz_files \
     --test-csv Test_Set_Detailed.csv

With SPES overlay:

.. code-block:: bash

   surfacia shap-viz \
     -i Training_Set_Detailed.csv \
     -x ./xyz_files \
     --test-csv Test_Set_Detailed.csv \
     --spes-csv SPES_Test_Set_Detailed.csv

Input Requirements
------------------

The training CSV should normally be produced by ``surfacia ml-analysis`` or ``surfacia workflow``. It must contain:

- sample identifiers
- ``Feature_*`` columns
- matching ``SHAP_*`` columns
- prediction and target columns when available

The SPES CSV is generated automatically by the ML stage when a test set is present. For interpretation details, see :doc:`../user_guide/spes`.

Dashboard Use
-------------

Use the feature selector to inspect a descriptor, then choose the external overlay mode:

- no overlay
- raw test set
- SPES layer, when ``--spes-csv`` is provided

The SPES layer colors external points by ``SPES_Score`` when that column is available.

Troubleshooting
---------------

- If the page does not open, try another port with ``--port 8053``.
- If remote access fails, start with ``--host 0.0.0.0`` and check firewall settings.
- If SPES does not appear in the overlay menu, confirm that ``--spes-csv`` points to an existing ``SPES_Test_Set_Detailed*.csv`` file.
- If structures are missing, check that ``-x`` points to the folder containing the matching xyz or generated surface files.
