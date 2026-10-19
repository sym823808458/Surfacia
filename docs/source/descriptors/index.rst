Choose a descriptor mode
============================================================

Choose regions before inspecting test outcomes. Modes are software capabilities,
not evidence that any mode always beats other representations.

.. list-table::
   :header-rows: 1
   :widths: 12 36 52

   * - Mode
     - Input choice
     - Question
   * - 1
     - Element, e.g. ``--element S``
     - How do element-centered local environments differ?
   * - 2
     - Aligned reference XYZ, ``--xyz1 fragment.xyz``
     - How does the environment of a common fragment vary?
   * - 3
     - Automated LOFFI features
     - What local patterns occur without a chosen element or fragment?

ESP is electrostatic potential; ALIE is average local ionization energy; LEAE
is local electron attachment energy. Read the actual column, region and generating
calculation together. Counts and units depend on mode and upstream outputs.
The extraction threshold does not replace reporting wavefunction method,
electron-density isovalue and grid. Investigate missing/unphysical values before
using automatic column deletion. ML format: :doc:`../getting_started/quick_start`.
