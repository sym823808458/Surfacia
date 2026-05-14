Molecular Descriptors Reference
===============================

Surfacia generates descriptors across molecular, atom-level, element-specific, and fragment-specific levels so that the final model can stay both compact and chemically interpretable.

.. toctree::
   :maxdepth: 2

   size_and_shape
   electronic_properties
   surface_analysis

Overview
--------

The descriptor system is designed to stay:

- **physically grounded** through wavefunction-derived surface properties
- **hierarchical** across global and local levels
- **task-adaptive** through three analysis modes
- **interpretable after selection** rather than only before modeling

Descriptor Categories
---------------------

**1. Size and Shape**

Descriptors for physical extent, anisotropy, compactness, and planarity.

- molecular weight
- isosurface area
- molecular dimensions
- sphericity and asphericity
- radius of gyration related descriptors

**2. Electronic Properties**

Descriptors for frontier orbitals, orbital delocalization, and whole-molecule electronic structure.

- HOMO / LUMO
- HOMO-LUMO gap
- orbital delocalization indices
- global electronic indicators

**3. Multi-Scale Surface Analysis**

Descriptors for local surface electronics and task-specific interpretation.

- element-specific descriptors
- fragment-specific descriptors
- automated LOFFI atom-level and functional-group-level descriptors

How to Choose a Mode
--------------------

Use the analysis mode that matches the amount of prior chemical knowledge you already have.

1. **Mode 1: Element-Specific**
   Best when one element is already suspected to be mechanistically important.
2. **Mode 2: Fragment-Specific**
   Best when a shared fragment, catalytic core, or pharmacophore is already known.
3. **Mode 3: LOFFI Automated**
   Best when you want broad exploratory analysis without hard-coding a mechanism first.

Naming Conventions
------------------

The feature names are intentionally systematic.

.. csv-table::
   :header: "Prefix or Suffix", "Meaning"
   :widths: 30, 70

   "``Atom_``", "Atom-level global statistics over the molecular surface"
   "``Fun_``", "Functional-group-level statistics from automated LOFFI grouping"
   "``Fragment_``", "Descriptors calculated for a user-defined fragment"
   "element-prefixed names", "Descriptors aggregated around a selected element in Mode 1"
   "``*_min`` / ``*_max``", "Most chemically extreme sites or groups"
   "``*_mean`` / ``*_average``", "Overall average behavior"
   "``*_delta``", "Heterogeneity or spread across sites, groups, or atoms"

Examples of Direct Chemical Meaning
-----------------------------------

.. csv-table::
   :header: "Descriptor", "Meaning", "How to Read It"
   :widths: 28, 34, 38

   "``ALIE_min``", "Most nucleophilic region", "Lower values often mean easier electron donation"
   "``ESP_max``", "Most electron-deficient region", "Higher values often indicate stronger electrophilic character"
   "``Fun_ESP_delta``", "Electrostatic contrast between functional groups", "Larger values suggest stronger intramolecular push-pull separation"
   "``Shape_Geometric_Asphericity``", "Pure geometric anisotropy", "Higher values mean less sphere-like shape"

Why This Matters for Modeling
-----------------------------

Surfacia descriptors are meant to support more than predictive accuracy.

- they keep the model tied to recognizable chemistry
- they make SHAP plots easier to interpret
- they support compact feature subsets instead of oversized black-box matrices
- they help translate numerical results into design-oriented reasoning

See Also
--------

- :doc:`size_and_shape` for geometry, compactness, and planarity
- :doc:`electronic_properties` for frontier orbitals and electronic structure
- :doc:`surface_analysis` for MQSA modes and surface-electronic descriptors
- :doc:`../getting_started/basic_concepts` for the conceptual overview
