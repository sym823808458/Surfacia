Surface Analysis Descriptors
============================

This page describes the Surfacia descriptors that come directly from molecular surface electronic structure and from multi-scale quantitative surface analysis.

Overview
--------

Surface descriptors are central to Surfacia because many chemically important processes are expressed at the molecular surface:

- intermolecular recognition
- steric accessibility
- electrostatic complementarity
- local electron donation and acceptance

Core Surface Properties
-----------------------

Surfacia uses three main surface-electronic quantities.

**ESP (Electrostatic Potential)**
  - Describes charge distribution on the molecular surface
  - ``ESP_min`` usually marks the most electron-rich region
  - ``ESP_max`` usually marks the most electron-deficient region

**ALIE (Average Local Ionization Energy)**
  - Reports local resistance to electron removal
  - Lower values usually indicate easier electron donation

**LEAE (Local Electron Attachment Energy)**
  - Reports local tendency to accept electrons
  - Useful for identifying electrophilic character and local acceptor strength

Common Statistics
-----------------

For a given property, Surfacia frequently reports:

- ``*_min``: most extreme low-value region
- ``*_max``: most extreme high-value region
- ``*_mean`` or ``*_average``: overall character
- ``*_delta``: heterogeneity across sites
- ``*_variance``: statistical spread across the surface

Three MQSA Modes
----------------

Mode 1: Element-Specific
~~~~~~~~~~~~~~~~~~~~~~~~

Use this mode when one element is already known to be chemically central.

- generates 13 descriptors
- aggregates surface properties over atoms of a chosen element
- useful for element-aware hypothesis testing

Mode 2: Fragment-Specific
~~~~~~~~~~~~~~~~~~~~~~~~~

Use this mode when you already know the important fragment, catalytic core, or pharmacophore.

- generates 18 descriptors
- uses the naming pattern ``Fragment_[Property]_[Statistic]``
- captures how a fixed local motif is perturbed by the surrounding structure

Mode 3: LOFFI Automated
~~~~~~~~~~~~~~~~~~~~~~~

Use this mode when you want broad exploratory analysis without imposing a mechanism first.

- generates 32 descriptors
- combines atom-level and functional-group-level summaries
- is best suited to diverse molecular datasets

How to Read the Names
---------------------

**Atom_ descriptors**
  - global atom-level summaries over the full molecular surface
  - example: ``Atom_ALIE_mean``

**Fun_ descriptors**
  - statistics over automatically detected functional groups
  - example: ``Fun_ESP_delta``

**Fragment_ descriptors**
  - statistics for a user-defined fragment
  - example: ``Fragment_ESP_mean``

**Element-centered descriptors**
  - statistics aggregated over a selected element in Mode 1
  - example: element-specific area, min, max, mean, and delta features

What the Modes Are Good For
---------------------------

Each mode answers a slightly different question:

- **Mode 1**: "Is this particular element driving the chemistry?"
- **Mode 2**: "How do substituents perturb a shared reactive fragment?"
- **Mode 3**: "What local and global surface features matter when I do not want to assume a mechanism first?"

Why These Descriptors Matter
----------------------------

In practical modeling, surface descriptors often provide the most chemically interpretable bridge between:

- quantum calculations
- compact machine-learning models
- SHAP explanations
- human-readable design insight

They are especially useful when you want to explain not only whether a molecule performs well, but also which regions and which physicochemical patterns are likely responsible.
