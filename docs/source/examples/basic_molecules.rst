Basic Molecules
===============

This example category is for simple molecules that make the main Surfacia ideas easy to inspect.

Good Starting Systems
---------------------

- small neutral organics
- single functional-group substitutions
- simple aromatic molecules

Suggested Mini Set
------------------

A useful first example set is:

- caffeine
- aspirin
- ibuprofen

These are helpful because they already differ in:

- size and shape
- heteroatom content
- polarity distribution
- functional-group composition

Example Input Table
-------------------

.. code-block:: text

   Sample Name,SMILES
   caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
   ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O

Suggested Command Sequence
--------------------------

.. code-block:: bash

   surfacia workflow -i molecules.csv --resume --test-samples "1,2"

Then inspect the main outputs:

- complete descriptor table
- compact model outputs
- SHAP results

What to Compare First
---------------------

For a small three-molecule set, start with descriptors that are easy to reason about:

- ``Molecule Weight``
- ``Isosurface Area``
- ``ESP_min`` and ``ESP_max``
- ``ALIE_min``
- selected ``Fun_*`` descriptors

What These Examples Are Good For
--------------------------------

- verifying that the workflow runs correctly
- learning how descriptor names map to chemistry
- checking how size, shape, and surface electronics change across simple structures

What You Can Usually Observe
----------------------------

With a small molecule set like this, you can often inspect:

- how ``Molecule Weight`` and ``Isosurface Area`` change with scaffold size
- how ``ESP`` and ``ALIE`` extrema shift with carbonyls, acids, or heterocycles
- how ``Fun_*`` descriptors summarize chemically distinct groups

How to Read the Results
-----------------------

Ask a few simple questions:

1. which molecule is the largest and most surface-exposed?
2. which molecule has the strongest polar or electron-rich region?
3. do the retained compact features match the chemistry you would expect from the structures?
4. can you explain the differences without reading every descriptor column?

If the answer to the last question is yes, then Surfacia's descriptor language is already starting to work for you.

Simple Interpretation Template
------------------------------

When writing up your own notes, a short template like this usually works well:

1. **Size and shape**
   Which molecule appears larger, more compact, or more anisotropic?
2. **Surface electronics**
   Which molecule shows the strongest electron-rich or electron-poor region?
3. **Functional grouping**
   Do the ``Fun_*`` descriptors separate the molecules in a chemically intuitive way?
4. **Compact model**
   If only a few descriptors survive, do they match what you would have expected from the structures?

What Counts as a Healthy Output
-------------------------------

For a small sanity-check example, a healthy result usually means:

- the workflow finishes cleanly
- descriptor values differ across molecules in plausible ways
- extreme values such as ``ESP_min`` or ``ALIE_min`` are not obviously nonsensical
- the retained features are easier to explain than the full feature table

Why This Example Matters
------------------------

Basic molecules are not only for testing installation. They are often the fastest way to understand whether Surfacia's descriptor language is intuitive to you before moving on to larger systems.
