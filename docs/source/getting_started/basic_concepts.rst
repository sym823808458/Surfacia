Basic Concepts
==============

Understanding the core ideas behind Surfacia will help you use the framework more effectively and interpret results in a chemically meaningful way.

The Interpretability Problem
----------------------------

Many machine-learning models in chemistry can predict well but remain difficult to explain in a way that is useful for molecular design.

.. admonition:: Why This Matters
   :class: warning

   In practical molecular discovery, chemists usually want more than prediction alone.

   - **Which** structural features drive the response?
   - **Why** does a modification help or hurt?
   - **How** can the result guide the next round of design?

Why Surfacia Focuses on Molecular Surfaces
------------------------------------------

Surfacia is built around the idea that many chemically important processes are expressed at the molecular surface.

- **Recognition** depends on surface complementarity
- **Reactivity** depends on local electronic structure
- **Accessibility** depends on size and shape
- **Property trends** often reflect a balance of global and local surface effects

Key Surface Properties
----------------------

Surfacia extracts three central surface-electronic quantities.

**Local Electron Attachment Energy (LEAE)**
   Reports local electron-accepting tendency and helps identify electrophilic character.

**Electrostatic Potential (ESP)**
   Describes surface charge distribution and highlights electron-rich or electron-poor regions.

**Average Local Ionization Energy (ALIE)**
   Reports local resistance to electron removal and helps identify electron-donating regions.

Multi-Scale Descriptor Design
-----------------------------

Surfacia organizes descriptors across three hierarchical levels so that the final model can remain interpretable.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 🔬 Atomic Level

      Local atom-centered statistics across the molecular surface.

      - site-specific reactivity
      - atom-level extremes and averages
      - local electronic variation

   .. grid-item-card:: 🧩 Functional Group Level

      Descriptors summarized over chemical fragments or automatically detected groups.

      - fragment-centered interpretation
      - modular structure-property reasoning
      - comparison between functional units

   .. grid-item-card:: 🌐 Molecular Level

      Global descriptors for overall molecular character.

      - size and shape
      - whole-molecule electronics
      - bulk surface behavior

Choosing the Right Analysis Mode
--------------------------------

Surfacia provides three complementary modes so that the descriptor strategy can follow the chemistry of your problem.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Mode 1: Element-Specific

      Best when one element is already known to be mechanistically important.

      - element-aware hypothesis testing
      - heteroatom-focused studies
      - compact descriptors centered on a chosen element

   .. grid-item-card:: Mode 2: Fragment-Specific

      Best when a catalytic core, scaffold, or pharmacophore is known in advance.

      - user-defined fragment analysis
      - scaffold-conserved series
      - substituent perturbation around a fixed motif

   .. grid-item-card:: Mode 3: Automated LOFFI

      Best when no strong mechanistic prior is available.

      - automated atom-level and functional-group-level analysis
      - exploratory studies on diverse datasets
      - broad discovery without predefining a key motif

The Workflow in Plain Language
------------------------------

Surfacia is designed as an end-to-end pipeline from molecular structures to interpretable design insight.

.. mermaid::

   graph LR
       A[Input Structures] --> B[3D Generation]
       B --> C[QM Calculation]
       C --> D[Surface Property Mapping]
       D --> E[Descriptor Generation]
       E --> F[Compact Modeling]
       F --> G[SHAP Interpretation]
       G --> H[Interactive Analysis]

Typical stages:

1. Build or read 3D structures from the input table
2. Run quantum calculations and generate wavefunction outputs
3. Extract molecular, atom-level, element-specific, or fragment-specific descriptors
4. Select compact feature subsets
5. Train interpretable predictive models
6. Explain predictions with SHAP and optional language-model-assisted summaries

Compact Models, Not Just Large Feature Matrices
-----------------------------------------------

Surfacia is designed to reduce descriptor dimensionality while keeping the final model chemically readable.

- **Consensus-driven feature selection** avoids over-reliance on one split or one ranking
- **Compact descriptor subsets** are favored over oversized feature matrices
- **SHAP analysis** preserves both global and sample-level interpretation
- **Trend fitting** can distinguish linear, threshold-like, and saturating relationships

Descriptor Naming Guide
-----------------------

Feature names are intentionally systematic so users can infer their meaning directly.

- ``Atom_*``: atom-level global statistics over the molecular surface
- ``Fun_*``: functional-group-level statistics from automated grouping
- ``Fragment_*``: descriptors for a user-defined fragment
- element-prefixed terms: descriptors aggregated around a selected element
- ``*_min`` / ``*_max``: chemically extreme regions
- ``*_mean`` / ``*_average``: overall character
- ``*_delta``: heterogeneity across sites or groups

What Makes Surfacia Different
-----------------------------

Surfacia is meant to be more than a prediction script.

- It keeps descriptors tied to recognizable physicochemical meaning
- It supports hypothesis-aware and hypothesis-free analysis
- It emphasizes compact models that remain chemically interpretable
- It connects numerical outputs to design-oriented explanation

Next Steps
----------

1. Try the :doc:`quick_start` page for a minimal working workflow
2. Read :doc:`../descriptors/index` for descriptor families and naming
3. Explore :doc:`../commands/index` for command-level details
4. Use :doc:`../tutorials/index` when you want guided examples
