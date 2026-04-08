Custom Descriptor Tutorial
==========================

This tutorial explains how to think about extending Surfacia with additional descriptors while preserving interpretability.

Design Principles
-----------------

- keep descriptors physically meaningful
- avoid redundant features when possible
- document naming and units clearly
- ensure outputs remain compatible with compact modeling

What Makes a Good Surfacia Descriptor
-------------------------------------

A useful descriptor should usually satisfy most of these conditions:

- it has a clear physicochemical interpretation
- it can be compared across molecules consistently
- it adds information not already captured by many existing features
- it remains understandable after SHAP-based interpretation

Good Candidates
---------------

- descriptors tied to a known chemical hypothesis
- additional local summaries over meaningful structural units
- derived quantities that help interpretation rather than obscure it

Less Helpful Additions
----------------------

Be cautious about adding descriptors that:

- duplicate an existing feature under a different name
- are difficult to explain chemically
- depend strongly on arbitrary preprocessing choices
- produce large feature growth without a clear interpretation benefit

Naming Suggestions
------------------

Follow the existing Surfacia style whenever possible:

- use meaningful prefixes such as ``Atom_``, ``Fun_``, or ``Fragment_``
- make units and scale easy to infer from the documentation
- keep statistical suffixes systematic: ``_min``, ``_max``, ``_mean``, ``_delta``

Validation Checklist
--------------------

Before treating a new descriptor as useful, ask:

1. does it vary meaningfully across molecules?
2. does it stay numerically stable?
3. does it survive compact feature selection in at least some tasks?
4. can a chemist understand what a high or low value means?
