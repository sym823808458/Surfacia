Machine Learning Tutorial
=========================

This tutorial focuses on how Surfacia turns descriptor tables into compact and interpretable machine-learning models.

Topics
------

- feature selection
- compact model construction
- validation and split-aware interpretation
- SHAP-based explanation

Recommended Workflow
--------------------

The most reliable Surfacia modeling loop is usually:

1. start from a descriptor table you trust
2. build a compact model instead of keeping every feature
3. inspect retained descriptors before celebrating the score
4. use SHAP to understand direction and magnitude of contributions

How to Judge a Model
--------------------

A useful Surfacia model is not only accurate. It should also be readable.

Good signs:

- the final feature subset is much smaller than the raw matrix
- retained descriptors are chemically meaningful
- SHAP patterns are consistent with known chemistry or plausible hypotheses

Warning signs:

- strong metric swings caused by a tiny test set
- top features that are difficult to interpret physically
- a model that improves numerically but becomes chemically opaque

Small-Data Reality
------------------

Some Surfacia use cases are inherently small-data problems. In that setting:

- cross-validation is often more informative than one held-out split
- unstable test-set metrics do not automatically invalidate the descriptor idea
- descriptor coherence may matter as much as raw score

What to Look for in SHAP
------------------------

For each important descriptor, ask:

- does higher or lower value help?
- is the relationship approximately linear or threshold-like?
- is the effect global or only visible in part of the dataset?
- can the trend be translated into a design idea?

What to Watch For
-----------------

- a smaller feature set can be more interpretable without sacrificing utility
- unstable test metrics are common in very small datasets
- the retained descriptors often tell a more useful story than the full matrix

Practical Outcome
-----------------

The best outcome of this tutorial is not just a score table. It is a short list of chemically interpretable descriptors that you would actually be willing to discuss or act on.
