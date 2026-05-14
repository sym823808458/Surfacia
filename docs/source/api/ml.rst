Machine Learning API
====================

The machine-learning layer handles compact model construction, feature selection, validation, and interpretability analysis.

What Belongs Here
-----------------

- feature selection
- model training and evaluation
- cross-validation logic
- SHAP-based interpretation

Typical Responsibilities
------------------------

- reduce large feature matrices to compact subsets
- fit predictive models
- compare model behavior across splits
- generate outputs that remain chemically interpretable

When to Use the ML API
----------------------

Use this layer when you want to:

- train models directly from feature tables
- test alternative feature-selection settings
- integrate Surfacia features into custom ML code
- inspect prediction and explanation outputs programmatically
