Advanced Analysis Tutorial
==========================

This tutorial focuses on situations where you want more control over Surfacia's analysis strategy.

Topics
------

- choosing among Mode 1, Mode 2, and Mode 3
- comparing descriptor subsets
- reading compact model outputs more critically
- interpreting nonlinear SHAP trends

When You Need This Tutorial
---------------------------

Move beyond the basic workflow when:

- one element is already central to your hypothesis
- one fragment or scaffold should be analyzed as the chemically meaningful unit
- the default exploratory workflow produces too many equally plausible explanations
- you need to compare hypothesis-aware and hypothesis-free descriptor strategies

Step 1: Choose the Right Mode
-----------------------------

Use the problem structure to choose the descriptor strategy.

**Mode 1**
  Best for element-centered questions such as sulfur-, fluorine-, or metal-related hypotheses.

**Mode 2**
  Best for scaffold-conserved systems where a known fragment or catalytic core should stay central.

**Mode 3**
  Best for broad discovery problems where you do not want to assume the mechanism in advance.

Step 2: Compare Representations, Not Just Scores
------------------------------------------------

A more useful comparison is often:

- which descriptors survive compact selection
- whether the retained descriptors tell a coherent chemical story
- whether the model becomes easier or harder to explain

In small datasets especially, a slightly weaker metric can still be more valuable if the retained features are chemically legible.

Step 3: Read SHAP More Carefully
--------------------------------

For advanced use, avoid stopping at the ranked feature list.

Look for:

- threshold-like behavior
- saturation effects
- sign reversals across a value range
- consistent chemistry across related molecules

These patterns often matter more than raw feature ranking.

Practical Comparison Strategy
-----------------------------

For one dataset, a good advanced workflow is:

1. run the default broad analysis
2. test a more hypothesis-aware representation if the chemistry suggests one
3. compare compact retained features across runs
4. keep the representation that is both interpretable and stable enough to support the research question

Best For
--------

- mechanism-aware studies
- scaffold-conserved series
- users comparing hypothesis-aware and exploratory workflows

Warning Signs
-------------

Treat results cautiously when:

- the retained features change wildly across splits
- the test set is extremely small
- the model depends heavily on one external condition rather than molecular descriptors
- SHAP explanations look mathematically clear but chemically implausible
