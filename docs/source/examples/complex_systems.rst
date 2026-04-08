Complex Systems
===============

This example category covers larger or more chemically structured systems where interpretation depends on local context.

Representative Use Cases
------------------------

- catalyst families
- heteroatom-rich molecules
- scaffold-conserved series
- molecules with multiple competing functional regions

Representative Surfacia Scenarios
---------------------------------

The following problem types are especially well matched to Surfacia:

- sulfur-containing modifier studies where one element is chemically central
- homologous catalyst series where a shared fragment defines the chemistry
- diverse property datasets where local and global effects compete

How to Decide Which Mode to Try
-------------------------------

Use this quick rule of thumb:

- if one element is already central, start with **Mode 1**
- if one fragment or scaffold is central, start with **Mode 2**
- if the chemistry is broad or unclear, start with **Mode 3**

Suggested Working Sequence
--------------------------

For a more complex project, a good strategy is:

1. run one broad baseline analysis
2. inspect the descriptor families that matter most
3. rerun with a more hypothesis-aware mode if the chemistry suggests it
4. compare compact retained feature sets rather than only comparing scores

Result Interpretation Template
------------------------------

For a complex system, this short reading pattern is often helpful:

1. **representation**
   Did you choose a mode that matches the chemistry?
2. **retained descriptors**
   Are the retained descriptors local, global, or mixed?
3. **coherence**
   Do those descriptors tell one story or several disconnected stories?
4. **stability**
   Would you still trust the interpretation if the held-out split changed?

How to Approach Them
--------------------

For complex systems, start by deciding whether your chemistry is best represented by:

- an element-specific view
- a fragment-specific view
- a broader exploratory LOFFI view

In larger systems, this choice often matters more than fine-tuning the model first.

Why They Matter
---------------

These systems are where multi-scale analysis becomes especially valuable because whole-molecule descriptors alone are often not enough.

Typical Payoff
--------------

Complex systems are often where Surfacia becomes most useful, because it can connect:

- global size and shape
- local electronic structure
- fragment-level interpretation
- compact model outputs that still support chemical reasoning

What to Watch For
-----------------

Complex systems are also where misleading interpretation can happen most easily.

Be careful when:

- the test set is tiny
- one experimental condition dominates the model
- many descriptors survive but do not form a coherent story
- the representation does not match the chemistry of the dataset

What Often Works Well
---------------------

Strong complex-system analyses often show:

- a representation choice that clearly fits the chemistry
- a compact feature subset rather than a diffuse large model
- descriptors spanning the scales that matter for the problem
- SHAP trends that can be translated into mechanistic language
