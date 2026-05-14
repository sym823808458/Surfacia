Descriptors API
===============

The descriptors layer is responsible for converting wavefunction and geometry information into chemically interpretable numerical features.

What Belongs Here
-----------------

- size and shape descriptor generation
- electronic descriptor calculation
- surface-electronic statistics
- multi-scale quantitative surface analysis

Typical Responsibilities
------------------------

- collect molecule-level descriptors
- calculate atom-level and functional-group-level summaries
- support element-specific and fragment-specific analysis
- organize outputs into model-ready feature matrices

When to Use the Descriptors API
-------------------------------

Use this layer when you want to:

- generate features outside the full CLI workflow
- experiment with descriptor subsets
- compare different analysis modes
- extend Surfacia with new chemically meaningful descriptors
