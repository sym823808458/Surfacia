Custom Workflows
================

This section highlights cases where Surfacia is used as a flexible workflow component rather than only through a single standard run.

Examples
--------

- descriptor generation without full modeling
- targeted element-specific or fragment-specific analysis
- combining Surfacia outputs with custom downstream scripts
- batch-oriented project workflows

Useful Real-World Patterns
--------------------------

Common custom usage patterns include:

- run the full workflow once, then repeat only modeling and interpretation
- compare Mode 1 and Mode 2 on the same scaffold family
- generate descriptor tables for use in external notebooks or scripts
- use resume-heavy workflows for long quantum-chemistry projects

Pattern 1: Reuse Descriptors
----------------------------

One practical pattern is to run the expensive stages once, then reuse the descriptor table for repeated modeling:

.. code-block:: bash

   surfacia workflow -i molecules.csv --resume
   surfacia ml-analysis -i descriptors.csv --target-property "TargetA"
   surfacia ml-analysis -i descriptors.csv --target-property "TargetB"

Pattern 2: Compare Modes on the Same Problem
--------------------------------------------

If the chemistry supports more than one interpretation strategy, compare them deliberately:

1. run a broad baseline
2. run an element-aware or fragment-aware variant
3. compare the retained compact descriptors
4. keep the version that tells the clearest chemical story

Pattern 3: Interpretation-First Projects
----------------------------------------

Some projects care more about readable design insight than about maximizing one metric. In that case:

- favor compact models
- inspect retained descriptor families carefully
- use SHAP to test whether the explanation is actually useful to a chemist

Pattern 4: Long-Running Resume Workflows
----------------------------------------

For expensive quantum-chemistry projects, a practical pattern is:

1. run the main workflow with resume enabled
2. stop and inspect intermediate outputs before committing to downstream modeling
3. recover failures only where needed
4. reuse the resulting descriptor tables for repeated interpretation passes

Why This Matters
----------------

Many research projects do not stay on one perfectly linear path. Custom workflows make Surfacia more practical when:

- calculations are interrupted
- descriptor strategy changes mid-project
- different collaborators handle QM, modeling, and interpretation separately

Recommended Mindset
-------------------

Think of Surfacia as a structured pipeline with reusable outputs, not only as a one-shot command.

Decision Checklist
------------------

Before branching into a custom workflow, ask:

- which stages are expensive enough that I should avoid repeating them?
- which outputs are reusable across multiple targets or hypotheses?
- am I optimizing for prediction, interpretation, or both?
- which representation choice would make the final result easiest to discuss with collaborators?
