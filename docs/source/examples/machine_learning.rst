Machine Learning Examples
=========================

These examples focus on how Surfacia descriptors behave once they enter a predictive modeling workflow.

Typical Questions
-----------------

- which descriptors survive compact feature selection?
- which signals are global versus local?
- how stable is the interpretation across splits?
- what do SHAP trends suggest about design?

Example Patterns Worth Studying
-------------------------------

This category is especially useful for comparing:

- small-data mechanism-aware studies
- scaffold-conserved fragment-centered studies
- broad property-prediction datasets

Suggested Minimal Pipeline
--------------------------

.. code-block:: bash

   surfacia workflow -i molecules.csv --resume --test-samples "1,2,3"
   surfacia ml-analysis -i descriptors.csv --target-property "YourTarget" --test-samples "1,2,3"
   surfacia shap-viz -i training_data.csv --api-key YOUR_API_KEY

This sequence is often enough to judge whether the descriptor representation is worth pushing further.

What to Compare
---------------

When reviewing a machine-learning example, compare at least four things:

1. raw descriptor count
2. retained compact feature set
3. predictive metrics
4. interpretability of the final descriptor story

Practical Reading Strategy
--------------------------

Do not start from the final metric alone.

Instead, ask:

- which features survived?
- do they cover more than one scale?
- do they match the chemistry of the problem?
- would you trust the explanation enough to guide a design decision?

Useful Comparison Table
-----------------------

When comparing machine-learning examples, keep notes on:

- descriptor count before selection
- descriptor count after selection
- whether retained features span multiple scales
- whether the SHAP interpretation is chemically readable
- whether the final explanation is stronger than a simple black-box score

Result Review Template
----------------------

After each modeling run, try recording:

.. csv-table::
   :header: "Question", "Why It Matters"
   :widths: 45, 55

   "How many features survived?", "Compact models are easier to explain and compare"
   "Do retained features span more than one scale?", "Mixed-scale models often reveal richer chemistry"
   "Are top descriptors physically meaningful?", "Interpretability depends on descriptor quality, not only model quality"
   "Does SHAP reveal thresholds or nonlinear behavior?", "Nonlinear structure-property patterns are often chemically informative"
   "Would this result support a design decision?", "This is the practical test of whether the model is useful"

What to Do When the Score Is Good but the Story Is Weak
-------------------------------------------------------

If the metric looks strong but the explanation is poor:

- check whether too many features survived
- compare a more compact model
- reconsider whether the descriptor mode matches the chemistry
- prioritize readability over a small numerical gain when the project goal is insight
