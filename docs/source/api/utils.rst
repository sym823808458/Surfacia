Utilities API
=============

The utilities layer contains helper functions that support file handling, data preparation, and interaction with external tools.

What Belongs Here
-----------------

- file and path helpers
- data-cleaning utilities
- reusable parsing logic
- helper functions shared across pipeline stages

Why It Matters
--------------

Utility code is often where workflow robustness comes from.

- it keeps data handling consistent
- it reduces duplicated logic
- it makes larger workflows easier to maintain

When to Use the Utilities API
-----------------------------

Use this layer when you want to:

- reuse shared logic in your own scripts
- understand how Surfacia prepares intermediate files
- extend the codebase without duplicating helper behavior
