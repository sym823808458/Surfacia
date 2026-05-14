Core API
========

The core layer covers the main workflow logic that connects input structures, quantum calculations, descriptor generation, and downstream analysis.

What Belongs Here
-----------------

- workflow orchestration
- input and output management
- configuration handling
- run-state tracking and resume logic

Typical Responsibilities
------------------------

The core layer is where Surfacia behaves like a pipeline rather than a collection of isolated scripts.

- read structured molecular input
- keep file relationships traceable across steps
- prepare working directories
- coordinate external software calls
- pass outputs cleanly to descriptor and modeling stages

When to Use the Core API
------------------------

Use the core API when you want to:

- automate larger batches programmatically
- wrap Surfacia in your own Python workflow
- customize how pipeline stages are chained together
- add new high-level workflow behaviors
