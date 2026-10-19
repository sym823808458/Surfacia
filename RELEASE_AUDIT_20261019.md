# Public release audit — 2026-10-19

Live checks before editing found:

| Surface | Observed state |
| --- | --- |
| GitHub main | `6f0e2ba689090ac5afad74f73e8a16dbd753bcf8`; README and runtime advertised/integrated SPES |
| PyPI | Latest 3.0.3, uploaded 2026-05-19; wheel SHA-256 `1e623ff2359f4d013ae2fe9b86dd2266a45c656e02a967cfbc29315962ba52a2` |
| Downloaded PyPI wheel | Candidate-ranking code in CLI, workflow, MCP launcher, ML, `ml/spes.py`, visualizer |
| Read the Docs | Project `surfacia`, language URL `zh-cn`, default branch main, successful build 32756096 at the same GitHub commit |
| Manual mismatches | ML page described unsupported classification/algorithm/target switches and k-fold CV |

The author's requested scope is molecular-surface descriptors, prediction and
interpretability. New source 4.0.0 removes the published ranking interfaces and
automatic outputs; the major version signals the interface removal. It does not
claim new quantum algorithms or replace existing manuscript protocols.

The paired manual separates install/version checks, whole workflow, descriptor
tables, outputs, actual CLI reference, boundaries and three worked learning paths.
The runnable dataset uses 36 public SMILES, RDKit features and a computed logP
target, with all source inputs and output provenance downloadable. No unpublished
experiment/manuscript files are included. The structure-only sulfur case has no
fabricated target or claimed quantum rerun.

## Version availability

This audit distinguishes source and uploaded distributions. PyPI 3.0.3 cannot be
edited in place. Version 4.0.0 is a new distribution, with build/install checks
and owner-authorized publication. `MAINTENANCE.md` documents the prepared
Trusted Publisher workflow. Verify current availability using the live PyPI endpoint.

Upgrade with `python -m pip install --upgrade "surfacia[viz]==4.0.0"`. Package mirrors may lag behind the official PyPI index. Older release artifacts remain unchanged.

## Publication verification

On 2026-10-19, the version-specific PyPI JSON endpoint confirmed both the 4.0.0
wheel and source distribution. Both SHA-256 hashes matched the checked local
artifacts. A fresh download and forced installation from the official PyPI
index succeeded; the imported module reported 4.0.0 from site-packages.

The installed package ran the 36-molecule descriptor-table case outside the
source directory, using 32 training rounds and six held-out samples. Validation
R-squared was 0.7662 and held-out R-squared was 0.0761, matching the documented
teaching case. Original training/test prediction files were produced; no SPES
ranking files were produced. This is not a quantum-surface benchmark.
