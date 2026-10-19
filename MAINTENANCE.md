# Maintenance

## Scope

The maintained public version covers molecular surface descriptors, predictive
modeling and ordinary SHAP interpretation. Go Beyond, SPES, SHAP-cliff and
candidate acquisition are not current functionality. Preserve historical Git
commits, release wheels and manuscript archives rather than rewriting them.

The active public-source checkout prepared on 2026-10-19 is this repository.
Other local copies can contain unrelated work; do not overwrite them.

## Content and cases

- Edit paired RST files in `docs/source/` and `docs/source/zh/`.
- Command options come from `surfacia.cli.build_parser`; regenerate them with
  `python tools/update_cli_reference.py`. Do not invent additional CLI options.
- `examples/descriptor_table/make_data.py` regenerates 36 public SMILES and RDKit
  properties. The target is computed Crippen logP, not an experimental label.
- Run `python tools/regenerate_demo.py` after installing `surfacia[viz]`. It runs
  the actual command and copies selected CSVs, plots, versions and hashes to docs.
  `--reuse-run` only rebuilds assets from an existing completed run.
- Review the recorded validation and held-out metrics together. Do not remove
  an unfavorable result or advertise this as a Surfacia quantum benchmark.
- The sulfur case contains structures only. Quantum executables and experimental
  catalytic targets are not bundled. Never fabricate surface/FE/yield results.
- Legacy URLs are retained as short pointers to the maintained pages.
- Documentation builds install only `docs/requirements.txt`, without scientific
  dependencies or quantum calculations.

## Checks

```bash
python tools/update_cli_reference.py
python -m unittest discover -s tests
python -m sphinx -W --keep-going -b html docs/source docs/build/html
python tools/check_docs.py
python -m build
python -m twine check dist/*
```

Use a clean environment for package installation and the runnable case. Inspect
desktop/mobile rendering and the language switch. A successful documentation
build is not evidence that Gaussian/xTB/Multiwfn ran.

## GitHub, Read the Docs and PyPI

1. Refresh `main`; retain unrelated changes and commit source/docs together.
2. Push and verify the Read the Docs build matches the commit.
3. Build versioned wheel/sdist and verify their contents and metadata. Versions
   are read from `pyproject.toml`; keep `surfacia.__version__` aligned.
4. PyPI uploads are immutable. Never claim a source version is available through
   pip until the PyPI JSON endpoint confirms it.
5. Preferred publishing is the `publish.yml` GitHub Actions workflow with PyPI
   Trusted Publishing. Configure the existing PyPI project publisher with owner
   `sym823808458`, repository `Surfacia`, workflow `publish.yml`, environment
   `pypi`. No API token belongs in Git or chat.
6. Trigger the publish workflow for the reviewed commit when authorized and
   configured. It builds and checks distributions, then uploads using OIDC.
   A missing publisher must be configured by the PyPI account owner.

Changes to actual scientific benchmark protocols need separate review. The
public CLI is not automatically the frozen manuscript benchmarking code.
