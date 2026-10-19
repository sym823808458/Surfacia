# Surfacia

Molecular surface descriptors, predictive modeling and chemical interpretation.

Surfacia connects **SMILES → 3D → xTB/Gaussian → Multiwfn → descriptors → ML/SHAP**.
Choose whole-molecule, molecular-surface or local regions to relate a model's
predictions to chemical structure. It analyzes molecular surfaces; it does not
by itself simulate an electrode or a solid catalyst interface.

## Start with your data

- **Already have descriptors:** run the [downloadable 36-molecule tutorial](https://surfacia.readthedocs.io/zh-cn/latest/examples/descriptor_table.html).
- **Start from molecular structures:** follow the [surface workflow](https://surfacia.readthedocs.io/zh-cn/latest/user_guide/workflows.html).
- **Inspect model explanations:** open original predictions and ordinary SHAP in the [interactive case](https://surfacia.readthedocs.io/zh-cn/latest/examples/interpretation.html).
- **中文手册:** [切换到完整中文内容](https://surfacia.readthedocs.io/zh-cn/latest/zh/index.html).

## Install and check your version

```bash
python -m pip install --upgrade "surfacia[viz]"
python -c "import surfacia; print(surfacia.__version__)"
surfacia --help
```

The current version is **4.0.0**. Upgrade an existing 3.0.3 installation to remove
the former candidate-ranking extension. Check the printed version after installing.
For development from GitHub source:

```bash
git clone https://github.com/sym823808458/Surfacia.git
cd Surfacia
python -m pip install -e ".[viz]"
```

Use a separate environment, preferably Python 3.10–3.12. The package declares
Python >=3.9. The ML compatibility ranges remain `xgboost>=2.1.4,<3` and
`shap>=0.48,<0.49`. The full quantum workflow additionally needs independently
installed/licensed xTB, Gaussian (`g16`, `formchk`) and Multiwfn. The descriptor-table
tutorial does not need these external tools.

## A runnable example

From the repository, after installing dependencies:

```bash
python examples/descriptor_table/make_data.py
surfacia ml-analysis -i examples/descriptor_table/descriptors.csv --manual --manual-features Full --test-samples "6,12,18,24,30,36" --epoch 32 --cores 1 --train-test-split 0.85 --no-generate-fitting
```

The six features and target are computed with RDKit. This is a reproducible
software tutorial, **not a quantum-surface benchmark or experimental logP validation**.
The current CLI uses XGBoost regression and repeated random splits. The manual
documents its evaluation boundaries and distinguishes it from a frozen research
benchmark. Ordinary SHAP is retained; attribution alone does not prove causality.

## Upgrade from 3.0.3

The former Go Beyond/SPES candidate-ranking extension is removed from current
source: no automatic ranking CSVs, ranking dashboard layer, `--spes-csv` option,
or `surfacia.ml.spes` API. Original regression predictions and SHAP interpretation
remain. See [migration notes](https://surfacia.readthedocs.io/zh-cn/latest/migration.html).
The major version marks removed interfaces, not a new quantum method. Historical
releases and manuscript archives remain available through their original records.

## Maintain and cite

- [Maintenance](https://github.com/sym823808458/Surfacia/blob/main/MAINTENANCE.md): update docs/cases and build/release checks.
- [Release audit](https://github.com/sym823808458/Surfacia/blob/main/RELEASE_AUDIT_20261019.md): GitHub, PyPI and Read the Docs findings.
- [Issues](https://github.com/sym823808458/Surfacia/issues)
- Author: Yuming Su / 苏禹铭 · 823808458@qq.com
- [Research preprint and citation](https://surfacia.readthedocs.io/zh-cn/latest/citation.html): ChemRxiv v1, posted 2026-08-27, DOI [10.26434/chemrxiv.15007949/v1](https://doi.org/10.26434/chemrxiv.15007949/v1). This archived paper predates the 4.0.0 scope revision; cite the exact software version and commit as well.
- License: MIT.
