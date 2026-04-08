# Surfacia

Surface Atomic Chemical Interaction Analyzer for descriptor extraction and interpretable machine learning in computational chemistry.

## At a Glance

- End-to-end workflow: `SMILES -> 3D -> xTB/Gaussian -> Multiwfn -> descriptors -> ML/SHAP`
- CLI-first design for local and remote Linux/HPC usage
- Supports full workflow mode and modular step-by-step execution

## Current Release

- Latest stable version: **3.0.2**
- Python requirement: **>= 3.9**
- License: **MIT**

PyPI: [https://pypi.org/project/surfacia/](https://pypi.org/project/surfacia/)

## Quick Install (Recommended for Most Users)

```bash
# optional but recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install surfacia
```

Check installation:

```bash
surfacia --help
python -c "import surfacia; print(surfacia.__version__)"
```

## Important Compatibility Note (ML Stage)

To avoid ML parsing failures such as:

`could not convert string to float: '[-3.xxxE0]'`

use the validated dependency range:

- `xgboost>=2.1.4,<3.0.0`
- `shap>=0.48.0,<0.49.0`

Quick fix:

```bash
pip install --force-reinstall "xgboost==2.1.4" "shap==0.48.0"
```

## External Software Requirements

Surfacia orchestrates external quantum-chemistry tools. Ensure these commands are available in your `PATH`:

- `xtb`
- `g16` (and `formchk`)
- `Multiwfn` or `Multiwfn_noGUI`

## Minimal Usage

Run full workflow:

```bash
surfacia workflow -i molecules.csv --test-samples "1,3"
```

Run ML only (for existing descriptor table):

```bash
surfacia ml-analysis -i FinalFull_Mode3_20_168.csv \
  --max-features 1 --stepreg-runs 1 \
  --train-test-split 0.85 --epoch 32 --cores 8 \
  --test-samples "1,2,3"
```

## MCP Server (New)

Surfacia now ships an MCP stdio server so agent clients can call workflow stages as structured tools.

Start MCP server:

```bash
surfacia-mcp --log-level INFO
# or
python -m surfacia.mcp.server --log-level INFO
```

Typical first tool calls:

1. `surfacia_check_environment`
2. `surfacia_detect_workflow_state`

## Source Install (Developers)

```bash
git clone https://github.com/sym823808458/Surfacia.git
cd Surfacia
pip install -e .
```

## Documentation

- Full docs: [https://surfacia.readthedocs.io/](https://surfacia.readthedocs.io/)
- Troubleshooting: [https://surfacia.readthedocs.io/en/latest/user_guide/troubleshooting.html](https://surfacia.readthedocs.io/en/latest/user_guide/troubleshooting.html)
- Mode3 top-20 remote debug example: [https://surfacia.readthedocs.io/en/latest/examples/mode3_top20_remote_debug.html](https://surfacia.readthedocs.io/en/latest/examples/mode3_top20_remote_debug.html)
- MCP server guide: [https://surfacia.readthedocs.io/en/latest/integrations/mcp_server.html](https://surfacia.readthedocs.io/en/latest/integrations/mcp_server.html)

## Citation

If Surfacia helps your research, please cite the project and related publication when available.

## Contact

- Author: Yuming Su
- Email: 823808458@qq.com
- Issues: [https://github.com/sym823808458/Surfacia/issues](https://github.com/sym823808458/Surfacia/issues)
