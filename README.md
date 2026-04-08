<div align="center">
  <img src="https://raw.githubusercontent.com/sym823808458/Surfacia/main/Surfacia.png" alt="Surfacia Logo" width="200"/>
</div>

# Surfacia: From Molecules to Insights, Automatically

<p align="center">
  <strong>An Automated Framework for Surface-Based Feature Engineering and Interpretable Machine Learning</strong>
</p>

<p align="center">
  <a href="https://github.com/sym823808458/Surfacia/releases"><img alt="Version" src="https://img.shields.io/badge/version-3.0.2-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9+-blue.svg"></a>
  <a href="https://github.com/sym823808458/Surfacia/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="#"><img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg"></a>
</p>

---

**Surfacia** is a next-generation computational chemistry platform that bridges the gap between experimental synthesis and data-driven discovery. It offers a complete, end-to-end pipeline that transforms a simple list of molecules (SMILES strings) into deep, actionable chemical insights. Our framework is designed to answer not just **"what"** works, but **"why"** it works, by leveraging interpretable, physics-based descriptors.

Our core philosophy is to empower **experimental and computational chemists alike** with the predictive power of quantum mechanics and AI, packaged in a highly automated and user-friendly workflow.

## Key Features

*   🚀 **One-Command Workflow**: Go from a list of SMILES to a fully trained, interpretable AI model with a single command: `surfacia workflow`. It automates the entire multi-step process, from 3D structure generation to advanced machine learning analysis.

*   🧠 **Intelligent Job Resumption**: Did your long-running calculations crash? No problem. The `--resume` flag enables Surfacia to intelligently detect completed steps, clean up failed jobs, and restart only from where it left off, saving you invaluable time and computational resources.

*   🔬 **Chemically Intuitive Descriptors**: Forget abstract mathematical fingerprints. Surfacia generates features based on fundamental physicochemical properties derived from the molecular surface (e.g., electrostatic potential, local ionization energy, electron affinity). These descriptors are directly interpretable and align with established chemical principles.

*   📊 **Interactive AI-Powered Insights**: Our models are not black boxes. The `shap-viz` command launches an interactive web dashboard that provides a deep dive into model behavior. Visually identify which atoms or functional groups drive molecular activity, explore structure-activity relationships, and receive automated explanations from an integrated AI assistant.

*   ⚙️ **Modular & Flexible**: While the automated workflow is powerful, Surfacia remains fully modular. Each step of the pipeline can be run independently, giving advanced users complete control over their computational experiments.

## Installation

### Prerequisites

1.  **Python**: Version 3.9 or newer is recommended.
2.  **External Software**: Surfacia orchestrates several state-of-the-art computational chemistry programs. Please ensure they are installed, licensed (where applicable), and accessible from your system's `PATH`.
    *   [**XTB**](https://xtb-docs.readthedocs.io/en/latest/setup.html): For rapid, semi-empirical geometry optimizations.
    *   [**Gaussian**](http://gaussian.com/): For high-accuracy quantum mechanical calculations.
    *   [**Multiwfn**](http://sobereva.com/multiwfn/): For comprehensive wavefunction analysis.

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/sym823808458/Surfacia.git
cd Surfacia

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
conda create -n surfacia python=3.9
conda activate surfacia
conda install -c conda-forge rdkit # On Linux/macOS
# venv\Scripts\activate    # On Windows

# 3. Install all required Python dependencies
pip install -r requirements.txt

# 4. Install the Surfacia package in editable mode
pip install -e .

or pip install surfacia
```

## Quick Start: Your First Analysis in 5 Minutes

Let's experience the power of Surfacia with a minimal example. Details see https://surfacia.readthedocs.io/zh-cn/latest/#

**Step 1: Prepare Your Input Data**

Create a CSV file named `molecules.csv`. It requires only two columns: `smiles` for the molecular structures and `target` for your measured experimental property (e.g., IC50, yield, etc.).

```csv
smiles,target
c1ccccc1,5.2
CC(=O)Oc1ccccc1C(=O)O,7.8
CN1C=NC2=C1C(=O)N(C)C(=O)N2C,9.5
C1CCCCC1,4.1
O=C(C)C,6.3
```

**Step 2: Launch the Automated Workflow**

Open your terminal in the same directory as `molecules.csv` and run the following command. We'll designate molecules at index 1 and 3 as our hold-out test set.

```bash
surfacia workflow -i molecules.csv --test-samples "1,3"
```

**Step 3: Sit Back and Relax**

Surfacia is now executing the entire computational pipeline for you:
1.  ✅ Converting SMILES to 3D structures.
2.  ✅ Optimizing geometries with XTB.
3.  ✅ Generating and submitting Gaussian calculations (this is the most time-consuming step).
4.  ✅ Analyzing wavefunctions with Multiwfn to compute surface properties.
5.  ✅ Extracting atomic and functional-group level descriptors.
6.  ✅ Performing automated feature selection and machine learning model training.
7.  ✅ **Launching an interactive web server for SHAP visualization!**

Once the workflow is complete, you will see a message like this:

```
🎉 Complete Surfacia Workflow Finished Successfully!
...
💡 Next steps:
   • Check the output folder for detailed results.
   • SHAP visualization should be running on http://localhost:8052
```

**Step 4: Explore Your Results**

Open your web browser and navigate to `http://localhost:8052`. You can now interactively explore how each chemical feature influences your target property, linking quantitative data directly back to molecular structure.

## Comprehensive Usage Guide

### 1. The All-in-One `workflow` Command

This is the recommended entry point for most users. It provides a powerful combination of automation and customization.

```bash
# Run the full pipeline with a specified test set
surfacia workflow -i molecules.csv --test-samples "1,5,10"

# Resume a previously interrupted job, automatically skipping completed steps
surfacia workflow -i molecules.csv --test-samples "1,5,10" --resume

# Customize the machine learning stage
surfacia workflow -i molecules.csv --max-features 8 --stepreg-runs 5
```
For a full list of options, run `surfacia workflow --help`.

### 2. Modular, Step-by-Step Execution

For advanced users who require granular control, each component of the workflow can be run as a separate command.

```bash
# Step 1: Convert SMILES to 3D coordinates
surfacia smi2xyz -i molecules.csv

# Step 2: Optimize geometries with XTB
surfacia xtb-opt --method gfn2 --opt-level tight

# Step 3: Generate Gaussian input files
surfacia xyz2gaussian --keywords "# PBE1PBE/6-311g* scrf(SMD,solvent=Water)"

# Step 4: Run Gaussian calculations
surfacia run-gaussian

# Step 5: Perform Multiwfn analysis on results
surfacia multiwfn

# Step 6: Extract features from Multiwfn output
# The input file is located in the newly created Surfacia_* directory
surfacia extract-features -i ./Surfacia*/FullOption*.csv --mode 3

# Step 7: Run machine learning analysis
surfacia ml-analysis -i ./Surfacia*/FinalFull*.csv --test-samples "1,5,10"

# Step 8: Launch the interactive visualization dashboard
surfacia shap-viz -i ./Surfacia*/Auto*/Training_Set_Detailed.csv -x .
```

### 3. Standalone Utility Tools

Surfacia also includes several convenient utilities for everyday chemical tasks.

```bash
# Generate high-quality 2D images of molecules from a CSV
surfacia mol-draw -i molecules.csv -o ./molecule_images

# Calculate common molecular properties (LogP, TPSA, etc.)
surfacia mol-info -i molecules.csv -o properties_report.csv

# Manually find and rerun only the failed jobs from a Gaussian batch
surfacia rerun-gaussian
```

To see all available commands, run `surfacia --help`. For detailed information on a specific command, use `surfacia <command> --help`.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Yuming Su** ([`823808458@qq.com`](mailto:823808458@qq.com))

Dr. Yuming Su is the primary developer of Surfacia. He completed his Ph.D. in 2024 from the College of Chemistry and Chemical Engineering at Xiamen University.

## How to Cite

If you use Surfacia in your research, please cite our work.
> Su, Y. *et al.* Surfacia: An Automated Framework for Surface-Based Feature Engineering and Interpretable Machine Learning. *Journal Name*, **Year**, *Volume*, Pages. (Please add your paper's citation here once published).
>
> Su, Y. Surfacia: Surface Atomic Chemical Interaction Analyzer, Version 3.0.1, https://github.com/sym823808458/Surfacia (accessed Month Year).
```
