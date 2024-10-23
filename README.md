## **Surfacia**
**SURF Atomic Chemical Interaction Analyzer - Surfacia**
![167273d2f11fef9ec1402476857c931](https://github.com/user-attachments/assets/e6c40f2a-4ca2-4ed6-a1f0-2d70767a5b55)

**Surfacia** (Surface Atomic Chemical Interaction Analyzer) is a comprehensive toolset designed to automate the workflow for analyzing surface atomic chemical interactions. It integrates molecular structure generation, quantum chemical computations, feature extraction, and machine learning analysis, providing a streamlined solution for **structure-activity relationships in chemistry research**, , with an emphasis on generating conclusions that are **interpretable by chemists**.

### **Features**
- **3D Structure Generation**: Converts SMILES strings to 3D molecular structures (XYZ format).
- **Integration with Computational Chemistry Software**: Supports Gaussian, XTB, and Multiwfn for quantum chemical computations.
- **Atom Reordering**: Reorders atoms in XYZ files for specific analyses.
- **Machine Learning**: Built-in scripts for feature extraction and regression analysis using XGBoost and stepwise regression.
- **SHAP Analysis**: Provides SHAP (SHapley Additive exPlanations) plots to interpret machine learning models.

### **Dependencies**
To use **Surfacia**, ensure the following dependencies are installed:

- **Python 3.9**
- **OpenBabel**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**
- **SHAP**

### **Required Computational Software**
Ensure that the following computational chemistry software is available on your system:

- **Gaussian**
- **XTB**
- **Multiwfn**

---

### **Workflow Overview**

### **Surfacia Command Line Usage Manual**

This manual provides a step-by-step guide on how to use Surfacia.

### **Prerequisites**
- **Anaconda/Miniconda:** Ensure you have Anaconda or Miniconda installed.
- **Surfacia Installation:** Clone or download Surfacia to your local machine.
- **Dependencies:** Install required dependencies listed in `requirements.txt`.

### Surfacia Command Line Usage Guide

## Prerequisites
- Ensure you have Surfacia installed and the required environment set up.
- Activate your Python environment (e.g., `conda activate sympy37`)

## General Usage
python scripts/surfacia_main.py <task> [options]

Available tasks: smi2xyz,xtbopt, reorder, extract_substructure, xyz2gaussian, run_gaussian, readmultiwfn, machinelearning, fchk2matches

## Task-specific Usage

1. Convert SMILES to XYZ  
python scripts/surfacia_main.py smi2xyz --smiles_csv <path_to_smiles_csv>

2.Run xtb Optimization  
python scripts/surfacia_main.py xtbopt --input_folder <xyz_directory> --output_folder <output_directory> [--param_file <optional_param_file>]

3. Reorder Atoms  
python scripts/surfacia_main.py reorder --element <element_symbol> --input_dir <input_directory> --output_dir <output_directory>

4. Extract Substructure  
python scripts/surfacia_main.py extract_substructure --substructure_file <path_to_substructure_file> --input_dir <input_directory> --output_dir <output_directory> --threshold <matching_threshold>

5. Convert XYZ to Gaussian Input  
python scripts/surfacia_main.py xyz2gaussian --xyz_folder <xyz_directory> --template_file <path_to_template_file> --output_dir <output_directory>

6. Run Gaussian Calculations  
python scripts/surfacia_main.py run_gaussian --com_dir <com_file_directory>

7. Process Multiwfn Output  
python scripts/surfacia_main.py readmultiwfn --input_dir <input_directory> --output_dir <output_directory> --smiles_target_csv <path_to_smiles_target_csv> --first_matches_csv <path_to_first_matches_csv> --descriptor_option <option_number>

8. Run Machine Learning Analysis  
python scripts/surfacia_main.py machinelearning --full_csv <path_to_full_csv> --output_dir <ml_output_directory> --nan_handling <nan_handling_option> --epoch <number_of_epochs> --core_num <cpu_cores> --train_test_split_ratio <split_ratio> --step_feat_num <number_of_features> [other_options]

9. FCHK to Matches  
python scripts/surfacia_main.py fchk2matches --input_path <fchk_directory> --xyz1_path <path_to_substructure_file> --threshold <matching_threshold>

## Additional Options
- Use --config <path_to_config_file> to specify a custom configuration file
- For detailed help on each task, use: python scripts/surfacia_main.py <task> --help

---
### ** Descriptor List Extracted from the Program**

This program extracts molecular descriptors based on the selected `descriptor_option`. There are three options:

1. **Molecular Properties Only** (`descriptor_option=1`)
2. **Molecular Properties + Specific Atomic Properties** (`descriptor_option=2`)
3. **Molecular Properties + Specific Atomic Properties + Fragment Properties** (`descriptor_option=3`)

### Descriptors Extracted

#### 1. Molecular Properties Only (`descriptor_option=1`)

Extracts only molecular-level descriptors.

**General Molecular Properties**
- **Sample Name**: Sample name
- **Atom Number**: Number of atoms
- **Molecule Weight**: Molecular weight (Da)
- **Occupied Orbitals**: Number of occupied orbitals
- **Isosurface Area**: Isosurface area (Å²)
- **Sphericity**: Sphericity
- **Volume (Å³)**: Molecular volume (Å³)
- **Density (g/cm³)**: Density (g/cm³)
- **Surface Exposed Atom Number**: Number of surface-exposed atoms

**Electronic Properties**
- **HOMO**: Highest Occupied Molecular Orbital energy (a.u.)
- **LUMO**: Lowest Unoccupied Molecular Orbital energy (a.u.)
- **HOMO-LUMO Gap**: HOMO-LUMO energy gap (a.u.)
- **Dipole Moment (a.u.)**: Dipole moment (a.u.)
- **Quadrupole Moment**: Quadrupole moment (a.u.)
- **Octopole Moment**: Octopole moment (a.u.)
- **ODI HOMO-1**: Orbital Delocalization Index of HOMO-1
- **ODI HOMO**: Orbital Delocalization Index of HOMO
- **ODI LUMO**: Orbital Delocalization Index of LUMO
- **ODI LUMO+1**: Orbital Delocalization Index of LUMO+1
- **ODI Mean**: Mean Orbital Delocalization Index
- **ODI Std**: Standard Deviation of Orbital Delocalization Index

**Molecular Size and Shape**
- **Farthest Distance**: Distance between the two farthest atoms (Å)
- **Molecular Radius**: Molecular radius (Å)
- **Molecular Size Short**: Short axis length (Å)
- **Molecular Size Medium**: Medium axis length (Å)
- **Molecular Size Long**: Long axis length (Å)
- **Long/Sum Size Ratio**: Ratio of long axis to sum of all axes
- **Length/Diameter**: Ratio of molecular length to diameter
- **MPP**: Molecular Planarity Parameter (Å)
- **SDP**: Span of Deviation from Plane (Å)

**Local Electronic Properties (LEAE, ESP, ALIE, etc.)**
- **LEAE Minimal Value**: Minimum Local Electron Attachment Energy (eV)
- **LEAE Maximal Value**: Maximum Local Electron Attachment Energy (eV)
- **ESP Minimal Value**: Minimum Electrostatic Potential (kcal/mol)
- **ESP Maximal Value**: Maximum Electrostatic Potential (kcal/mol)
- **ESP Overall Average Value (kcal/mol)**: Average Electrostatic Potential
- **ESP Overall Variance ((kcal/mol)²)**: Variance of Electrostatic Potential
- **Balance of Charges (ν)**: Charge balance
- **Product of σ²_tot and ν ((kcal/mol)²)**: Product of σ²_tot and ν
- **Internal Charge Separation (Π) (kcal/mol)**: Internal charge separation
- **Molecular Polarity Index (MPI) (kcal/mol)**: Molecular Polarity Index
- **Polar Surface Area (Å²)**: Polar surface area
- **Polar Surface Area (%)**: Percentage of polar surface area
- **ALIE Minimal Value**: Minimum Averaged Local Ionization Energy (eV)
- **ALIE Maximal Value**: Maximum Averaged Local Ionization Energy (eV)
- **ALIE Average Value**: Average Averaged Local Ionization Energy (eV)
- **ALIE Variance**: Variance of Averaged Local Ionization Energy (eV²)

#### 2. Molecular Properties + Specific Atomic Properties (`descriptor_option=2`)

Includes all descriptors from Option 1 plus specific atomic descriptors for selected atoms (e.g., Atom1).

**Specific Atomic Properties**
- **Atom1_LEAE_Minimal_value**: Minimum LEAE for Atom 1
- **Atom1_LEAE_All_average**: Average LEAE for Atom 1
- **Atom1_ESP_All_area_(Å²)**: Total ESP area for Atom 1 (Å²)
- **Atom1_ESP_Minimal_value_(kcal/mol)**: Minimum ESP for Atom 1 (kcal/mol)
- **Atom1_ESP_Maximal_value_(kcal/mol)**: Maximum ESP for Atom 1 (kcal/mol)
- **Atom1_ESP_All_average_(kcal/mol)**: Average ESP for Atom 1 (kcal/mol)
- **Atom1_ESP_Pi_(kcal/mol)**: Internal charge separation of ESP for Atom 1 (Π, kcal/mol)
- **Atom1_ALIE_Min_value**: Minimum ALIE for Atom 1 (eV)
- **Atom1_ALIE_Max_value**: Maximum ALIE for Atom 1 (eV)

**Note**: Only key atomic descriptors are retained to account for possible `NaN` values due to atoms not being surface-exposed.

#### 3. Molecular Properties + Specific Atomic Properties + Fragment Properties (`descriptor_option=3`)

Includes all descriptors from Option 2 plus fragment-level descriptors based on fragment indices from `first_matches_csv`.

**Fragment-Level Properties**
- **Frag_LEAE_Minimal_Value**: Minimum LEAE for the fragment (eV)
- **Frag_LEAE_Maximal_Value**: Maximum LEAE for the fragment (eV)
- **Frag_LEAE_Average_Value**: Average LEAE for the fragment (eV)
- **Frag_ESP_Minimal_Value**: Minimum ESP for the fragment (kcal/mol)
- **Frag_ESP_Maximal_Value**: Maximum ESP for the fragment (kcal/mol)
- **Frag_ESP_Overall_Surface_Area_(Å²)**: Overall ESP surface area for the fragment (Å²)
- **Frag_ESP_Average_Value**: Average ESP for the fragment (kcal/mol)
- **Frag_ESP_variance_Value**: Variance of ESP for the fragment ((kcal/mol)²)
- **Frag_ESP_Pi_Value**: Internal charge separation of ESP for the fragment (Π, kcal/mol)
- **Frag_ALIE_Minimal_Value**: Minimum ALIE for the fragment (eV)
- **Frag_ALIE_Maximal_Value**: Maximum ALIE for the fragment (eV)
- **Frag_ALIE_Average_Value**: Average ALIE for the fragment (eV)
- **Frag_ALIE_Variance_Value**: Variance of ALIE for the fragment (eV²)

**Note**: Fragment descriptors also include atomic features of the atoms forming the fragment, as specified by the `sub.xyz1` files.

**Key Atomic Descriptors Retained (`AtomN_` Prefix)**
- **AtomN_LEAE_Minimal_value**
- **AtomN_LEAE_All_average**
- **AtomN_ESP_All_area_(Å²)**
- **AtomN_ESP_Minimal_value_(kcal/mol)**
- **AtomN_ESP_Maximal_value_(kcal/mol)**
- **AtomN_ESP_All_average_(kcal/mol)**
- **AtomN_ESP_Pi_(kcal/mol)**
- **AtomN_ALIE_Min_value**
- **AtomN_ALIE_Max_value**

**Note**: These key descriptors are essential for evaluating the local electronic properties and chemical activity of specific atoms. The number of atoms (`N`) is determined based on fragments specified by `sub.xyz1` files.

---
## **Developer**
**Dr. Yuming Su** is the primary developer of **Surfacia: Surface Atomic Chemical Interaction Analyzer**. Dr. Su completed his Ph.D. in 2024 from the **College of Chemistry and Chemical Engineering** at **Xiamen University**.

---

## **Citation**
If you use **Surfacia** in your work, please cite the following:

Su, Y. **Surfacia: Surface Atomic Chemical Interaction Analyzer** (Version 0.0.1) [Software]. 2024. Available at: [GitHub Repository URL]
