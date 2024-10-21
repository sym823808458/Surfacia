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

Available tasks: smi2xyz, reorder, extract_substructure, xyz2gaussian, run_gaussian, readmultiwfn, machinelearning, fchk2matches

## Task-specific Usage

1. Convert SMILES to XYZ  
python scripts/surfacia_main.py smi2xyz --smiles_csv <path_to_smiles_csv>

2. Reorder Atoms  
python scripts/surfacia_main.py reorder --element <element_symbol> --input_dir <input_directory> --output_dir <output_directory>

3. Extract Substructure  
python scripts/surfacia_main.py extract_substructure --substructure_file <path_to_substructure_file> --input_dir <input_directory> --output_dir <output_directory> --threshold <matching_threshold>

4. Convert XYZ to Gaussian Input  
python scripts/surfacia_main.py xyz2gaussian --xyz_folder <xyz_directory> --template_file <path_to_template_file> --output_dir <output_directory>

5. Run Gaussian Calculations  
python scripts/surfacia_main.py run_gaussian --com_dir <com_file_directory>

6. Process Multiwfn Output  
python scripts/surfacia_main.py readmultiwfn --input_dir <input_directory> --output_dir <output_directory> --smiles_target_csv <path_to_smiles_target_csv> --first_matches_csv <path_to_first_matches_csv> --descriptor_option <option_number>

7. Run Machine Learning Analysis  
python scripts/surfacia_main.py machinelearning --input_x <path_to_feature_matrix> --input_y <path_to_labels> --input_title <path_to_feature_names> --ml_input_dir <ml_input_directory> [additional_options]

8. FCHK to Matches  
python scripts/surfacia_main.py fchk2matches --input_path <fchk_directory> --xyz1_path <path_to_substructure_file> --threshold <matching_threshold>

## Additional Options
- Use --config <path_to_config_file> to specify a custom configuration file
- For detailed help on each task, use: python scripts/surfacia_main.py <task> --help

---
### ** Descriptor List Extracted from the Program**

#### General Molecular Properties
- **Sample Name**: Sample name
- **Atom Number**: Number of atoms
- **Molecule Weight**: Molecular weight (Da)
- **Occupied Orbitals**: Number of occupied orbitals
- **Isosurface area**: Isosurface area (Å²)
- **Sphericity**: Sphericity
- **Volume (Angstrom³)**: Molecular volume (Å³)
- **Density (g/cm³)**: Density (g/cm³)
- **Surface eff atom num**: Number of surface effective atoms

#### Electronic Properties
- **HOMO**: Highest Occupied Molecular Orbital energy (a.u.)
- **LUMO**: Lowest Unoccupied Molecular Orbital energy (a.u.)
- **HOMO-LUMO Gap**: HOMO-LUMO energy gap (a.u.)
- **Dipole Moment (a.u.)**: Dipole moment (a.u.)
- **Quadrupole Moment**: Quadrupole moment (a.u.)
- **Octopole Moment**: Octopole moment (a.u.)
- **ODI HOMO-1**: Orbital Delocalization Index of HOMO-1 orbital
- **ODI HOMO**: Orbital Delocalization Index of HOMO orbital
- **ODI LUMO**: Orbital Delocalization Index of LUMO orbital
- **ODI LUMO+1**: Orbital Delocalization Index of LUMO+1 orbital
- **ODI Mean**: Mean value of Orbital Delocalization Index
- **ODI Std**: Standard deviation of Orbital Delocalization Index

#### Molecular Size and Shape
- **Farthest Distance**: Distance between the two farthest atoms (Å)
- **Molecular Radius**: Molecular radius (Å)
- **Molecular Size Short**: Molecular size (short axis, Å)
- **Molecular Size Medium**: Molecular size (medium axis, Å)
- **Molecular Size Long**: Molecular size (long axis, Å)
- **Long/Sum Size Ratio**: Ratio of long axis length to the sum of all three axes lengths
- **Length/Diameter**: Ratio of molecular length to diameter
- **MPP**: Molecular Planarity Parameter (Å)
- **SDP**: Span of Deviation from Plane (Å)

#### Local Electronic Properties (LEAE, ESP, ALIE, etc.)
- **LEAE Minimal Value**: Minimum value of Local Electron Attachment Energy (eV)
- **LEAE Maximal Value**: Maximum value of Local Electron Attachment Energy (eV)
- **ESP Minimal Value**: Minimum value of Electrostatic Potential (kcal/mol)
- **ESP Maximal Value**: Maximum value of Electrostatic Potential (kcal/mol)
- **ESP Overall Average Value (kcal/mol)**: Overall average value of Electrostatic Potential (kcal/mol)
- **ESP Overall Variance ((kcal/mol)²)**: Overall variance of Electrostatic Potential ((kcal/mol)²)
- **Balance of Charges (nu)**: Balance of charges (ν)
- **Product of sigma²_tot and nu ((kcal/mol)²)**: Product of σ²_tot and ν ((kcal/mol)²)
- **Internal Charge Separation (Pi) (kcal/mol)**: Internal charge separation (Π, kcal/mol)
- **Molecular Polarity Index (MPI) (kcal/mol)**: Molecular Polarity Index (MPI, kcal/mol)
- **Polar Surface Area (Angstrom²)**: Polar surface area (Å²)
- **Polar Surface Area (%)**: Percentage of polar surface area (%)
- **ALIE Minimal Value**: Minimum value of Averaged Local Ionization Energy (eV)
- **ALIE Maximal Value**: Maximum value of Averaged Local Ionization Energy (eV)
- **ALIE Average Value**: Average value of Averaged Local Ionization Energy (eV)
- **ALIE Variance**: Variance of Averaged Local Ionization Energy (eV²)

#### Atomic Properties
In the program, there is a matrix `matrix['Matrix Data']`, which contains the following atomic-level descriptors (one value per atom):

- **Atom#**: Atom number
- **LEAE All area**: Total area of LEAE
- **LEAE Positive area**: Positive area of LEAE
- **LEAE Negative area**: Negative area of LEAE
- **LEAE Minimal value**: Minimum value of LEAE
- **LEAE Maximal value**: Maximum value of LEAE
- **LEAE All average**: Overall average of LEAE
- **LEAE Positive average**: Average of positive LEAE
- **LEAE Negative average**: Average of negative LEAE
- **LEAE All variance**: Overall variance of LEAE
- **LEAE Positive variance**: Variance of positive LEAE
- **LEAE Negative variance**: Variance of negative LEAE
- **ESP All area (Å²)**: Total area of ESP (Å²)
- **ESP Positive area (Å²)**: Positive area of ESP (Å²)
- **ESP Negative area (Å²)**: Negative area of ESP (Å²)
- **ESP Minimal value (kcal/mol)**: Minimum value of ESP (kcal/mol)
- **ESP Maximal value (kcal/mol)**: Maximum value of ESP (kcal/mol)
- **ESP All average (kcal/mol)**: Overall average of ESP (kcal/mol)
- **ESP Positive average (kcal/mol)**: Average of positive ESP (kcal/mol)
- **ESP Negative average (kcal/mol)**: Average of negative ESP (kcal/mol)
- **ESP All variance ((kcal/mol)²)**: Overall variance of ESP ((kcal/mol)²)
- **ESP Positive variance ((kcal/mol)²)**: Variance of positive ESP ((kcal/mol)²)
- **ESP Negative variance ((kcal/mol)²)**: Variance of negative ESP ((kcal/mol)²)
- **ESP Pi (kcal/mol)**: Internal charge separation of ESP (Π, kcal/mol)
- **ESP nu**: Charge balance of ESP (ν)
- **ESP nu*sigma²**: Product of σ²_tot and ν of ESP
- **ALIE Area (Å²)**: Area of ALIE (Å²)
- **ALIE Min value**: Minimum value of ALIE (eV)
- **ALIE Max value**: Maximum value of ALIE (eV)
- **ALIE Average**: Average value of ALIE (eV)
- **ALIE Variance**: Variance of ALIE (eV²)

 When applied in practical scenarios, the surface chemical interactions of individual atoms can sometimes be extremely important. However, in many cases, certain physical quantities may result in `NaN`, as not every atom is exposed on the surface, or due to other computational reasons. Therefore, some descriptors in this section must be ignored. Here, we retain the following key features, with the prefix `Atom1` to indicate they pertain to atomic properties:

- **Atom1_LEAE_Minimal_value**
- **Atom1_LEAE_All_average**
- **Atom1_ESP_All_area_(Å²)**
- **Atom1_ESP_Minimal_value_(kcal/mol)**
- **Atom1_ESP_Maximal_value_(kcal/mol)**
- **Atom1_ESP_All_average_(kcal/mol)**
- **Atom1_ESP_Pi_(kcal/mol)**
- **Atom1_ALIE_Min_value**
- **Atom1_ALIE_Max_value**
  
---
## **Developer**
**Dr. Yuming Su** is the primary developer of **Surfacia: Surface Atomic Chemical Interaction Analyzer**. Dr. Su completed his Ph.D. in 2024 from the **College of Chemistry and Chemical Engineering** at **Xiamen University**.

---

## **Citation**
If you use **Surfacia** in your work, please cite the following:

Su, Y. **Surfacia: Surface Atomic Chemical Interaction Analyzer** (Version 0.0.1) [Software]. 2024. Available at: [GitHub Repository URL]
