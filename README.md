## **Surfacia**
**SURF Atomic Chemical Interaction Analyzer - Surfacia**
![167273d2f11fef9ec1402476857c931](https://github.com/user-attachments/assets/e6c40f2a-4ca2-4ed6-a1f0-2d70767a5b55)

**Surfacia** (Surface Atomic Chemical Interaction Analyzer) is a comprehensive toolset designed to automate the workflow for analyzing surface atomic chemical interactions. It integrates molecular structure generation, quantum chemical computations, feature extraction, and machine learning analysis, providing a streamlined solution for surface chemistry research.

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

### **Command Line Usage**  
 **1. Activate Environment and Install Surfacia**  
conda activate sympy37  # Replace 'sympy37' with your environment name if needed  
cd /path/to/Surfacia/  # Replace with the actual path to your Surfacia directory  
python setup.py install  
**2. Convert SMILES to XYZ**  
python scripts/surfacia_main.py smi2xyz --smiles_csv data/input_smiles.csv # Replace 'data/input_smiles.csv' with your SMILES input file  
**3. Reorder Atoms (Optional)**  
python scripts/surfacia_main.py reorder --element <element_symbol> --input_dir data/input_xyz/ --output_dir data/reordered_xyz/ # Replace placeholders with actual values  
**4. Convert XYZ to Gaussian Input Files**  
python scripts/surfacia_main.py xyz2gaussian --xyz_folder data/input_xyz/ --template_file config/template.com --output_dir data/gaussian_input/ # Replace placeholders with actual values  
**5. Run Gaussian Calculations**  
python scripts/surfacia_main.py run_gaussian --com_dir data/gaussian_input/ --esp_descriptor_dir config/ESP_descriptor.txt1 # Replace placeholders with actual values  
**6. Process Multiwfn Output**  
python scripts/surfacia_main.py readmultiwfn --input_dir data/gaussian_output/ --output_dir data/multiwfn_output/ --smiles_target_csv data/input_smiles.csv # Replace placeholders with actual values  
**7. Run Machine Learning Analysis**  
python scripts/surfacia_main.py machinelearning --ml_input_dir data/ml_input/ --test_indices <test_indices> # Replace placeholders with actual values  

---
### I. Descriptor List Extracted from the Program

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
  
---
## **Developer**
**Dr. Yuming Su** is the primary developer of **Surfacia: Surface Atomic Chemical Interaction Analyzer**. Dr. Su completed his Ph.D. in 2024 from the **College of Chemistry and Chemical Engineering** at **Xiamen University**.

---

## **Citation**
If you use **Surfacia** in your work, please cite the following:

Su, Y. **Surfacia: Surface Atomic Chemical Interaction Analyzer** (Version 0.0.1) [Software]. 2024. Available at: [GitHub Repository URL]
