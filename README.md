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

## **Developer**
**Dr. Yuming Su** is the primary developer of **Surfacia: Surface Atomic Chemical Interaction Analyzer**. Dr. Su completed his Ph.D. in 2024 from the **College of Chemistry and Chemical Engineering** at **Xiamen University**.

---

## **Citation**
If you use **Surfacia** in your work, please cite the following:

Su, Y. **Surfacia: Surface Atomic Chemical Interaction Analyzer** (Version 0.0.1) [Software]. 2024. Available at: [GitHub Repository URL]
