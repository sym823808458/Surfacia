## **Surfacia**
**SURF Atomic Chemical Interaction Analyzer - Surfacia**
![b3897a252151144e7189324c07bb5dd](https://github.com/user-attachments/assets/b018d383-4626-494c-86d0-8748a7c35cb4)

**Surfacia** (Surface Atomic Chemical Interaction Analyzer) is a powerful tool that automates the workflow for analyzing surface atomic chemical interactions and generating interpretable descriptors, focusing on properties that are directly relevant to chemists. It extracts molecular descriptors related to electrostatic potential (ESP) and electron interaction properties, such as averaged local ionization energy (ALIE) and local electron attachment energy (LEAE). These descriptors are particularly useful for quantifying electron-withdrawing effects and steric hindrance, which are key factors in understanding molecular behavior in chemical interactions.

By leveraging these surface properties, Surfacia provides a chemically interpretable framework for quantitative structure-property relationships (QSPR). The integration of molecular structure generation, quantum chemical computations, feature extraction, and machine learning analysis within Surfacia offers a comprehensive and streamlined solution for interpretable machine learning in chemistry research.

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

## **Workflow Overview**

### **Surfacia Command Line Usage Manual**

This manual provides a step-by-step guide on how to use the Surfacia software through the command line. The steps include activating the appropriate environment, running various tasks within Surfacia, and installing the software as needed.

### **Prerequisites**
- Ensure you have **Anaconda** or **Miniconda** installed.
- **Surfacia** software should be cloned or downloaded to your machine.
- Dependencies should be installed (you can find them in `requirements.txt`).

### **Activate Environment and Install Surfacia**

conda activate sympy37  # Replace 'sympy37' with your environment name if needed  
cd /path/to/Surfacia/  # Replace with the actual path to your Surfacia directory  
python setup.py install  

### **step-by-step python guide**
Convert SMILES to XYZ  
python scripts/surfacia_main.py smi2xyz --smiles_csv data/177smiles.csv  

Reorder Atoms  
python scripts/surfacia_main.py reorder --element P --input_dir data/<timestamp>/ --output_dir data/<timestamp>/reorder/  

Convert XYZ to Gaussian Input Files  
python scripts/surfacia_main.py xyz2gaussian --xyz_folder data/<timestamp>/reorder/ --template_file config/template.com --output_dir data/<timestamp>/reorder/gaussian_files  

Run Gaussian Calculations and Convert Checkpoint Files  
python scripts/surfacia_main.py run_gaussian --com_dir data/<timestamp>/reorder/gaussian_files/ --esp_descriptor_dir config/ESP_descriptor.txt1  

Read Multiwfn Output  
python scripts/surfacia_main.py readmultiwfn --input_dir /home/yumingsu/Python/Project_Surfacia/Surfacia/data/20241010-150850/reorder/gaussian_files/ --output_dir /home/yumingsu/Python/Project_Surfacia/Surfacia/data/20241010-150850/reorder/gaussian_files/ --smiles_target_csv /home/yumingsu/Python/Project_Surfacia/Surfacia/data/177smiles.csv  

Run Machine Learning Analysis  
python scripts/surfacia_main.py machinelearning --ml_input_dir data/ml_input/ --test_indices <test_indices> # Replace placeholders with actual values  

---

## **Developer**
**Dr. Yuming Su** is the primary developer of **Surfacia: Surface Atomic Chemical Interaction Analyzer**. Dr. Su completed his Ph.D. in 2024 from the **College of Chemistry and Chemical Engineering** at **Xiamen University**.  
orcid=0009-0001-9510-4173

---

## **Citation**
If you use **Surfacia** in your work, please cite the following:

Su, Y. **Surfacia: Surface Atomic Chemical Interaction Analyzer** (Version 0.0.1) [Software]. 2024. Available at: [GitHub Repository URL]
