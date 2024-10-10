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

### **Generate 3D Structures**
Run the `smi2xyz.py` script to convert SMILES strings from the input CSV into 3D coordinates (XYZ format). The input CSV should contain two columns: `SMILES` and `TARGET`. The script outputs multiple `.xyz` files (e.g., `000001.xyz`, `000002.xyz`, etc.).

### **Reorder Atom Positions (Optional)**
Run the `extract_element_first.py` script to reorder atoms in XYZ files. This step allows for rearranging specific atoms to the beginning of the file. Input and output are both XYZ files, and the reordered files are saved in the `reorder/` folder.

### **XTB Structure Optimization (Optional)**
Run the `run_xtb.sh` script (submitted via `sbatch run_xtb.sh`) to perform rough optimization using **XTB**. Input and output are XYZ files, and optimized files are saved in the `reorder/xtb_opted/` folder.

### **Generate Gaussian Input Files**
Run the `xyz2gaussian.py` script to convert XYZ files into Gaussian input files (`.com` format). It also generates a Slurm script (`run_gaussian.sh`) for submitting Gaussian jobs.  
**Input**: XYZ and `template.com` files.  
**Output**: multiple `.com` and `.sh` files.

### **Run Gaussian Calculations**
Submit the `.com` files and Slurm script via `sbatch run_gaussian.sh` to begin Gaussian calculations.  
**Input**: `.com` and `.sh` files.  
**Output**: `.chk` and `.log` files.

### **Convert Wavefunction Files**
After Gaussian completes, run `chk2fchk.sh` to convert `.chk` files into `.fchk` format (required by Multiwfn).  
**Input**: `.chk` files.  
**Output**: `.fchk` files.

### **Calculate Descriptors**
Run the `run_Multiwfn.sh` script (submitted via `sbatch Multiwfn.sh`) to compute molecular descriptors using **Multiwfn**. The script calculates both global and surface atomic interaction descriptors, saving results in `.txt` files.

### **Read Descriptor Data**
Run the `readMultiwfn.py` script to extract descriptors from the `.txt` files and compile them into a feature matrix. This matrix, along with the `smiles` and `target` columns from the initial CSV, is merged into a single file and saved as a `.csv`.

### **Split Feature Matrix**
Run `split_featurematrix.py` to split the merged data into four separate CSV files: `Smiles`, `Value`, `Feature`, and `Title`, for use in machine learning.

### **Machine Learning Analysis**
Run the `XGB_Stepregression.py` script (submitted via `sbatch python.sh`) to perform machine learning analysis using the feature matrix. You must provide a CSV file containing `title`, `featurematrix`, `label`, and `SMILES`. The script outputs machine learning results and SHAP analysis plots.

---

## **Developer**
**Dr. Yuming Su** is the primary developer of **Surfacia: Surface Atomic Chemical Interaction Analyzer**. Dr. Su completed his Ph.D. in 2024 from the **College of Chemistry and Chemical Engineering** at **Xiamen University**.  
orcid=0009-0001-9510-4173

---

## **Citation**
If you use **Surfacia** in your work, please cite the following:

Su, Y. **Surfacia: Surface Atomic Chemical Interaction Analyzer** (Version 0.0.1) [Software]. 2024. Available at: [GitHub Repository URL]
