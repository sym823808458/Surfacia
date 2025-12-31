Core Modules
=============

This section documents the core modules of Surfacia that handle quantum chemical calculations, molecular structure conversion, and workflow orchestration.

.. module:: surfacia.core.gaussian

Gaussian Module
---------------

.. automodule:: surflacia.core.gaussian
   :members:
   :undoc-members:
   :show-inheritance:

The Gaussian module provides functions for converting molecular structures to Gaussian input files and running quantum chemical calculations.

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.gaussian.xyz_to_com

.. autofunction:: surfacia.core.gaussian.process_xyz_files

.. autofunction:: surfacia.core.gaussian.xyz2gaussian_main

.. autofunction:: surfacia.core.gaussian.run_gaussian

.. autofunction:: surfacia.core.gaussian.rerun_failed_calculations

Constants
~~~~~~~~~

.. data:: GAUSSIAN_KEYWORD_LINE

   Default Gaussian calculation keywords line. Default: ``"# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3"``

.. data:: DEFAULT_CHARGE

   Default molecular charge. Default: ``0``

.. data:: DEFAULT_MULTIPLICITY

   Default spin multiplicity. Default: ``1``

.. data:: DEFAULT_NPROC

   Default number of processors. Default: ``32``

.. data:: DEFAULT_MEMORY

   Default memory allocation. Default: ``"30GB"``

Examples
~~~~~~~~

**Convert XYZ to Gaussian input:**

.. code-block:: python

   from surfacia.core.gaussian import xyz_to_com
   
   # Convert a single XYZ file to COM format
   xyz_to_com("molecule_000001.xyz")

**Process all XYZ files:**

.. code-block:: python

   from surfacia.core.gaussian import process_xyz_files
   
   # Convert all XYZ files in current directory
   process_xyz_files()

**Run Gaussian calculations:**

.. code-block:: python

   from surfacia.core.gaussian import run_gaussian
   
   # Run Gaussian on all .com files in current directory
   run_gaussian()

**Customize parameters:**

.. code-block:: python

   from surfacia.core import gaussian
   
   # Modify default settings
   gaussian.GAUSSIAN_KEYWORD_LINE = "# B3LYP/6-31G* opt freq"
   gaussian.DEFAULT_CHARGE = -1
   gaussian.DEFAULT_MULTIPLICITY = 2
   gaussian.DEFAULT_NPROC = 64
   gaussian.DEFAULT_MEMORY = "50GB"
   
   # Then process files with new settings
   gaussian.process_xyz_files()

**Rerun failed calculations:**

.. code-block:: python

   from surfacia.core.gaussian import rerun_failed_calculations
   
   # Automatically detect and rerun failed calculations
   rerun_failed_calculations()

.. module:: surfacia.core.smi2xyz

SMILES to XYZ Module
---------------------

.. automodule:: surfacia.core.smi2xyz
   :members:
   :undoc-members:
   :show-inheritance:

The SMILES to XYZ module handles conversion of SMILES strings from CSV files to 3D molecular coordinates in XYZ format.

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.smi2xyz.read_smiles_csv

.. autofunction:: surfacia.core.smi2xyz.clean_column_name

.. autofunction:: surfacia.core.smi2xyz.validate_csv_columns

.. autofunction:: surfacia.core.smi2xyz.smiles_to_xyz

.. autofunction:: surfacia.core.smi2xyz.smi2xyz_main

Examples
~~~~~~~~

**Convert SMILES to XYZ:**

.. code-block:: python

   from surfacia.core.smi2xyz import smi2xyz_main
   
   # Convert SMILES from CSV to XYZ files
   smi2xyz_main("molecules.csv")

**With custom extension checking:**

.. code-block:: python

   from surfacia.core.smi2xyz import smi2xyz_main
   
   # Check for .fchk files to skip already processed samples
   smi2xyz_main("molecules.csv", check_extensions=['.fchk', '.log'])

**Validate CSV structure:**

.. code-block:: python

   from surfacia.core.smi2xyz import read_smiles_csv, validate_csv_columns
   
   # Read and validate CSV
   data = read_smiles_csv("molecules.csv")
   if validate_csv_columns(data):
       print("CSV is valid and ready for processing")

**Clean column names:**

.. code-block:: python

   from surfacia.core.smi2xyz import clean_column_name
   
   # Clean problematic column names
   cleaned = clean_column_name("Experimental Value (mol/L)")
   # Returns: "experimental_value_mol_l"

**Single SMILES conversion:**

.. code-block:: python

   from surfacia.core.smi2xyz import smiles_to_xyz
   
   # Convert a single SMILES string
   xyz_content, success = smiles_to_xyz("CCO", "ethanol")
   if success:
       with open("ethanol.xyz", 'w') as f:
           f.write(xyz_content)

.. module:: surfacia.core.xtb_opt

XTB Optimization Module
-----------------------

.. automodule:: surfacia.core.xtb_opt
   :members:
   :undoc-members:
   :show-inheritance:

The XTB optimization module performs geometry optimizations using the xTB program with customizable parameters.

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.xtb_opt.run_xtb_opt

Examples
~~~~~~~~

**Run with default parameters:**

.. code-block:: python

   from surfacia.core.xtb_opt import run_xtb_opt
   
   # Run XTB optimization with default settings
   run_xtb_opt()

**Run with custom parameter file:**

.. code-block:: python

   from surfacia.core.xtb_opt import run_xtb_opt
   
   # Create a parameter file with custom options
   with open("xtb_params.txt", 'w') as f:
       f.write("--opt verytight --gfn 1 --molden")
   
   # Run with custom parameters
   run_xtb_opt("xtb_params.txt")

**Recommended parameter combinations:**

.. code-block:: python

   # Fast optimization (development/testing)
   param_fast = "--opt crude --gfn 1 --molden"
   
   # Standard optimization (most cases)
   param_standard = "--opt normal --gfn 2 --molden"
   
   # High quality optimization (final calculations)
   param_hq = "--opt tight --gfn 2 --molden --alpb water"
   
   # Very high quality (critical structures)
   param_vhq = "--opt verytight --gfn 2 --molden --alpb water"

**Process outputs:**

After running XTB optimization:

- Optimized XYZ files overwrite originals
- ``*.molden.input`` files for visualization
- ``*.out`` log files for each molecule
- Temporary files automatically cleaned up

.. module:: surfacia.core.multiwfn

Multiwfn Module
----------------

.. automodule:: surfacia.core.multiwfn
   :members:
   :undoc-members:
   :show-inheritance:

The Multiwfn module handles running Multiwfn calculations for surface analysis and processing the results into structured format.

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.multiwfn.extract_after

.. autofunction:: surfacia.core.multiwfn.extract_before

.. autofunction:: surfacia.core.multiwfn.extract_between

.. autofunction:: surfacia.core.multiwfn.create_descriptors_content

.. autofunction:: surfacia.core.multiwfn.run_multiwfn_on_fchk_files

.. autofunction:: surfacia.core.multiwfn.align_with_mapping_simple

.. autofunction:: surfacia.core.multiwfn.process_txt_files

Examples
~~~~~~~~

**Run Multiwfn analysis:**

.. code-block:: python

   from surfacia.core.multiwfn import run_multiwfn_on_fchk_files
   
   # Run on all .fchk files in current directory
   run_multiwfn_on_fchk_files()

**Run on specific directory:**

.. code-block:: python

   from surfacia.core.multiwfn import run_multiwfn_on_fchk_files
   
   # Process files in specific directory
   run_multiwfn_on_fchk_files(
       input_path="./calculations",
       mapping_file="./sample_mapping.csv"
   )

**Process Multiwfn output files:**

.. code-block:: python

   from surfacia.core.multiwfn import process_txt_files
   
   # Convert Multiwfn TXT files to CSV format
   process_txt_files(
       input_directory="./multiwfn_output",
       output_directory="./processed_results"
   )

**Text extraction utilities:**

.. code-block:: python

   from surfacia.core.multiwfn import (
       extract_after,
       extract_before,
       extract_between
   )
   
   text = "Result: 42.5, Final answer: yes"
   
   # Extract text after a keyword
   after = extract_after(text, "Result:")
   # Returns: " 42.5, Final answer: yes"
   
   # Extract text before a keyword
   before = extract_before(text, "Final")
   # Returns: "Result: 42.5, "
   
   # Extract text between delimiters
   between = extract_between(text, "Result:", "Final")
   # Returns: " 42.5, "

.. module:: surfacia.core.workflow

Workflow Module
----------------

.. automodule:: surfacia.core.workflow
   :members:
   :undoc-members:
   :show-inheritance:

The workflow module provides the main workflow orchestration class that coordinates all analysis steps from SMILES to SHAP visualization.

Classes
~~~~~~~

.. autoclass:: surfacia.core.workflow.SurfaciaWorkflow
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.workflow.workflow_main

Workflow Pipeline
~~~~~~~~~~~~~~~~~

The :class:`SurfaciaWorkflow` class implements a complete analysis pipeline with the following steps:

1. **SMILES to XYZ** (Step 1): Convert SMILES strings to 3D structures
2. **XTB Optimization** (Step 2): Optional geometry optimization
3. **XYZ to Gaussian** (Step 3): Generate Gaussian input files
4. **Run Gaussian** (Step 4): Execute quantum chemical calculations
5. **Multiwfn Analysis** (Step 5): Surface property calculations
6. **Feature Extraction** (Step 6): Extract molecular descriptors
7. **ML Analysis** (Step 7): Machine learning and feature selection
8. **SHAP Visualization** (Step 8): Interactive result exploration

Examples
~~~~~~~~

**Basic workflow:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Initialize workflow with input data
   workflow = SurfaciaWorkflow("molecules.csv")
   
   # Run complete workflow
   workflow.run_full_workflow()
   
   # Get workflow summary
   summary = workflow.get_workflow_summary()
   print(summary)

**Advanced workflow configuration:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Initialize with custom settings
   workflow = SurfaciaWorkflow(
       input_csv="molecules.csv",
       working_dir="./my_analysis"
   )
   
   # Run with custom parameters
   workflow.run_full_workflow(
       resume=True,
       skip_xtb=True,
       gaussian_keywords="# B3LYP/6-31G* opt freq",
       extract_mode=3,
       test_samples=["79", "22", "82"],
       max_features=8,
       stepreg_runs=5,
       api_key="your_zhipuai_api_key"
   )

**Using workflow_main function:**

.. code-block:: python

   from surfacia.core.workflow import workflow_main
   
   # Quick workflow execution
   workflow_main(
       input_csv="molecules.csv",
       test_samples="79,22,82,36,70,80",
       max_features=5,
       stepreg_runs=3
   )

**Step-by-step workflow:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   workflow = SurfaciaWorkflow("molecules.csv")
   
   # Run individual steps
   workflow._step1_smi2xyz()
   workflow._step2_xtb_opt()
   workflow._step3_xyz2gaussian(
       keywords="# PBE1PBE/6-311g*",
       charge=0,
       multiplicity=1
   )
   workflow._step4_run_gaussian()
   workflow._step5_multiwfn_analysis()
   workflow._step6_extract_features(mode=3)
   workflow._step7_ml_analysis(
       max_features=5,
       stepreg_runs=3,
       epoch=64
   )
   workflow._step8_shap_visualization(api_key="your_key")

**Workflow with resumption:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Resume from interrupted workflow
   workflow = SurfaciaWorkflow("molecules.csv")
   workflow.run_full_workflow(resume=True)
   
   # Resume automatically detects:
   # - Completed .fchk files
   # - Failed calculations
   # - Skips to appropriate step

.. module:: surfacia.core.rerun_gaussian

Rerun Gaussian Module
----------------------

.. automodule:: surfacia.core.rerun_gaussian
   :members:
   :undoc-members:
   :show-inheritance:

The rerun Gaussian module provides functionality for recovering from failed calculations and system interruptions.

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.rerun_gaussian.rerun_failed_gaussian_calculations

.. autofunction:: surfacia.core.rerun_gaussian.main

Examples
~~~~~~~~

**Rerun failed calculations:**

.. code-block:: python

   from surfacia.core.rerun_gaussian import rerun_failed_gaussian_calculations
   
   # Automatically detect and rerun failed calculations
   rerun_failed_gaussian_calculations()

**Using main function:**

.. code-block:: python

   from surfacia.core.rerun_gaussian import main
   
   # Execute recovery process
   main()

**Recovery Scenarios:**

The rerun functionality handles:

1. **Empty .fchk files**: Detects and reprocesses calculations that produced empty output
2. **Missing .fchk files**: Identifies molecules with .com files but no corresponding .fchk
3. **Corrupted .chk files**: Removes corrupted checkpoint files before rerunning
4. **System crashes**: Recovers from interrupted calculation runs

**Integration with workflow:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   from surfacia.core.rerun_gaussian import rerun_failed_gaussian_calculations
   
   # Attempt recovery before running workflow
   rerun_failed_gaussian_calculations()
   
   # Then run or resume workflow
   workflow = SurfaciaWorkflow("molecules.csv")
   workflow.run_full_workflow(resume=True)

.. module:: surfacia.core.descriptors

Descriptors Module
------------------

.. automodule:: surfacia.core.descriptors
   :members:
   :undoc-members:
   :show-inheritance:

The descriptors module provides functions for calculating molecular shape and size descriptors from atomic coordinates.

Functions
~~~~~~~~~

.. autofunction:: surfacia.core.descriptors.get_atomic_mass

.. autofunction:: surfacia.core.descriptors.calculate_principal_moments_of_inertia

.. autofunction:: surfacia.core.descriptors.calculate_asphericity

.. autofunction:: surfacia.core.descriptors.calculate_gyradius

.. autofunction:: surfacia.core.descriptors.calculate_relative_gyradius

.. autofunction:: surfacia.core.descriptors.calculate_waist_variance

.. autofunction:: surfacia.core.descriptors.calculate_geometric_asphericity

Examples
~~~~~~~~

**Calculate molecular descriptors:**

.. code-block:: python

   from surfacia.core.descriptors import (
       get_atomic_mass,
       calculate_principal_moments_of_inertia,
       calculate_gyradius,
       calculate_asphericity
   )
   
   # Atomic coordinates (example: water)
   coords = [
       [0.0, 0.0, 0.0],      # O
       [0.957, 0.0, 0.0],   # H1
       [-0.239, 0.927, 0.0]  # H2
   ]
   
   # Atomic masses
   masses = [get_atomic_mass('O'), 
             get_atomic_mass('H'), 
             get_atomic_mass('H')]
   
   # Calculate moments of inertia
   I1, I2, I3 = calculate_principal_moments_of_inertia(coords, masses)
   
   # Calculate radius of gyration
   rg = calculate_gyradius(coords, masses)
   
   # Calculate asphericity
   asphericity = calculate_asphericity(I1, I2, I3)

**Complete shape analysis:**

.. code-block:: python

   from surfacia.core.descriptors import (
       calculate_principal_moments_of_inertia,
       calculate_gyradius,
       calculate_asphericity,
       calculate_relative_gyradius,
       calculate_waist_variance,
       calculate_geometric_asphericity
   )
   
   def analyze_molecular_shape(coords, element_symbols):
       masses = [get_atomic_mass(e) for e in element_symbols]
       
       I1, I2, I3 = calculate_principal_moments_of_inertia(coords, masses)
       rg = calculate_gyradius(coords, masses)
       asp = calculate_asphericity(I1, I2, I3)
       rel_rg = calculate_relative_gyradius(rg, I1, I2, I3)
       wv = calculate_waist_variance(coords, masses)
       g_asp = calculate_geometric_asphericity(I1, I2, I3)
       
       return {
           'I1': I1,
           'I2': I2,
           'I3': I3,
           'gyradius': rg,
           'asphericity': asp,
           'relative_gyradius': rel_rg,
           'waist_variance': wv,
           'geometric_asphericity': g_asp
       }

**Atomic mass lookup:**

.. code-block:: python

   from surfacia.core.descriptors import get_atomic_mass
   
   # Get atomic masses for common elements
   print(get_atomic_mass('H'))   # 1.008
   print(get_atomic_mass('C'))   # 12.011
   print(get_atomic_mass('O'))   # 15.999
   print(get_atomic_mass('N'))   # 14.007
   print(get_atomic_mass('S'))   # 32.065

**Shape descriptors interpretation:**

- **Asphericity**: Measures deviation from spherical shape (0 = sphere)
- **Radius of gyration**: Measure of molecular size
- **Relative gyradius**: Size relative to moment of inertia
- **Waist variance**: Molecular width variation
- **Geometric asphericity**: Shape based on principal axes

**Processing multiple molecules:**

.. code-block:: python

   import numpy as np
   from pathlib import Path
   
   def read_xyz_file(filepath):
       """Read XYZ file and return coordinates and symbols."""
       with open(filepath, 'r') as f:
           lines = f.readlines()
           n_atoms = int(lines[0].strip())
           coords = []
           symbols = []
           for line in lines[2:2+n_atoms]:
               parts = line.split()
               symbols.append(parts[0])
               coords.append([float(x) for x in parts[1:4]])
           return np.array(coords), symbols
   
   def batch_descriptors(xyz_files):
       """Calculate descriptors for multiple XYZ files."""
       results = []
       for xyz_file in xyz_files:
           coords, symbols = read_xyz_file(xyz_file)
           desc = analyze_molecular_shape(coords, symbols)
           desc['filename'] = xyz_file
           results.append(desc)
       return results

Integration and Usage
=====================

The core modules are designed to work together seamlessly:

**Complete analysis pipeline:**

.. code-block:: python

   from surfacia.core.smi2xyz import smi2xyz_main
   from surfacia.core.xtb_opt import run_xtb_opt
   from surfacia.core.gaussian import process_xyz_files, run_gaussian
   from surfacia.core.multiwfn import run_multiwfn_on_fchk_files
   
   # Step 1: SMILES to XYZ
   smi2xyz_main("molecules.csv")
   
   # Step 2: XTB optimization (optional)
   run_xtb_opt()
   
   # Step 3: Generate Gaussian inputs
   process_xyz_files()
   
   # Step 4: Run Gaussian
   run_gaussian()
   
   # Step 5: Multiwfn analysis
   run_multiwfn_on_fchk_files()

**Using the workflow class:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Recommended approach: use workflow class
   workflow = SurfaciaWorkflow("molecules.csv")
   workflow.run_full_workflow(
       resume=True,
       max_features=5,
       test_samples="1,5,10,15,20"
   )

**Error handling:**

.. code-block:: python

   from surfacia.core.rerun_gaussian import rerun_failed_gaussian_calculations
   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Recovery pattern
   try:
       workflow = SurfaciaWorkflow("molecules.csv")
       workflow.run_full_workflow()
   except Exception as e:
       print(f"Workflow interrupted: {e}")
       # Attempt recovery
       rerun_failed_gaussian_calculations()
       # Resume workflow
       workflow.run_full_workflow(resume=True)

Best Practices
==============

**1. Use the workflow class for complete analyses:**

The :class:`SurfaciaWorkflow` class provides the most convenient interface for running complete analyses with built-in error handling and resumption capabilities.

**2. Check prerequisites:**

Ensure all required software is installed:

- Gaussian 16 (g16 command)
- Multiwfn
- XTB (optional, for geometry optimization)
- OpenBabel (for SMILES conversion)

**3. Validate input data:**

.. code-block:: python

   from surfacia.core.smi2xyz import read_smiles_csv, validate_csv_columns
   
   data = read_smiles_csv("molecules.csv")
   if not validate_csv_columns(data):
       print("Invalid CSV format")
       return

**4. Monitor progress:**

All modules provide console output with progress information. Monitor these messages to track analysis progress.

**5. Use resumption for long calculations:**

The ``resume=True`` option in workflow enables intelligent continuation, saving significant time for long-running analyses.

**6. Backup important data:**

Before running large analyses, backup:

- Input CSV files
- Generated XYZ files
- Intermediate results

**7. Resource management:**

Adjust computational resources based on molecule size and available hardware:

.. code-block:: python

   from surfacia.core import gaussian
   
   # For small molecules
   gaussian.DEFAULT_NPROC = 16
   gaussian.DEFAULT_MEMORY = "20GB"
   
   # For large molecules
   gaussian.DEFAULT_NPROC = 64
   gaussian.DEFAULT_MEMORY = "100GB"

**8. Error recovery:**

Use the rerun functionality to recover from failed calculations:

.. code-block:: python

   from surfacia.core.rerun_gaussian import rerun_failed_gaussian_calculations
   
   # Always attempt recovery before starting workflow
   rerun_failed_gaussian_calculations()

**9. Batch processing:**

For multiple datasets, use shell scripts or Python loops:

.. code-block:: python

   datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
   
   for dataset in datasets:
       print(f"Processing {dataset}...")
       workflow = SurfaciaWorkflow(dataset)
       workflow.run_full_workflow()

**10. Result organization:**

The workflow automatically organizes results in timestamped directories. Follow this convention for custom analyses:

```
Surfacia_3.0_YYYYMMDD_HHMMSS/
├── FullOption_*.csv
├── FinalFull_*.csv
├── Training_Set_Detailed_*.csv
├── Test_Set_Detailed_*.csv
└── analysis_results/
```

These core modules provide the foundation for all Surfacia analyses, from simple molecular descriptor calculations to complete end-to-end machine learning pipelines.
