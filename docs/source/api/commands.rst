Command Reference
=================

This section provides detailed documentation for all Surfacia command-line interface (CLI) commands. Each command is described with its purpose, syntax, parameters, and practical examples based on the actual implementation in ``cli.py``.

Main Workflow Commands
----------------------

The main workflow commands provide a complete end-to-end analysis pipeline from SMILES input to machine learning results and interactive visualization.

workflow
~~~~~~~~

Complete end-to-end analysis pipeline that automates all steps from SMILES to SHAP visualization. This is the recommended command for running full analyses.

**Syntax:**
```bash
surfacia workflow -i INPUT_FILE [OPTIONS]
```

**Required Parameters:**

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Initial CSV file containing SMILES strings and target values.

**Optional Parameters:**

- ``--resume``
  Resume workflow from existing calculations (smart continuation). Automatically detects completed .fchk files and reruns failed calculations only.

- ``--skip-xtb``
  Skip XTB geometry optimization step.

- ``--test-samples TEST_SAMPLES``
  Comma-separated test sample names or numbers for ML analysis (e.g., "79,22,82,36,70,80").

- ``--keywords KEYWORDS``
  Gaussian calculation keywords (default: "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3").

- ``--charge CHARGE``
  Molecular charge (default: 0).

- ``--multiplicity MULTIPLICITY``
  Spin multiplicity (default: 1).

- ``--nproc NPROC``
  Number of processors for Gaussian calculations (default: 32).

- ``--memory MEMORY``
  Memory allocation for Gaussian calculations (default: "30GB").

- ``--extract-mode {1,2,3}``
  Feature extraction mode: 1=element-specific, 2=fragment-specific, 3=LOFFI comprehensive (default: 3).

- ``--extract-element ELEMENT``
  Target element for mode 1 (e.g., S, N, O).

- ``--extract-xyz1 XYZ1``
  Fragment XYZ file for mode 2.

- ``--extract-threshold THRESHOLD``
  Surface analysis threshold for mode 2 (default: 0.001).

- ``--max-features MAX_FEATURES``
  Maximum features for ML analysis (default: 5).

- ``--stepreg-runs STEPREG_RUNS``
  Number of stepwise regression runs (default: 3).

- ``--initial-features INITIAL_FEATURES``
  Comma-separated initial features for stepwise regression.

- ``--train-test-split TRAIN_TEST_SPLIT``
  Train/test split ratio (default: 0.85).

- ``--shap-fit-threshold SHAP_FIT_THRESHOLD``
  SHAP R² threshold for feature recommendation (default: 0.3).

- ``--no-generate-fitting``
  Disable generation of fitting plots for manual mode.

- ``--epoch EPOCH``
  Number of training epochs (default: 64).

- ``--cores CORES``
  Number of CPU cores for ML analysis (default: 32).

- ``--api-key API_KEY``
  ZhipuAI API key for SHAP visualization AI assistant.

- ``--port PORT``
  Port for SHAP visualization server (default: 8052).

- ``--host HOST``
  Host for SHAP visualization server (default: "0.0.0.0").

**Examples:**

.. code-block:: bash

   # Complete workflow with default settings
   surfacia workflow -i molecules.csv
   
   # Resume workflow from existing calculations
   surfacia workflow -i molecules.csv --resume
   
   # Skip XTB optimization and use custom Gaussian settings
   surfacia workflow -i molecules.csv --skip-xtb --keywords "# B3LYP/6-31G* opt freq"
   
   # Include test samples and customize ML parameters
   surfacia workflow -i molecules.csv --test-samples "1,5,10,15,20" --max-features 8 --stepreg-runs 5
   
   # Full custom workflow
   surfacia workflow -i molecules.csv \
       --resume \
       --keywords "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3" \
       --nproc 64 \
       --memory "50GB" \
       --test-samples "79,22,82,36,70,80" \
       --max-features 8 \
       --stepreg-runs 5 \
       --epoch 128 \
       --cores 64 \
       --api-key "your_zhipuai_api_key"

**Pipeline Steps:**

1. **SMILES to XYZ Conversion**: Convert SMILES strings to 3D coordinates
2. **XTB Optimization**: Optional geometry optimization with XTB
3. **Gaussian Input Generation**: Create Gaussian .com files
4. **Gaussian Calculations**: Run quantum chemical calculations
5. **Multiwfn Analysis**: Surface and electronic property analysis
6. **Feature Extraction**: Extract molecular surface descriptors
7. **Machine Learning Analysis**: Feature selection and model training
8. **Interactive SHAP Visualization**: Web-based result exploration

**Resume Mode:**

The ``--resume`` flag enables intelligent continuation:
- Automatically detects completed calculations
- Reruns only failed Gaussian calculations
- Skips to appropriate step based on existing files
- Saves significant computation time

**Output Files:**

- ``Surfacia_3.0_*/``: Main output directory with timestamp
- ``FullOption*.csv``: Raw feature extraction results
- ``FinalFull*.csv``: Processed features for ML analysis
- ``Training_Set_Detailed*.csv``: Training set with SHAP values
- ``Test_Set_Detailed*.csv``: Test set with SHAP values (if test samples specified)
- Various analysis plots and summary files

---

Individual Step Commands
------------------------

These commands allow you to run individual steps of the workflow for more granular control.

smi2xyz
~~~~~~~~

Convert SMILES strings from CSV file to XYZ coordinate files.

**Syntax:**
```bash
surfacia smi2xyz -i INPUT_FILE
```

**Required Parameters:**

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Input CSV file containing SMILES strings.

**Functionality:**

1. Reads SMILES strings from input CSV
2. Converts each SMILES to 3D molecular structure
3. Saves as numbered XYZ files (000001.xyz, 000002.xyz, etc.)
4. Creates ``sample_mapping.csv`` with original data
5. Cleans column names and handles missing values

**Example:**

.. code-block:: bash

   surfacia smi2xyz -i molecules.csv

**Output Files:**

- ``000001.xyz, 000002.xyz, ...``: Molecular structure files
- ``sample_mapping.csv``: Mapping between original data and XYZ files

**Error Handling:**

- Validates CSV structure before processing
- Skips samples with empty SMILES strings
- Reports conversion failures with specific error messages

---

xtb-opt
~~~~~~~~

Perform XTB geometry optimization on all XYZ files in current directory.

**Syntax:**
```bash
surfacia xtb-opt [OPTIONS]
```

**Optional Parameters:**

- ``--method {gfn1,gfn2}``
  XTB method to use (default: gfn2).

- ``--opt-level {crude,sloppy,loose,normal,tight,verytight}``
  Optimization convergence level (default: normal).

- ``--solvent SOLVENT``
  Solvent for ALPB model (default: none).

- ``--params PARAMS``
  Custom XTB parameter file with command-line options.

**Functionality:**

1. Processes all ``*.xyz`` files in current directory
2. Runs XTB optimization with specified parameters
3. Overwrites original XYZ files with optimized geometries
4. Generates molden input files for visualization
5. Cleans up temporary files

**Examples:**

.. code-block:: bash

   # Default XTB optimization
   surfacia xtb-opt
   
   # Use GFN1-xTB with loose optimization
   surfacia xtb-opt --method gfn1 --opt-level loose
   
   # Optimize in water solvent with tight convergence
   surfacia xtb-opt --solvent water --opt-level tight
   
   # Use custom parameter file
   surfacia xtb-opt --params my_xtb_params.txt

**Output Files:**

- Optimized ``*.xyz`` files (overwrite originals)
- ``*.molden.input`` files for each molecule
- ``*.out`` log files for each calculation

**Error Handling:**

- 5-minute timeout per molecule
- Skips failed molecules and continues with others
- Reports success/failure counts

---

xyz2gaussian
~~~~~~~~~~~~

Generate Gaussian input files from XYZ coordinates.

**Syntax:**
```bash
surfacia xyz2gaussian [OPTIONS]
```

**Optional Parameters:**

- ``--keywords KEYWORDS``
  Gaussian calculation keywords (default: "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3").

- ``--charge CHARGE``
  Molecular charge (default: 0).

- ``--multiplicity MULTIPLICITY``
  Spin multiplicity (default: 1).

- ``--nproc NPROC``
  Number of processors (default: 32).

- ``--memory MEMORY``
  Memory allocation (default: "30GB").

**Functionality:**

1. Processes all ``*.xyz`` files in current directory
2. Creates Gaussian ``*.com`` files with specified parameters
3. Includes checkpoint file specifications
4. Uses molecular charge and multiplicity settings

**Examples:**

.. code-block:: bash

   # Default settings
   surfacia xyz2gaussian
   
   # Custom method and basis set
   surfacia xyz2gaussian --keywords "# B3LYP/6-31G* opt freq"
   
   # Charged molecule with more resources
   surfacia xyz2gaussian --charge -1 --multiplicity 2 --nproc 64 --memory 50GB

**Output Files:**

- ``*.com`` files for each XYZ file
- Each file includes checkpoint file specification (``*.chk``)

**Error Handling:**

- Validates XYZ file format before conversion
- Reports successful conversions with file names

---

run-gaussian
~~~~~~~~~~~~

Execute Gaussian calculations for all .com files in current directory.

**Syntax:**
```bash
surfacia run-gaussian
```

**Functionality:**

1. Finds all ``*.com`` files in current directory
2. Runs Gaussian calculations using ``g16`` command
3. Converts ``*.chk`` files to ``*.fchk`` format using ``formchk``
4. Processes files in alphabetical order

**Prerequisites:**

- Gaussian 16 must be installed and accessible as ``g16``
- Sufficient computational resources for specified calculations

**Examples:**

.. code-block:: bash

   surfacia run-gaussian

**Output Files:**

- ``*.log`` files: Gaussian calculation logs
- ``*.chk`` files: Binary checkpoint files
- ``*.fchk`` files: Formatted checkpoint files for Multiwfn

**Error Handling:**

- Continues with remaining files if individual calculations fail
- Reports errors for failed calculations
- No automatic retry mechanism (use ``rerun-gaussian`` for that)

---

multiwfn
~~~~~~~~

Run Multiwfn analysis on all .fchk files to extract surface and electronic properties.

**Syntax:**
```bash
surfacia multiwfn [OPTIONS]
```

**Optional Parameters:**

- ``--input-dir INPUT_DIR``
  Directory containing .fchk files (default: current directory).

- ``--output-dir OUTPUT_DIR``
  Output directory for results (default: current directory).

**Functionality:**

1. **Multiwfn Calculations**: Runs comprehensive surface analysis on each .fchk file
2. **Result Processing**: Processes Multiwfn output files into CSV format
3. **Property Extraction**: Extracts various molecular surface descriptors

**Properties Analyzed:**

- Surface electron density distributions
- Electrostatic potential maps
- Local ionization potentials
- Electron affinity distributions
- Fukui functions
- Various surface descriptors

**Examples:**

.. code-block:: bash

   # Analyze all .fchk files in current directory
   surfacia multiwfn
   
   # Analyze files in specific directory
   surfacia multiwfn --input-dir ./calculations --output-dir ./results

**Output Files:**

- ``*.txt`` files: Raw Multiwfn output for each molecule
- Processed CSV files with extracted features
- Analysis summary files

**Error Handling:**

- Processes each file independently
- Reports failed analyses with specific error messages
- Continues with remaining files

---

extract-features
~~~~~~~~~~~~~~~~

Extract atomic properties and features from Multiwfn results.

**Syntax:**
```bash
surfacia extract-features -i INPUT_FILE --mode MODE [OPTIONS]
```

**Required Parameters:**

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Input CSV file from Multiwfn analysis (typically ``FullOption*.csv``).

- ``--mode {1,2,3}``
  Feature extraction mode:
  
  - **Mode 1**: Element-specific analysis (requires ``--element``)
  - **Mode 2**: Fragment-specific analysis (requires ``--xyz1``)
  - **Mode 3**: LOFFI comprehensive analysis (default)

**Mode-Specific Parameters:**

- ``--element ELEMENT``
  Target element for mode 1 (e.g., S, N, O, C).

- ``--xyz1 XYZ1``
  Fragment XYZ file for mode 2.

- ``--threshold THRESHOLD``
  Surface analysis threshold (default: 0.001).

**Mode Descriptions:**

**Mode 1 - Element-Specific:**
- Analyzes surface properties around specified element
- Extracts local electronic environment descriptors
- Useful for studying specific atomic sites

**Mode 2 - Fragment-Specific:**
- Analyzes properties around specified molecular fragment
- Uses fragment XYZ file to define analysis region
- Good for studying functional groups or substructures

**Mode 3 - LOFFI Comprehensive:**
- Comprehensive surface analysis using LOFFI method
- Analyzes entire molecular surface
- Most commonly used mode

**Examples:**

.. code-block:: bash

   # Mode 1: Sulfur atom analysis
   surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 1 --element S
   
   # Mode 2: Fragment analysis with custom threshold
   surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 2 --xyz1 fragment.xyz --threshold 0.002
   
   # Mode 3: Comprehensive LOFFI analysis
   surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 3

**Output Files:**

- ``FinalFull*.csv``: Processed feature matrix for ML analysis
- Feature selection and preprocessing results
- Analysis summary and statistics

**Error Handling:**

- Validates mode-specific parameters
- Checks input file format and existence
- Reports missing required parameters clearly

---

ml-analysis
~~~~~~~~~~~~

Machine learning analysis with automatic feature selection and SHAP interpretation.

**Syntax:**
```bash
surfacia ml-analysis -i INPUT_FILE [OPTIONS]
```

**Required Parameters:**

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Input CSV file with features (typically ``FinalFull*.csv``).

**Test Set Parameters:**

- ``--test-samples TEST_SAMPLES``
  Comma-separated test sample names or numbers (e.g., "79,22,82,36,70,80").

- ``--nan-handling {drop_rows,drop_columns}``
  How to handle NaN values (default: drop_columns).

**Mode Selection:**

- ``--manual``
  Use manual feature selection mode (default: automatic workflow mode).

**Manual Mode Parameters:**

- ``--manual-features MANUAL_FEATURES``
  Comma-separated feature names for manual mode, or "Full" for all features.

- ``--no-generate-fitting``
  Disable generation of fitting plots for manual mode.

**Workflow Mode Parameters (Default):**

- ``--max-features MAX_FEATURES``
  Maximum features for automatic selection (default: 5).

- ``--stepreg-runs STEPREG_RUNS``
  Number of stepwise regression runs (default: 3).

- ``--initial-features INITIAL_FEATURES``
  Comma-separated initial features for stepwise regression.

- ``--shap-fit-threshold SHAP_FIT_THRESHOLD``
  SHAP R² threshold for feature recommendation (default: 0.3).

**Common Parameters:**

- ``--train-test-split TRAIN_TEST_SPLIT``
  Train/test split ratio (default: 0.85).

- ``--epoch EPOCH``
  Number of training epochs (default: 64).

- ``--cores CORES``
  Number of CPU cores to use (default: 32).

**Analysis Modes:**

**Automatic Workflow Mode (Default):**
- Runs comprehensive baseline analysis with all features
- Performs multiple runs of stepwise regression
- Analyzes SHAP values and fitting quality
- Provides intelligent feature recommendations
- Generates final analysis with recommended features

**Manual Feature Selection Mode:**
- Analyzes user-specified features
- Can use "Full" to analyze all available features
- Generates SHAP analysis for specified features
- Provides detailed performance metrics

**Examples:**

.. code-block:: bash

   # Automatic workflow mode (default)
   surfacia ml-analysis -i ./Surfacia*/FinalFull.csv \
       --test-samples "79,22,82,36,70,80" \
       --max-features 5 \
       --stepreg-runs 3
   
   # Manual mode with specific features
   surfacia ml-analysis -i ./Surfacia*/FinalFull.csv \
       --manual \
       --manual-features "S_ALIE_min,C_LEAE_min,Fun_ESP_delta"
   
   # Manual mode with all features
   surfacia ml-analysis -i ./Surfacia*/FinalFull.csv \
       --manual \
       --manual-features "Full"
   
   # Custom training parameters
   surfacia ml-analysis -i ./Surfacia*/FinalFull.csv \
       --epoch 128 \
       --cores 64 \
       --train-test-split 0.8

**Output Structure:**

**Workflow Mode:**
- ``Baseline_Analysis/``: Analysis with all features
- ``Run_1/, Run_2/, Run_3/``: Stepwise regression results
- ``Final_Manual_Analysis/``: Results with recommended features
- ``feature_recommendations_*.csv``: Ranked feature recommendations
- ``workflow_summary_*.txt``: Complete analysis summary

**Manual Mode:**
- ``Manual_Feature_Analysis_*`` or ``Full_Feature_Analysis_*``: Results directory
- Prediction scatter plots and data
- Training/test set CSV files with SHAP values
- ``SHAP_Plots/``: Individual feature SHAP analysis
- ``SHAP_Raw_Data/``: Raw SHAP data for each feature

**Performance Metrics:**

- **Cross-Validation**: MSE, MAE, R² with standard deviations
- **Test Set Performance**: If test samples specified
- **Feature Rankings**: Based on SHAP values and importance
- **Fitting Analysis**: Mathematical relationships for each feature

**Error Handling:**

- Validates input file format and structure
- Handles missing test samples gracefully
- Provides clear error messages for parameter validation
- Generates detailed analysis records

---

shap-viz
~~~~~~~~

Interactive SHAP visualization with 3D molecular structure display and AI assistant.

**Syntax:**
```bash
surfacia shap-viz -i INPUT_FILE -x XYZ_FOLDER [OPTIONS]
```

**Required Parameters:**

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Training_Set_Detailed CSV file from ML analysis.

- ``-x XYZ_FOLDER``, ``--xyz-folder XYZ_FOLDER``
  Folder containing XYZ files and molecular structure files.

**Optional Parameters:**

- ``--test-csv TEST_CSV``
  Test_Set_Detailed CSV file for test set visualization.

- ``--api-key API_KEY``
  ZhipuAI API key for AI assistant features.

- ``--skip-surface-gen``
  Skip surface PDB generation if files already exist.

- ``--port PORT``
  Port for the web server (default: 8052).

- ``--host HOST``
  Host for the web server (default: 0.0.0.0 for all interfaces).

**Functionality:**

**Interactive Features:**
- SHAP scatter plots with interactive exploration
- 3D molecular visualization with isosurface displays
- Real-time molecular property analysis
- Feature importance visualization

**AI Assistant:**
- Powered by ZhipuAI for intelligent analysis
- Provides insights about molecular features
- Explains SHAP value patterns
- Suggests molecular optimization strategies

**Visualization Capabilities:**
- Interactive 3D molecular structures
- Surface property maps
- Electronic density visualizations
- Feature-based molecular comparisons

**Examples:**

.. code-block:: bash

   # Basic SHAP visualization
   surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_folder
   
   # Include test set and AI assistant
   surfacia shap-viz -i Training_Set_Detailed.csv \
       -x ./xyz_folder \
       --test-csv Test_Set_Detailed.csv \
       --api-key your_zhipuai_api_key
   
   # Custom server configuration
   surfacia shap-viz -i Training_Set_Detailed.csv \
       -x ./xyz_folder \
       --port 8080 \
       --host localhost

**Web Interface:**

- **Main Dashboard**: Overview of analysis results
- **SHAP Explorer**: Interactive feature analysis
- **3D Viewer**: Molecular structure visualization
- **AI Chat**: Intelligent assistant for insights
- **Data Tables**: Detailed numerical results

**Technical Details:**

- Built with Plotly for interactive visualizations
- Uses Three.js for 3D molecular rendering
- Integrates with ZhipuAI API for AI features
- Responsive design for various screen sizes

**Error Handling:**

- Validates input file formats
- Checks for required molecular structure files
- Provides clear error messages for missing dependencies
- Gracefully handles API key issues

---

Utility Tools
-------------

These are independent utility tools for molecular drawing, property analysis, and error recovery.

mol-draw
~~~~~~~~~

Generate 2D molecular structure images from SMILES strings.

**Syntax:**
```bash
surfacia mol-draw (--smiles SMILES | -i INPUT_FILE) [OPTIONS]
```

**Input Options (Mutually Exclusive):**

- ``--smiles SMILES``
  Single SMILES string to draw.

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Input CSV file containing SMILES column.

**Optional Parameters:**

- ``-o OUTPUT``, ``--output OUTPUT``
  Output directory (for CSV) or file path (for single SMILES).

- ``--size WIDTH HEIGHT``
  Image size as width height in pixels (default: 800 800).

- ``--prefix PREFIX``
  Filename prefix for batch processing (default: mol).

**Functionality:**

- Uses RDKit for high-quality molecular drawing
- Supports batch processing from CSV files
- Generates PNG images with 300 DPI resolution
- Customizable image dimensions and styling

**Examples:**

.. code-block:: bash

   # Draw single molecule
   surfacia mol-draw --smiles "CCO" -o ethanol.png
   
   # Batch draw from CSV file
   surfacia mol-draw -i molecules.csv -o molecule_images
   
   # High resolution batch drawing
   surfacia mol-draw -i molecules.csv --size 1200 1200 -o high_res_images
   
   # Custom filename prefix
   surfacia mol-draw -i molecules.csv --prefix compound -o structures

**Output:**

- High-quality PNG images (300 DPI)
- Automatic file naming with SMILES-based names
- Batch processing with progress reporting

**Error Handling:**

- Validates SMILES strings before drawing
- Handles invalid SMILES gracefully
- Reports processing progress for batch operations

---

mol-info
~~~~~~~~

Calculate and display molecular properties for SMILES strings.

**Syntax:**
```bash
surfacia mol-info (--smiles SMILES | -i INPUT_FILE) [OPTIONS]
```

**Input Options (Mutually Exclusive):**

- ``--smiles SMILES``
  Single SMILES string to analyze.

- ``-i INPUT_FILE``, ``--input INPUT_FILE``
  Input CSV file containing SMILES column.

**Optional Parameters:**

- ``-o OUTPUT``, ``--output OUTPUT``
  Output CSV file for molecular properties.

- ``--properties PROPERTIES [PROPERTIES ...]``
  Properties to calculate: mw, logp, hbd, hba, tpsa, rotatable_bonds, aromatic_rings, heavy_atoms (default: mw logp hbd hba tpsa rotatable_bonds).

**Available Properties:**

- ``mw``: Molecular weight
- ``logp``: Partition coefficient (octanol/water)
- ``hbd``: Hydrogen bond donors
- ``hba``: Hydrogen bond acceptors
- ``tpsa``: Topological polar surface area
- ``rotatable_bonds``: Number of rotatable bonds
- ``aromatic_rings``: Number of aromatic rings
- ``heavy_atoms``: Number of heavy atoms

**Examples:**

.. code-block:: bash

   # Analyze single molecule with default properties
   surfacia mol-info --smiles "CCO"
   
   # Analyze specific properties
   surfacia mol-info --smiles "CCO" --properties mw logp hbd hba
   
   # Batch analyze from CSV and save results
   surfacia mol-info -i molecules.csv -o molecular_properties.csv
   
   # Calculate all available properties
   surfacia mol-info -i molecules.csv --properties mw logp hbd hba tpsa rotatable_bonds aromatic_rings heavy_atoms

**Output:**

- Console display for single molecules
- CSV file for batch analysis
- Property values with appropriate units
- Error reporting for invalid SMILES

**Error Handling:**

- Validates SMILES strings before analysis
- Reports calculation errors clearly
- Handles missing properties gracefully

---

rerun-gaussian
~~~~~~~~~~~~~~

Rerun failed Gaussian calculations and recover from interrupted runs.

**Syntax:**
```bash
surfacia rerun-gaussian
```

**Functionality:**

This utility command identifies and reruns failed calculations in two scenarios:

1. **Empty .fchk Files**: Detects and reprocesses calculations that resulted in empty formatted checkpoint files
2. **Missing .fchk Files**: Identifies .xyz files that have corresponding .com files but missing .fchk files

**Process:**

1. Scans current directory for .fchk files and checks file sizes
2. Identifies empty .fchk files (size = 0 bytes)
3. Finds .xyz files without corresponding .fchk files
4. Removes corrupted files (empty .fchk and corresponding .chk files)
5. Reruns Gaussian calculations for identified failures
6. Converts successful .chk files to .fchk format

**Safety Features:**

- Automatically removes corrupted files before rerunning
- Preserves original .com files for reuse
- Provides detailed progress reporting
- Only processes genuinely failed calculations

**Use Cases:**

- **System Crashes**: When calculations were interrupted by system shutdown or crashes
- **Resource Limitations**: When some calculations failed due to memory or time limits
- **Partial Completion**: When only a subset of calculations completed successfully
- **Quality Control**: To ensure all calculations produced valid output

**Examples:**

.. code-block:: bash

   # Check and rerun failed calculations
   surfacia rerun-gaussian

**Output Reports:**

- Number of failed calculations found
- Progress updates for each rerun
- Final success/failure summary
- Detailed error messages for persistent failures

**Error Handling:**

- Validates file integrity before processing
- Continues with remaining calculations if individual reruns fail
- Reports specific error causes for troubleshooting
- Provides clear status updates throughout the process

**Prerequisites:**

- Gaussian 16 must be installed and accessible
- Original .com files must be present for failed calculations
- Sufficient resources to rerun failed calculations

**Integration with Workflow:**

This command is automatically called by the ``workflow --resume`` functionality, but can also be used independently for manual recovery operations.

---

Command Line Interface Overview

Global Help
------------

To get help for any specific command:

.. code-block:: bash

   surfacia --help              # Show all available commands
   surfacia <command> --help    # Show help for specific command
   surfacia workflow --help      # Example: detailed workflow help

Command Categories
-----------------

The Surfacia CLI is organized into several logical categories:

**Main Pipeline Commands:**
- ``workflow``: Complete end-to-end analysis
- ``smi2xyz``, ``xtb-opt``, ``xyz2gaussian``, ``run-gaussian``: Individual workflow steps
- ``multiwfn``, ``extract-features``: Analysis and feature extraction
- ``ml-analysis``, ``shap-viz``: Machine learning and visualization

**Utility Commands:**
- ``mol-draw``, ``mol-info``: Molecular analysis tools
- ``rerun-gaussian``: Error recovery utility

**Workflow Integration:**

All individual step commands are designed to work together seamlessly:

1. **Input Validation**: Each command validates required inputs
2. **Output Compatibility**: Commands produce standardized outputs
3. **Error Handling**: Graceful failure with informative messages
4. **Progress Reporting**: Detailed status updates
5. **File Management**: Automatic cleanup and organization

**Best Practices:**

**For New Users:**
- Start with the complete ``workflow`` command
- Use ``--resume`` flag for long-running calculations
- Monitor progress through console output

**For Advanced Users:**
- Use individual commands for granular control
- Customize parameters for specific research needs
- Leverage utility tools for debugging and optimization

**For Production Use:**
- Implement proper error handling in scripts
- Use ``--resume`` for robust pipeline execution
- Monitor resource usage and optimize accordingly

**Resource Requirements:**

**Memory and CPU:**
- ``workflow``: 32+ cores, 30+ GB RAM recommended
- ``ml-analysis``: 16+ cores, 16+ GB RAM
- ``shap-viz``: 8+ cores, 8+ GB RAM
- ``gaussian`` steps: Highly dependent on molecule size

**Disk Space:**
- Small molecules (<50 atoms): ~100 MB per molecule
- Medium molecules (50-200 atoms): ~1-5 GB per molecule  
- Large molecules (>200 atoms): 10+ GB per molecule

**Time Estimates:**
- Small molecules: Minutes to hours
- Medium molecules: Hours to days
- Large molecules: Days to weeks

**Common Workflows:**

**Standard Research Pipeline:**

.. code-block:: bash

   # Complete analysis with test set
   surfacia workflow -i dataset.csv \
       --test-samples "1,5,10,15,20" \
       --max-features 8 \
       --stepreg-runs 5

**Development and Testing:**

.. code-block:: bash

   # Quick test with minimal settings
   surfacia workflow -i small_dataset.csv \
       --skip-xtb \
       --epoch 32 \
       --max-features 3

**Production Analysis:**

.. code-block:: bash

   # Full production pipeline with custom settings
   surfacia workflow -i production_dataset.csv \
       --resume \
       --keywords "# B3LYP/6-311g** opt freq" \
       --nproc 64 \
       --memory "100GB" \
       --max-features 10 \
       --stepreg-runs 10 \
       --epoch 256 \
       --cores 64

**Troubleshooting Guide:**

**Common Issues and Solutions:**

1. **SMILES Conversion Failures**
   - Check CSV format and column names
   - Validate SMILES strings using external tools
   - Ensure OpenBabel is properly installed

2. **Gaussian Calculation Failures**
   - Use ``rerun-gaussian`` to recover from failures
   - Check Gaussian installation and license
   - Verify computational resources (memory, disk space)

3. **Multiwfn Analysis Errors**
   - Ensure .fchk files are complete and non-empty
   - Check Multiwfn installation and path
   - Verify file permissions

4. **Machine Learning Issues**
   - Check for NaN values in feature files
   - Ensure sufficient training samples (>20 recommended)
   - Validate test sample names

5. **SHAP Visualization Problems**
   - Verify Training_Set_Detailed.csv exists
   - Check XYZ files are present and accessible
   - Ensure required Python packages are installed

**Getting Help:**

**Documentation Resources:**
- This command reference: Complete parameter documentation
- API documentation: Detailed function references
- Tutorials: Step-by-step guides for common tasks

**Community Support:**
- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and community support
- Documentation: Updates and contributions

**Professional Support:**
- Email: 823808458@qq.com
- GitHub: https://github.com/sym823808458/Surfacia

**Command Line Tips:**

**Productivity Enhancements:**

1. **Tab Completion**: Most shells support tab completion for commands and parameters
2. **Command History**: Use shell history to recall complex commands
3. **Scripting**: Include commands in shell scripts for automation
4. **Redirection**: Save output to files for later analysis

.. code-block:: bash

   # Save complete workflow output
   surfacia workflow -i data.csv 2>&1 | tee workflow.log

   # Run analysis in background
   nohup surfacia workflow -i data.csv > workflow.out 2>&1 &

**Parameter Management:**

1. **Configuration Files**: Create parameter files for complex analyses
2. **Environment Variables**: Set common parameters as environment variables
3. **Shell Aliases**: Create shortcuts for frequently used command combinations

.. code-block:: bash

   # Example shell aliases
   alias surf-quick='surfacia workflow --skip-xtb --epoch 32'
   alias surf-full='surfacia workflow --max-features 10 --stepreg-runs 5'

**Batch Processing:**

1. **Multiple Datasets**: Process multiple datasets sequentially
2. **Parameter Sweeps**: Test different parameter combinations
3. **Parallel Processing**: Use different screen sessions for parallel workflows

.. code-block:: bash

   # Process multiple datasets
   for dataset in data1.csv data2.csv data3.csv; do
       surfacia workflow -i $dataset --max-features 5
   done

This comprehensive command reference provides all the information needed to effectively use Surfacia's command-line interface for molecular surface analysis and machine learning.
