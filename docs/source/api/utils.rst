Utility Modules
================

This section documents the utility modules of Surfacia that provide molecular information analysis, file processing, and helper functions for common tasks.

.. module:: surfacia.utils.mol_info

Molecular Information
--------------------

.. automodule:: surfacia.utils.mol_info
   :members:
   :undoc-members:
   :show-inheritance:

The molecular information module provides functions for calculating molecular properties and descriptors from SMILES strings.

Functions
~~~~~~~~~

.. autofunction:: surfacia.utils.mol_info.calculate_molecular_properties

.. autofunction:: surfacia.utils.mol_info.calculate_property

.. autofunction:: surfacia.utils.mol_info.mol_info_main

Constants
~~~~~~~~~

.. data:: AVAILABLE_PROPERTIES

   List of available molecular properties that can be calculated:
   
   - ``mw``: Molecular weight
   - ``logp``: Partition coefficient (octanol/water)
   - ``hbd``: Hydrogen bond donors
   - ``hba``: Hydrogen bond acceptors
   - ``tpsa``: Topological polar surface area
   - ``rotatable_bonds``: Number of rotatable bonds
   - ``aromatic_rings``: Number of aromatic rings
   - ``heavy_atoms``: Number of heavy atoms

.. data:: DEFAULT_PROPERTIES

   Default set of properties calculated when no specific properties are requested:
   ``['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds']``

Examples
~~~~~~~~

**Calculate single molecule properties:**

.. code-block:: python

   from surfacia.utils.mol_info import calculate_molecular_properties
   
   # Calculate all default properties for ethanol
   properties = calculate_molecular_properties("CCO")
   
   # Properties returns a dictionary
   print(f"Molecular weight: {properties['mw']}")
   print(f"LogP: {properties['logp']}")

**Calculate specific properties:**

.. code-block:: python

   from surfacia.utils.mol_info import calculate_molecular_properties
   
   # Calculate only specific properties
   properties = calculate_molecular_properties(
       "CCO",
       properties=['mw', 'hbd', 'hba', 'tpsa']
   )

**Process multiple molecules:**

.. code-block:: python

   from surfacia.utils.mol_info import mol_info_main
   
   # Process all molecules from CSV file
   mol_info_main(
       input_file="molecules.csv",
       output_file="molecular_properties.csv",
       properties=['mw', 'logp', 'hbd', 'hba', 'tpsa', 
                  'rotatable_bonds', 'aromatic_rings', 'heavy_atoms']
   )

**Using main function with single SMILES:**

.. code-block:: python

   from surfacia.utils.mol_info import mol_info_main
   
   # Calculate properties for a single molecule
   mol_info_main(
       smiles="CCO",
       properties=['mw', 'logp']
   )

**Batch processing:**

.. code-block:: python

   import pandas as pd
   from surfacia.utils.mol_info import calculate_molecular_properties
   
   # Read SMILES from CSV
   df = pd.read_csv("molecules.csv")
   
   # Calculate properties for each molecule
   properties_list = []
   for smiles in df['smiles']:
       props = calculate_molecular_properties(smiles)
       properties_list.append(props)
   
   # Create DataFrame with properties
   props_df = pd.DataFrame(properties_list)
   result_df = pd.concat([df, props_df], axis=1)
   result_df.to_csv("molecules_with_properties.csv", index=False)

Property Descriptions
~~~~~~~~~~~~~~~~~~~~~

**Molecular Weight (mw):**
- Total mass of all atoms in the molecule
- Units: g/mol (atomic mass units)
- Important for reactivity and solubility

**Partition Coefficient (logP):**
- Measure of hydrophobicity/lipophilicity
- Logarithm of octanol/water partition ratio
- Indicates membrane permeability
- Typical range: -2 (very hydrophilic) to 6 (very hydrophobic)

**Hydrogen Bond Donors (hbd):**
- Number of hydrogen atoms that can donate hydrogen bonds
- Affects solubility and binding
- Usually O-H and N-H groups

**Hydrogen Bond Acceptors (hba):**
- Number of atoms that can accept hydrogen bonds
- Affects solubility and binding
- Usually O, N, and S atoms

**Topological Polar Surface Area (tpsa):**
- Surface area of polar atoms
- Units: Å² (square angstroms)
- Predicts drug transport properties
- Important for bioavailability

**Rotatable Bonds:**
- Number of bonds that can rotate freely
- Affects molecular flexibility
- Influences binding and bioavailability

**Aromatic Rings:**
- Number of aromatic ring systems
- Affects electronic properties
- Influences π-π stacking interactions

**Heavy Atoms:**
- Number of atoms other than hydrogen
- Basic molecular size measure
- Affects pharmacokinetics

Error Handling
~~~~~~~~~~~~~~~

The module provides robust error handling:

.. code-block:: python

   from surfacia.utils.mol_info import calculate_molecular_properties
   
   # Handle invalid SMILES
   try:
       props = calculate_molecular_properties("INVALID_SMILES")
   except Exception as e:
       print(f"Error: {e}")
   
   # Result: "Error: Invalid SMILES string"

Integration with Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

Molecular information can be integrated with the main workflow:

.. code-block:: python

   import pandas as pd
   from surfacia.utils.mol_info import calculate_molecular_properties
   from surfacia.core.workflow import SurfaciaWorkflow
   
   # Add molecular properties to dataset
   df = pd.read_csv("molecules.csv")
   
   # Calculate properties
   props_list = []
   for smiles in df['smiles']:
       props = calculate_molecular_properties(smiles)
       props_list.append(props)
   
   # Add properties to dataframe
   props_df = pd.DataFrame(props_list)
   df = pd.concat([df, props_df], axis=1)
   
   # Save enhanced dataset
   df.to_csv("molecules_enhanced.csv", index=False)
   
   # Run workflow with enhanced dataset
   workflow = SurfaciaWorkflow("molecules_enhanced.csv")
   workflow.run_full_workflow()

.. module:: surfacia.utils.extractor

Feature Extractor
----------------

.. automodule:: surfacia.utils.extractor
   :members:
   :undoc-members:
   :show-inheritance:

The feature extractor module provides functions for extracting molecular surface features from Multiwfn analysis results using different extraction modes.

Functions
~~~~~~~~~

.. autofunction:: surfacia.utils.extractor.extract_features_main

.. autofunction:: surfacia.utils.extractor.element_specific_extraction

.. autofunction:: surfacia.utils.extractor.fragment_specific_extraction

.. autofunction:: surfacia.utils.extractor.loffi_comprehensive_extraction

.. autofunction:: surfacia.utils.extractor.process_extraction

Examples
~~~~~~~~

**Mode 1 - Element-Specific Extraction:**

.. code-block:: python

   from surfacia.utils.extractor import extract_features_main
   
   # Extract features around sulfur atoms
   extract_features_main(
       input_file="FullOption.csv",
       mode=1,
       element="S",
       output_file="FinalFull_S_features.csv"
   )

**Mode 2 - Fragment-Specific Extraction:**

.. code-block:: python

   from surfacia.utils.extractor import extract_features_main
   
   # Extract features around a specific fragment
   extract_features_main(
       input_file="FullOption.csv",
       mode=2,
       xyz_file="fragment.xyz",
       threshold=0.002,
       output_file="FinalFull_fragment_features.csv"
   )

**Mode 3 - LOFFI Comprehensive Extraction:**

.. code-block:: python

   from surfacia.utils.extractor import extract_features_main
   
   # Comprehensive surface analysis (default mode)
   extract_features_main(
       input_file="FullOption.csv",
       mode=3,
       output_file="FinalFull.csv"
   )

**Custom threshold for fragment analysis:**

.. code-block:: python

   from surfacia.utils.extractor import extract_features_main
   
   # Tighter threshold for more precise fragment analysis
   extract_features_main(
       input_file="FullOption.csv",
       mode=2,
       xyz_file="fragment.xyz",
       threshold=0.001,
       output_file="FinalFull_precise.csv"
   )

Extraction Modes
~~~~~~~~~~~~~~~

**Mode 1: Element-Specific Analysis**
- Focuses on surface properties around a specific element
- Requires specification of target element (e.g., S, N, O, C)
- Useful for studying reactivity of specific atomic sites
- Extracts local electronic environment descriptors

**Mode 2: Fragment-Specific Analysis**
- Analyzes properties around a user-defined molecular fragment
- Requires fragment XYZ file
- Useful for studying functional groups or substructures
- Threshold parameter defines analysis region size

**Mode 3: LOFFI Comprehensive Analysis**
- Analyzes entire molecular surface
- Most commonly used mode
- Provides complete surface descriptor set
- No additional parameters required

Output Format
~~~~~~~~~~~~~

All extraction modes produce a structured CSV file with:

- Sample names
- Target values
- Extracted surface descriptors
- Statistical summaries (mean, min, max, std)
- Element/fragment specific features (depending on mode)

Best Practices
~~~~~~~~~~~~~~

**1. Choose appropriate mode:**

- **Mode 1** for studying specific reactive sites
- **Mode 2** for functional group analysis
- **Mode 3** for comprehensive analysis (most common)

**2. Optimize threshold for fragment analysis:**

- Larger threshold (0.002-0.003): Broader analysis region
- Smaller threshold (0.001-0.0015): More precise, localized analysis

**3. Validate extraction results:**

.. code-block:: python

   import pandas as pd
   
   # Load extracted features
   df = pd.read_csv("FinalFull.csv")
   
   # Check for missing values
   missing_values = df.isnull().sum()
   
   # Basic statistics
   print(df.describe())

**4. Combine extraction modes:**

For comprehensive analysis, run multiple modes:

.. code-block:: python

   from surfacia.utils.extractor import extract_features_main
   
   # Run all three modes
   extract_features_main("FullOption.csv", mode=1, element="S",
                       output_file="Mode1_S.csv")
   extract_features_main("FullOption.csv", mode=2, xyz_file="fragment.xyz",
                       output_file="Mode2_fragment.csv")
   extract_features_main("FullOption.csv", mode=3,
                       output_file="Mode3_comprehensive.csv")

Integration with ML Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracted features are used as input for machine learning:

.. code-block:: python

   from surfacia.utils.extractor import extract_features_main
   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   # Step 1: Extract features
   extract_features_main(
       input_file="FullOption.csv",
       mode=3,
       output_file="FinalFull.csv"
   )
   
   # Step 2: Run ML analysis
   analyzer = WorkflowAnalyzer(data_file="FinalFull.csv")
   analyzer.run(max_features=5, n_runs=3)

File Processing Utilities
==========================

.. module:: surfacia.utils.file_utils

File Utilities
---------------

.. automodule:: surfacia.utils.file_utils
   :members:
   :undoc-members:
   :show-inheritance:

The file utilities module provides helper functions for file I/O operations, directory management, and data processing.

Functions
~~~~~~~~~

.. autofunction:: surfacia.utils.file_utils.ensure_directory

.. autofunction:: surfacia.utils.file_utils.find_files_by_pattern

.. autofunction:: surfacia.utils.file_utils.read_file_list

.. autofunction:: surfacia.utils.file_utils.write_file_list

.. autofunction:: surfacia.utils.file_utils.get_timestamp

Examples
~~~~~~~~

**Ensure directory exists:**

.. code-block:: python

   from surfacia.utils.file_utils import ensure_directory
   
   # Create directory if it doesn't exist
   output_dir = ensure_directory("./results/analysis")
   
   # Safe to write files now
   with open(f"{output_dir}/results.txt", 'w') as f:
       f.write("Analysis results")

**Find files by pattern:**

.. code-block:: python

   from surfacia.utils.file_utils import find_files_by_pattern
   
   # Find all CSV files in directory
   csv_files = find_files_by_pattern("./data", "*.csv")
   
   # Find all XYZ files
   xyz_files = find_files_by_pattern("./calculations", "*.xyz")
   
   # Find files with specific prefix
   data_files = find_files_by_pattern("./", "data_*")

**Get timestamp for file naming:**

.. code-block:: python

   from surfacia.utils.file_utils import get_timestamp
   
   # Get formatted timestamp
   timestamp = get_timestamp()
   # Example output: "20241215_143022"
   
   # Use for creating unique filenames
   output_file = f"analysis_results_{timestamp}.csv"

Data Processing Helpers
======================

.. module:: surfacia.utils.data_utils

Data Utilities
---------------

.. automodule:: surfacia.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

The data utilities module provides functions for data manipulation, validation, and preprocessing.

Functions
~~~~~~~~~

.. autofunction:: surfacia.utils.data_utils.validate_dataframe

.. autofunction:: surfacia.utils.data_utils.clean_dataframe

.. autofunction:: surfacia.utils.data_utils.merge_dataframes

.. autofunction:: surfacia.utils.data_utils.calculate_statistics

Examples
~~~~~~~~

**Validate DataFrame structure:**

.. code-block:: python

   import pandas as pd
   from surfacia.utils.data_utils import validate_dataframe
   
   df = pd.read_csv("data.csv")
   
   # Validate required columns exist
   is_valid = validate_dataframe(
       df,
       required_columns=['smiles', 'target'],
       numeric_columns=['target']
   )
   
   if not is_valid:
       print("DataFrame validation failed")

**Clean DataFrame:**

.. code-block:: python

   from surfacia.utils.data_utils import clean_dataframe
   
   # Remove duplicates and handle missing values
   cleaned_df = clean_dataframe(
       df,
       drop_duplicates=True,
       fill_na_method='mean'  # or 'median', 'mode', 'drop'
   )

**Calculate statistics:**

.. code-block:: python

   from surfacia.utils.data_utils import calculate_statistics
   
   # Get comprehensive statistics
   stats = calculate_statistics(df, columns=['feature1', 'feature2'])
   
   # Returns: mean, std, min, max, quartiles
   print(f"Mean: {stats['mean']}")
   print(f"Std: {stats['std']}")

Integration Examples
====================

**Complete workflow with utilities:**

.. code-block:: python

   import pandas as pd
   from surfacia.utils.file_utils import ensure_directory, get_timestamp
   from surfacia.utils.mol_info import calculate_molecular_properties
   from surfacia.utils.data_utils import clean_dataframe, calculate_statistics
   
   # Step 1: Load and clean data
   df = pd.read_csv("molecules.csv")
   df = clean_dataframe(df, drop_duplicates=True)
   
   # Step 2: Calculate molecular properties
   props_list = []
   for smiles in df['smiles']:
       props = calculate_molecular_properties(smiles)
       props_list.append(props)
   props_df = pd.DataFrame(props_list)
   df = pd.concat([df, props_df], axis=1)
   
   # Step 3: Calculate statistics
   stats = calculate_statistics(df, columns=df.columns[2:])
   
   # Step 4: Save results with timestamp
   timestamp = get_timestamp()
   output_dir = ensure_directory(f"./results_{timestamp}")
   df.to_csv(f"{output_dir}/enhanced_data.csv", index=False)
   stats.to_csv(f"{output_dir}/statistics.csv")

**Batch processing with file utilities:**

.. code-block:: python

   from surfacia.utils.file_utils import find_files_by_pattern, ensure_directory
   from surfacia.utils.extractor import extract_features_main
   
   # Find all Multiwfn output files
   output_files = find_files_by_pattern("./multiwfn_output", "*.txt")
   
   # Process each file
   results_dir = ensure_directory("./extraction_results")
   
   for output_file in output_files:
       # Extract features
       extract_features_main(
           input_file=output_file,
           mode=3,
           output_file=f"{results_dir}/{output_file.stem}_features.csv"
       )

**Error handling pattern:**

.. code-block:: python

   import pandas as pd
   from surfacia.utils.data_utils import validate_dataframe
   from surfacia.utils.mol_info import calculate_molecular_properties
   
   def process_molecules(input_csv, output_csv):
       """Process molecules with error handling."""
       
       # Load data
       df = pd.read_csv(input_csv)
       
       # Validate structure
       if not validate_dataframe(df, required_columns=['smiles', 'target']):
           raise ValueError("Invalid CSV structure")
       
       # Calculate properties
       properties = []
       for idx, row in df.iterrows():
           try:
               props = calculate_molecular_properties(row['smiles'])
               properties.append(props)
           except Exception as e:
               print(f"Error processing molecule {idx}: {e}")
               properties.append({})  # Empty dict for failed molecules
       
       # Add properties to dataframe
       props_df = pd.DataFrame(properties)
       result_df = pd.concat([df, props_df], axis=1)
       
       # Save results
       result_df.to_csv(output_csv, index=False)
       return result_df

Best Practices for Utilities
=============================

**1. Always validate input data:**

.. code-block:: python

   from surfacia.utils.data_utils import validate_dataframe
   
   df = pd.read_csv("input.csv")
   
   # Validate before processing
   if not validate_dataframe(df, required_columns=['smiles', 'target']):
       raise ValueError("Invalid input data")

**2. Use timestamps for output files:**

.. code-block:: python

   from surfacia.utils.file_utils import get_timestamp
   
   timestamp = get_timestamp()
   output_file = f"results_{timestamp}.csv"

**3. Ensure directories exist:**

.. code-block:: python

   from surfacia.utils.file_utils import ensure_directory
   
   # Always ensure output directory exists
   output_dir = ensure_directory("./results")
   # Safe to write files now

**4. Handle errors gracefully:**

.. code-block:: python

   from surfacia.utils.mol_info import calculate_molecular_properties
   
   for smiles in smiles_list:
       try:
           props = calculate_molecular_properties(smiles)
           # Process properties
       except Exception as e:
           print(f"Error for {smiles}: {e}")
           # Continue with next molecule

**5. Clean data before analysis:**

.. code-block:: python

   from surfacia.utils.data_utils import clean_dataframe
   
   # Remove duplicates and handle missing values
   df = clean_dataframe(
       df,
       drop_duplicates=True,
       fill_na_method='median'
   )

**6. Document your processing:**

.. code-block:: python

   # Add metadata to results
   from surfacia.utils.file_utils import get_timestamp
   
   metadata = {
       'timestamp': get_timestamp(),
       'processing_steps': [
           'cleaned_data',
           'calculated_properties',
           'extracted_features'
       ],
       'parameters': {
           'mode': 3,
           'threshold': 0.002
       }
   }

**7. Use consistent naming conventions:**

.. code-block:: text

   Input:  molecules.csv
   Output: FinalFull_YYYYMMDD_HHMMSS.csv
   Logs:   workflow_log_YYYYMMDD_HHMMSS.txt
   Plots:  prediction_scatter_YYYYMMDD_HHMMSS.png

**8. Verify file paths:**

.. code-block:: python

   from pathlib import Path
   
   # Always verify files exist
   if not Path("input.csv").exists():
       raise FileNotFoundError("Input file not found")

**9. Log processing steps:**

.. code-block:: python

   import logging
   
   logging.basicConfig(filename='processing.log', level=logging.INFO)
   
   logging.info("Starting data processing")
   # Processing steps
   logging.info("Completed data processing")

**10. Test with small datasets first:**

.. code-block:: python

   # Test with 10 molecules
   test_df = df.head(10)
   
   # Process test data
   results = process_molecules(test_df, "test_results.csv")
   
   # Verify results before processing full dataset
   if results.shape[0] == 10:
       print("Test successful, processing full dataset...")
       full_results = process_molecules(df, "full_results.csv")

These utility modules provide essential helper functions for data processing, file management, and molecular property calculations that support the main Surfacia workflow.
