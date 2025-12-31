Descriptors and Features Module
==============================

The descriptors and features module provides comprehensive molecular descriptor extraction capabilities, including atomic properties, surface analysis, LOFFI algorithm, and fragment-based feature extraction.

Main Functions
-------------

Atomic Property Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: surfacia.features.atom_properties.run_atom_prop_extraction

LOFFI Algorithm
~~~~~~~~~~~~~~~

.. autofunction:: surfacia.features.loffi.apply_loffi_algorithm

Fragment Matching
~~~~~~~~~~~~~~~~~

.. autofunction:: surfacia.features.fragment_match.setup_logging
.. autofunction:: surfacia.features.fragment_match.read_xyz
.. autofunction:: surfacia.features.fragment_match.find_substructure
.. autofunction:: surfacia.features.fragment_match.sort_substructure_atoms

Usage Examples
-------------

Mode 1: Element-Specific Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract features for specific atomic elements (e.g., sulfur atoms):

.. code-block:: python

    from surfacia.features import run_atom_prop_extraction
    
    # Extract sulfur-specific features
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=1,
        target_element="S",
        threshold=0.001
    )
    
    # Extract nitrogen-specific features
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=1,
        target_element="N",
        threshold=0.002
    )

Mode 2: Fragment-Based Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract features based on molecular fragments:

.. code-block:: python

    from surfacia.features import run_atom_prop_extraction
    
    # Extract features for specific fragment
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=2,
        xyz1_path="fragment.xyz",
        threshold=0.001
    )

Mode 3: LOFFI Comprehensive Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply LOFFI algorithm for comprehensive surface analysis:

.. code-block:: python

    from surfacia.features import run_atom_prop_extraction
    from surfacia.features.loffi import apply_loffi_algorithm
    
    # Run comprehensive LOFFI analysis
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=3
    )
    
    # Apply LOFFI algorithm directly
    loffi_results = apply_loffi_algorithm(data_file="FullOption2.csv")

Fragment Analysis and Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from surfacia.features.fragment_match import (
        setup_logging, read_xyz, find_substructure, sort_substructure_atoms
    )
    
    # Setup logging for detailed output
    logger = setup_logging()
    
    # Read molecular structure
    atoms, coords = read_xyz("molecule.xyz")
    
    # Find specific substructure
    fragment_atoms = find_substructure(atoms, coords, target_atoms=["S", "O"])
    
    # Sort fragment atoms by distance
    sorted_atoms = sort_substructure_atoms(fragment_atoms, coords)

Advanced Usage
-------------

Custom Threshold Adjustment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adjust surface analysis thresholds for different sensitivity levels:

.. code-block:: python

    from surfacia.features import run_atom_prop_extraction
    
    # High sensitivity (detect more surface features)
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=1,
        target_element="O",
        threshold=0.0005  # Lower threshold = higher sensitivity
    )
    
    # Low sensitivity (detect only major features)
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=1,
        target_element="O",
        threshold=0.005  # Higher threshold = lower sensitivity
    )

Multi-Element Analysis
~~~~~~~~~~~~~~~~~~~~~~

Analyze multiple elements in sequence:

.. code-block:: python

    from surfacia.features import run_atom_prop_extraction
    
    # Analyze multiple elements
    elements = ["S", "N", "O"]
    base_file = "FullOption2.csv"
    
    for element in elements:
        print(f"Analyzing {element} atoms...")
        run_atom_prop_extraction(
            input_file=base_file,
            mode=1,
            target_element=element,
            threshold=0.001
        )

LOFFI Algorithm Details
-----------------------

The LOFFI (Local Orbital-Free Functional Indicator) algorithm provides comprehensive surface analysis:

**Key Features:**

* **Surface Electron Density Analysis**: Analyzes electron density distribution
* **Local Property Mapping**: Maps various electronic properties onto molecular surface
* **Multi-Scale Analysis**: Analyzes properties at different resolution levels
* **Integration with ML**: Optimized for machine learning feature extraction

**Supported Properties:**

.. list-table:: LOFFI Surface Properties
   :header-rows: 1

   * - Property
     - Description
     - Application
   * - **ALIE** (Average Local Ionization Energy)
     - Average energy required to remove electrons
     - Reactivity prediction
   * - **LEAE** (Local Electron Affinity Energy)
     - Local electron affinity
     - Nucleophilic site identification
   * - **ESP** (Electrostatic Potential)
     - Electrostatic potential at surface
     - Intermolecular interactions
   * - **VDW** (van der Waals Surface)
     - Van der Waals molecular surface
     - Steric properties
   * - **MEP** (Molecular Electrostatic Potential)
     - 3D electrostatic potential
     - Binding site analysis

Feature Extraction Modes
------------------------

Mode 1: Element-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Focuses on specific atomic elements within molecules:

**Use Cases:**
* Catalysis research (e.g., transition metal analysis)
* Drug design (e.g., sulfur-containing compounds)
* Material science (e.g., nitrogen-doped materials)

**Output Features:**
* Element-specific surface properties
* Local environment descriptors
* Electronic structure metrics

**Example Output:**
```
S_VDW_surface_area    S_ALIE_min    S_ALIE_max    S_ALIE_avg
125.432              8.234          12.456        10.345
S_ESP_min            S_ESP_max      S_ESP_avg     S_surface_charge
-0.234               0.456          0.111         -0.023
```

Mode 2: Fragment-Based Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes specific molecular fragments or functional groups:

**Use Cases:**
* Functional group analysis
* Pharmacophore identification
* Structure-activity relationships

**Fragment Matching Process:**

1. **Fragment Definition**: Define target fragment structure
2. **Substructure Search**: Find matches in molecules
3. **Property Mapping**: Map surface properties onto fragments
4. **Feature Extraction**: Extract fragment-specific descriptors

**Example Applications:**
```python
# Analyze benzene ring fragments
fragment_xyz = "benzene_fragment.xyz"
run_atom_prop_extraction(
    input_file="FullOption2.csv",
    mode=2,
    xyz1_path=fragment_xyz,
    threshold=0.001
)
```

Mode 3: LOFFI Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete molecular surface analysis using LOFFI algorithm:

**Comprehensive Feature Set:**
* Full surface electronic properties
* Multi-resolution analysis
* Integrated descriptor calculation
* ML-optimized feature vectors

**Output Structure:**
```
Sample_Name,ALIE_min,ALIE_max,ALIE_avg,LEAE_min,LEAE_max,LEAE_avg,
ESP_min,ESP_max,ESP_avg,VDW_area,MEP_mean,MEP_std,...
molecule_1,8.123,12.456,10.289,1.234,3.456,2.345,-0.234,0.456,0.111,
125.432,0.123,0.045,...
```

Integration with CLI
--------------------

Feature extraction is available through command line:

.. code-block:: bash

    # Mode 1: Element-specific extraction
    surfacia extract-features -i FullOption2.csv --mode 1 --element S --threshold 0.001
    
    # Mode 2: Fragment-based extraction
    surfacia extract-features -i FullOption2.csv --mode 2 --xyz1 fragment.xyz
    
    # Mode 3: LOFFI comprehensive analysis
    surfacia extract-features -i FullOption2.csv --mode 3

Data Requirements
-----------------

Input File Format
~~~~~~~~~~~~~~~~~

The feature extraction expects Multiwfn output files processed into CSV format:

**Required Columns:**
* ``Sample_Name``: Unique molecular identifier
* ``Atom_Number``: Atomic index in the molecule
* ``X, Y, Z``: Atomic coordinates
* Surface property columns (from Multiwfn analysis)

**Example Input:**
```csv
Sample_Name,Atom_Number,Element,X,Y,Z,Electron_Density,ESP,ALIE
mol1_S1,1,S,0.123,1.456,2.789,0.0234,0.123,8.456
mol1_S1,2,O,1.234,2.567,3.890,0.0345,-0.234,9.567
...
```

Output File Format
~~~~~~~~~~~~~~~~~~

Feature extraction generates enhanced CSV files with new descriptor columns:

**Mode 1 Output (Element-Specific):**
```csv
Sample_Name,S_VDW_area,S_ALIE_min,S_ALIE_max,S_ALIE_avg,S_ESP_min,S_ESP_max,...
mol1_S1,125.432,8.234,12.456,10.345,-0.234,0.456,...
```

**Mode 2 Output (Fragment-Based):**
```csv
Sample_Name,Fragment_VDW_area,Fragment_ALIE_avg,Fragment_ESP_range,...
mol1_frag1,45.123,9.234,0.567,...
```

**Mode 3 Output (LOFFI Comprehensive):**
```csv
Sample_Name,ALIE_min,ALIE_max,ALIE_avg,LEAE_min,LEAE_max,LEAE_avg,
ESP_min,ESP_max,ESP_avg,VDW_area,MEP_mean,...
mol1,8.123,12.456,10.289,1.234,3.456,2.345,-0.234,0.456,0.111,
125.432,0.123,...
```

Performance Considerations
--------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~

**Mode Performance Ranking (Fastest to Slowest):**

1. **Mode 1 (Element-Specific)**: Fast, focused analysis
2. **Mode 2 (Fragment-Based)**: Medium, requires substructure matching
3. **Mode 3 (LOFFI Comprehensive)**: Slow, full surface analysis

**Memory Requirements:**

* **Small molecules** (<50 atoms): <100 MB RAM
* **Medium molecules** (50-200 atoms): 100-500 MB RAM  
* **Large molecules** (>200 atoms): 500 MB+ RAM

**Processing Time Estimates:**

* **Mode 1**: ~1-5 seconds per molecule
* **Mode 2**: ~5-15 seconds per molecule
* **Mode 3**: ~15-60 seconds per molecule

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~

**For Large Datasets:**

1. **Batch Processing**: Process molecules in groups
2. **Threshold Optimization**: Use appropriate threshold values
3. **Parallel Processing**: Utilize multiple CPU cores
4. **Memory Management**: Monitor and manage RAM usage

**For Quality Results:**

1. **Threshold Selection**: Test different threshold values
2. **Element Choice**: Focus on relevant elements only
3. **Fragment Validation**: Ensure fragment structures are correct
4. **Output Validation**: Check extracted feature ranges

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

* **Surface Quality**: Ensure high-quality Multiwfn calculations
* **File Organization**: Keep input files well-organized
* **Naming Conventions**: Use consistent sample naming
* **Quality Control**: Validate input data before extraction

Feature Selection
~~~~~~~~~~~~~~~~~

* **Domain Knowledge**: Use chemical intuition for element/fragment choice
* **Threshold Tuning**: Optimize thresholds for specific applications
* **Validation**: Cross-validate extracted features with known properties
* **Documentation**: Document extraction parameters for reproducibility

Integration with ML Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from surfacia.features import run_atom_prop_extraction
    from surfacia.ml import ChemMLWorkflow
    
    # Step 1: Extract features
    run_atom_prop_extraction(
        input_file="FullOption2.csv",
        mode=3  # Comprehensive analysis
    )
    
    # Step 2: Run ML analysis on extracted features
    results = ChemMLWorkflow.run_analysis(
        mode='workflow',
        data_file='FinalFull.csv',  # Output from feature extraction
        max_features=8,
        n_runs=5
    )
    
    print(f"Best MSE: {results['final']['mse']:.4f}")
    print(f"Selected features: {results['final']['selected_features']}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**"No matching atoms found"**

* **Cause**: Specified element not present in molecules
* **Solution**: Check element symbols and molecular composition

**"Fragment matching failed"**

* **Cause**: Fragment structure incorrect or not present
* **Solution**: Validate fragment XYZ structure and connectivity

**"Memory overflow during LOFFI analysis"**

* **Cause**: Molecules too large or memory insufficient
* **Solution**: Reduce dataset size or increase available RAM

**"Threshold too high/low"**

* **Cause**: Inappropriate threshold for molecular system
* **Solution**: Experiment with different threshold values

Quality Control
~~~~~~~~~~~~~~~

**Feature Range Validation:**
```python
# Check reasonable feature ranges
def validate_features(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Check ALIE values (should be 5-15 eV for typical molecules)
    alie_cols = [col for col in df.columns if 'ALIE' in col]
    for col in alie_cols:
        if df[col].min() < 5 or df[col].max() > 15:
            print(f"Warning: {col} has unusual range: {df[col].min():.2f}-{df[col].max():.2f}")
```

**Consistency Checks:**
* Verify consistent sample naming across files
* Check for duplicate entries
* Validate coordinate ranges (should be reasonable molecular dimensions)
* Ensure feature correlations make chemical sense

Notes
-----

* Feature extraction requires completed Multiwfn calculations
* LOFFI algorithm is optimized for organic molecules
* Surface properties are sensitive to calculation quality
* Threshold values significantly impact extracted features
* Results depend on the quality of input quantum chemical calculations
* Different modes provide complementary information for comprehensive analysis
