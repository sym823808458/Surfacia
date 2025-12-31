Multi-scale Quantitative Surface Analysis (MQSA) Modes
====================================================

A comprehensive understanding of molecular properties is achieved by analyzing them across different hierarchical levels, from individual elements and functional groups to the entire molecule. The MQSA module offers three distinct, scenario-driven modes.

Overview
--------

MQSA provides 63 total features across three complementary analysis modes:

1. **Mode 1: Element-Specific Analysis** (13 features)
2. **Mode 2: Fragment-Specific Analysis** (18 features)
3. **Mode 3: LOFFI Automated Analysis** (32 features)

Comparative Overview
--------------------

.. list-table:: MQSA Modes Comparison
   :widths: 25 25 25 25
   :header-rows: 1

   * - Mode 1: Element-Specific
     - Mode 2: Fragment-Specific
     - Mode 3: LOFFI Automated
   * - Quantifies collective properties of a single element type
     - Characterizes user-defined fragment as integrated unit
     - Performs unbiased comprehensive surface analysis
   * - "What is the role of this specific element?"
     - "For this core fragment, what are its properties?"
     - "What surface features drive properties without bias?"
   * - Identify all atoms of target_element
     - Identify all atoms of the fragment
     - Atom_: Global statistics on all surface atoms
     - Fun_: Statistics on auto-detected functional groups
   * - Quantifying an Element's Role
     - A Portrait of the Fragment
     - A) Surface Panorama B) Functional Modules
   * - Element-centric hypothesis testing
     - Fragment-centric quantification
     - Exploratory/Data-driven analysis
   * - Chemical symbol (e.g., 'S')
     - .xyz1 file defining fragment
     - No input required; fully automated
   * - Semi-automated, low flexibility
     - Semi-automated, high flexibility
     - Fully automated, universal flexibility
   * - 13 features
     - 18 features
     - 32 features

3.1 Mode 1: Element-Specific Analysis (13 Features)
------------------------------------------------------

This mode focuses on quantifying the collective properties of a specific element type within a molecule. It is particularly valuable for element-centric hypothesis testing.

**Mathematical Framework**

Let S_X be set of all atoms of target element X. The 13 descriptors are calculated as:

**X_area (Å²)**
   - **Definition**: Average surface area of target element atoms
   - **Formula**: 
     
     .. math::
     
        \mathrm{X\_area}=\frac{1}{\left|S_X\right|}\sum_{i \in S_X} \mathrm{Area}_i
   
   - **Application**: Indicates collective exposure and availability for interaction

**Property Statistics (12 Features)**

For each electronic property P (LEAE, ESP, ALIE in eV), four statistical descriptors are generated:

**X_P_min**
   - **Definition**: Minimum property value on any atom of element X
   - **Formula**: 
     
     .. math::
     
        \mathrm{X\_P\_min}=\min{\left(\left\{P_i \mid i \in S_X\right\}\right)}
   
   - **Application**: Identifies most extreme characteristic site

**X_P_max**
   - **Definition**: Maximum property value on any atom of element X
   - **Formula**: 
     
     .. math::
     
        \mathrm{X\_P\_max}=\max{\left(\left\{P_i \mid i \in S_X\right\}\right)}
   
   - **Application**: Identifies opposite extreme characteristic

**X_P_average**
   - **Definition**: Average property value across all atoms of element X
   - **Formula**: 
     
     .. math::
     
        \mathrm{X\_P\_average}=\frac{1}{\left|S_X\right|}\sum_{i \in S_X} P_i
   
   - **Application**: Overall electronic character of the element

**X_P_delta**
   - **Definition**: Range (max - min) of the property across all atoms
   - **Formula**: 
     
     .. math::
     
        \mathrm{X\_P\_delta}=\mathrm{X\_P\_max}-\mathrm{X\_P\_min}
   
   - **Application**: Quantifies heterogeneity among different atoms

**Feature List**

For element X, generates:
- X_area
- X_LEAE_min, X_LEAE_max, X_LEAE_average, X_LEAE_delta
- X_ESP_min, X_ESP_max, X_ESP_average, X_ESP_delta
- X_ALIE_min, X_ALIE_max, X_ALIE_average, X_ALIE_delta

3.2 Mode 2: Fragment-Specific Analysis (18 Features)
-----------------------------------------------------

This mode characterizes a user-defined chemical fragment (e.g., pharmacophore or reaction core) as a single, integrated unit.

**Naming Convention**

Fragment_[Property]_[Statistic]

- **Fragment_**: Prefix indicating descriptor calculated for user-defined fragment
- **[Property]**: LEAE, ESP, ALIE, or area
- **[Statistic]**: min, max, mean, or delta

**Statistical Descriptors (16 Features)**

**Fragment_LEAE_min/max/mean/delta**
   - **Definition**: LEAE statistics across fragment
   - **Application**: Electron-accepting capability extremes and average

**Fragment_ESP_min/max/mean/delta**
   - **Definition**: ESP statistics across fragment
   - **Application**: Electrostatic properties extremes and average

**Fragment_ALIE_min/max/mean/delta**
   - **Definition**: ALIE statistics across fragment
   - **Application**: Electron-donating capability extremes and average

**Fragment_area_min/max/mean/delta**
   - **Definition**: Surface area statistics across fragment
   - **Application**: Spatial extent variation within fragment

**Specialized Descriptors (2 Features)**

**Fragment_Count**
   - **Units**: unitless integer
   - **Definition**: Total number of fragment instances found in molecule
   - **Application**: Quantifies repetition or symmetry

**Fragment_Total_Area (Å²)**
   - **Definition**: Sum of surface areas of all identified fragment instances
   - **Application**: Total exposure of fragment type

3.3 Mode 3: LOFFI Automated Analysis (32 Features)
------------------------------------------------------

The Local-Functionality-Fragment-Integration (LOFFI) approach is a standardized, fully automated workflow for exploratory analysis. It provides a fixed 32-feature set from two complementary perspectives.

### Part A: Atomic-Level Perspective (Atom_* Descriptors, 16 Features)

This set describes the global statistical distribution of electronic properties across all atoms on the molecular van der Waals surface.

**Physical Meaning**

Defines the theoretical limits and overall background of the molecule's reactivity:
- Atom_ALIE_min = absolute most nucleophilic point anywhere on surface
- Atom_ESP_max = absolute most electrophilic site on molecule
- Atom_LEAE_delta = global heterogeneity in electron affinity

**Feature List (16 Features)**

For each property (LEAE, ESP, ALIE, area):
- Atom_P_min
- Atom_P_max
- Atom_P_mean
- Atom_P_delta

### Part B: Functional Group-Level Perspective (Fun_* Descriptors, 16 Features)

This set quantifies the relationships between the molecule's distinct chemical functional units.

**Physical Meaning**

Reveals interplay between functional "building blocks":
- Fun_ALIE_min = strongest electron donor functional group
- Fun_ESP_delta = polarity difference between most electron-rich/poor groups
- Fun_area_mean = average size of functional building blocks

**Calculation Algorithm**

1. **Automated Recognition**: Partition molecule into non-overlapping functional groups using priority-based system (Aromatic Systems > High-Priority Groups > General Patterns)

2. **Group Characterization**: For each identified group, calculate single, area-weighted property value representing the group as a whole

3. **Inter-Group Statistics**: Perform statistical analysis on the list of property values generated in step 2

**Feature List (16 Features)**

For each property (LEAE, ESP, ALIE, area):
- Fun_P_min
- Fun_P_max
- Fun_P_mean
- Fun_P_delta

**Interpretation Examples**

- Fun_ALIE_min: Which functional group, as an entire unit, is the strongest electron donor
- Fun_ESP_delta: Quantitative measure of intramolecular "push-pull" effect
- Fun_area_mean: Average size of functional modules constituting the molecule

Mode Selection Guide
--------------------

**When to Use Mode 1: Element-Specific**

✅ Studying contribution of specific element (e.g., sulfur in drug molecules)
✅ Element-centric hypotheses (e.g., "Does fluorine increase activity?")
✅ Comparing same element across different chemical environments

❌ Not suitable for: Fragment-specific or global molecular analysis

**When to Use Mode 2: Fragment-Specific**

✅ Quantifying pharmacophore properties
✅ Understanding environmental effects on core fragment
✅ Fragment-based SAR studies
✅ R-group analysis in combinatorial libraries

❌ Not suitable for: Exploratory analysis without predefined fragments

**When to Use Mode 3: LOFFI Automated**

✅ Exploratory analysis of diverse molecular libraries
✅ Finding QSAR drivers without pre-existing bias
✅ Data-driven feature discovery
✅ Standardized, reproducible analysis

❌ Not suitable for: Focused element or fragment-specific questions

Practical Examples
-----------------

**Example 1: Element-Specific Analysis**

.. code-block:: python

   # Analyze sulfur contribution in drug molecules
   from surfacia.features import atom_properties
   
   # Calculate Mode 1 descriptors for sulfur
   results = atom_properties.run_atom_prop_extraction(
       input_file="molecules.csv",
       mode=1,
       target_element="S"
   )
   
   # Output includes: S_area, S_LEAE_min, S_ESP_average, etc.

**Example 2: Fragment-Specific Analysis**

.. code-block:: python

   # Analyze benzene ring fragment
   from surfacia.features import fragment_match
   
   # Define fragment in .xyz1 format
   # Calculate Mode 2 descriptors
   results = fragment_match.apply_loffi_algorithm(
       input_file="molecules.csv",
       fragment_file="benzene_ring.xyz1"
   )
   
   # Output includes: Fragment_LEAE_min, Fragment_ESP_mean, Fragment_Count

**Example 3: LOFFI Automated Analysis**

.. code-block:: python

   # Fully automated exploratory analysis
   from surfacia.features import atom_properties
   
   # Calculate Mode 3 descriptors (default)
   results = atom_properties.run_atom_prop_extraction(
       input_file="molecules.csv",
       mode=3  # LOFFI mode
   )
   
   # Output includes: Atom_ALIE_min, Atom_ESP_delta,
   # Fun_ALIE_min, Fun_ESP_delta, etc. (32 features total)

Applications
------------

**Drug Discovery**
   - Mode 1: Identify key elements in drug design (e.g., halogens, heteroatoms)
   - Mode 2: Quantify pharmacophore contributions
   - Mode 3: Unbiased SAR analysis of screening libraries

**Materials Science**
   - Mode 1: Element-specific property engineering
   - Mode 2: Functional group effect quantification
   - Mode 3: Polymer property prediction

**Catalysis**
   - Mode 1: Active site element characterization
   - Mode 2: Catalyst fragment analysis
   - Mode 3: Discovery of new catalytic motifs

**Chemical Education**
   - Mode 1: Teaching periodic table concepts
   - Mode 2: Functional group chemistry
   - Mode 3: Introducing multi-scale analysis

References
----------

- **LOFFI Method**: Su et al. on Local-Functionality-Fragment-Integration
- **Surface Analysis**: Multiwfn documentation
- **Chemical Intuition**: Classic organic chemistry textbooks

See Also
--------

- :doc:`size_and_shape`: Geometric descriptor definitions
- :doc:`electronic_properties`: Electronic descriptor definitions
- :doc:`../api/descriptors`: API reference for descriptor extraction
