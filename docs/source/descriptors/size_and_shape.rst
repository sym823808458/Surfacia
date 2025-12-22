Size and Shape Descriptors
==========================

Geometric descriptors that characterize the size and shape properties of molecular surfaces.

Overview
--------

Size and shape descriptors provide fundamental geometric information about molecular surfaces, including volume, surface area, and various shape indices. These descriptors are essential for understanding molecular recognition, binding affinity, and physicochemical properties.

Available Descriptors
---------------------

Volume Descriptors
~~~~~~~~~~~~~~~~~~

**Molecular Volume**
  - Total volume enclosed by the molecular surface
  - Units: Ų (cubic Angstroms)
  - Calculation: Integration over the molecular surface
  - Applications: Molecular size comparison, density calculations

**Cavity Volume**
  - Volume of internal cavities and pockets
  - Important for binding site analysis
  - Calculated using cavity detection algorithms
  - Relevant for drug design and enzyme studies

**Solvent-Accessible Volume**
  - Volume accessible to solvent molecules
  - Depends on probe radius (typically 1.4 Å for water)
  - Used in solvation energy calculations
  - Important for solubility predictions

Surface Area Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~

**Molecular Surface Area**
  - Total area of the molecular surface
  - Units: Ų (square Angstroms)
  - Fundamental descriptor for surface-based properties
  - Correlates with many physicochemical properties

**Solvent-Accessible Surface Area (SASA)**
  - Surface area accessible to solvent
  - Probe-dependent calculation
  - Important for solvation studies
  - Used in implicit solvent models

**Polar Surface Area**
  - Surface area of polar atoms (N, O, S, P)
  - Important for drug permeability predictions
  - Correlates with blood-brain barrier penetration
  - Used in ADMET property prediction

Shape Descriptors
~~~~~~~~~~~~~~~~~

**Sphericity Index**
  - Measures how close the molecule is to a perfect sphere
  - Range: 0 (linear) to 1 (perfect sphere)
  - Formula: (π^(1/3) * (6V)^(2/3)) / A
  - Where V = volume, A = surface area

**Asphericity**
  - Quantifies deviation from spherical shape
  - Based on principal moments of inertia
  - Higher values indicate more elongated shapes
  - Important for molecular recognition studies

**Eccentricity**
  - Measures the elongation of the molecule
  - Based on the ratio of principal axes
  - Values close to 0 indicate spherical shapes
  - Values close to 1 indicate highly elongated shapes

**Compactness**
  - Ratio of surface area to volume
  - Normalized measure of molecular compactness
  - Lower values indicate more compact molecules
  - Important for understanding molecular packing

Geometric Moments
~~~~~~~~~~~~~~~~~

**Principal Moments of Inertia**
  - I₁, I₂, I₃ (ordered: I₁ ≤ I₂ ≤ I₃)
  - Characterize mass distribution
  - Used to calculate shape indices
  - Important for rotational properties

**Radius of Gyration**
  - Root-mean-square distance from center of mass
  - Measure of molecular size and compactness
  - Important for polymer and protein studies
  - Used in molecular dynamics analysis

**Inertial Shape Descriptors**
  - Derived from principal moments
  - Include prolate/oblate parameters
  - Characterize molecular anisotropy
  - Used in shape-based drug design

Calculation Methods
-------------------

Surface Generation
~~~~~~~~~~~~~~~~~~

1. **Molecular Surface Construction**
   - Van der Waals surface generation
   - Solvent-accessible surface calculation
   - Smooth surface triangulation
   - Quality control and validation

2. **Grid-Based Methods**
   - Volumetric grid generation
   - Surface point sampling
   - Integration over grid points
   - Accuracy vs. computational cost trade-offs

3. **Analytical Methods**
   - Exact geometric calculations where possible
   - Analytical surface area formulas
   - Precise volume calculations
   - Higher accuracy for simple shapes

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example calculation of size and shape descriptors
   from surfacia.descriptors import SizeShapeDescriptors
   
   # Initialize calculator
   calculator = SizeShapeDescriptors()
   
   # Calculate descriptors from surface file
   descriptors = calculator.calculate(
       surface_file="molecule.wfn",
       probe_radius=1.4,  # For SASA calculations
       grid_resolution=0.1  # Grid spacing in Angstroms
   )
   
   # Access individual descriptors
   volume = descriptors['molecular_volume']
   surface_area = descriptors['surface_area']
   sphericity = descriptors['sphericity_index']

Parameters and Options
~~~~~~~~~~~~~~~~~~~~~~

**Probe Radius**
  - Default: 1.4 Å (water molecule)
  - Affects SASA and cavity calculations
  - Can be adjusted for different solvents
  - Typical range: 1.0-2.0 Å

**Grid Resolution**
  - Controls calculation accuracy
  - Trade-off between speed and precision
  - Typical values: 0.05-0.2 Å
  - Finer grids for small molecules

**Surface Type**
  - Van der Waals surface
  - Solvent-accessible surface
  - Molecular surface (Connolly surface)
  - Each type gives different results

Applications
------------

Drug Design
~~~~~~~~~~~

- **Molecular size filtering**: Remove compounds outside size ranges
- **Shape-based screening**: Find molecules with similar shapes
- **Binding site complementarity**: Match ligand and receptor shapes
- **ADMET prediction**: Use size/shape for property prediction

Chemical Space Analysis
~~~~~~~~~~~~~~~~~~~~~~~

- **Diversity analysis**: Quantify structural diversity
- **Clustering**: Group molecules by shape similarity
- **Outlier detection**: Identify unusual molecular shapes
- **Library design**: Ensure shape diversity in compound libraries

Property Prediction
~~~~~~~~~~~~~~~~~~~

- **Solubility**: Correlate with surface area and volume
- **Permeability**: Use polar surface area and shape
- **Binding affinity**: Shape complementarity analysis
- **Selectivity**: Shape-based selectivity prediction

Validation and Quality Control
------------------------------

Accuracy Assessment
~~~~~~~~~~~~~~~~~~~

- **Reference calculations**: Compare with literature values
- **Method validation**: Test against known standards
- **Convergence testing**: Ensure grid-independent results
- **Cross-validation**: Verify consistency across methods

Common Issues
~~~~~~~~~~~~~

- **Surface artifacts**: Check for unrealistic surface features
- **Grid effects**: Ensure adequate resolution
- **Probe accessibility**: Verify realistic cavity detection
- **Numerical precision**: Monitor calculation stability

Best Practices
~~~~~~~~~~~~~~

1. **Use appropriate probe radius** for the intended application
2. **Validate grid resolution** with convergence tests
3. **Check surface quality** before descriptor calculation
4. **Compare multiple methods** for critical applications
5. **Document parameters** used for reproducibility

Integration with Other Descriptors
-----------------------------------

Size and shape descriptors are often combined with:

- **Electronic descriptors**: For comprehensive molecular characterization
- **Surface analysis descriptors**: For detailed surface properties
- **Pharmacophore descriptors**: For drug design applications
- **Topological descriptors**: For structure-activity relationships

The combination provides a complete picture of molecular properties and enables more accurate predictions and analyses.

References and Further Reading
------------------------------

- Connolly, M.L. (1983). Solvent-accessible surfaces of proteins and nucleic acids. *Science*, 221(4612), 709-713.
- Richards, F.M. (1977). Areas, volumes, packing, and protein structure. *Annual Review of Biophysics and Bioengineering*, 6(1), 151-176.
- Ertl, P., Rohde, B., & Selzer, P. (2000). Fast calculation of molecular polar surface area as a sum of fragment-based contributions. *Journal of Medicinal Chemistry*, 43(20), 3714-3717.