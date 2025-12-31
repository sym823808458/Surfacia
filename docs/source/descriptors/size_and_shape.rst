Size and Shape Descriptors
=========================

The characterization of a molecule's physical presence is founded upon a set of fundamental descriptors that quantify its size, mass, and overall shape. These parameters provide the essential framework for understanding steric interactions, transport properties, and intermolecular recognition.

Overview
--------

Size and shape descriptors (22 features total) characterize:

- **Mass and composition**: Basic molecular properties
- **Dimensions**: 3D spatial extent measurements
- **Planarity**: Deviation from planar geometry
- **Shape indices**: Mass distribution metrics

1.1 Basic Properties & Mass (3 Features)
-------------------------------------

The most elementary molecular descriptors establish the system's basic composition.

**Atom Number**
   - **Units**: unitless
   - **Definition**: Total count of all atoms constituting the molecule
   - **Calculation**: Simple atom enumeration
   - **Application**: Determines molecular complexity and computational cost

**Occupied Orbitals**
   - **Units**: unitless
   - **Definition**: Total number of electron-occupied molecular orbitals in ground-state configuration
   - **Calculation**: Count of occupied MOs from wavefunction analysis
   - **Application**: Related to electronic structure and orbital space

**Molecule Weight**
   - **Units**: Da (Daltons)
   - **Definition**: Cumulative mass of all constituent atoms
   - **Calculation**: Sum of standard IUPAC atomic weights
   - **Formula**: M = Σ(m_i) where m_i is atomic mass
   - **Application**: Baseline for molecular size normalization

1.2 Dimensional & Size (9 Features)
-----------------------------------

The three-dimensional extent and surface characteristics of a molecule are quantified through geometric descriptors computed from the electron density isosurface.

**Isosurface Area**
   - **Units**: Å²
   - **Definition**: Surface area of the 0.01 a.u. electron density isosurface
   - **Calculation**: Triangulation of constant electron density surface
   - **Application**: Measures accessible surface for external interactions, relates to solvation and binding

**Sphericity**
   - **Units**: unitless (0 to 1)
   - **Definition**: Deviation of surface from a perfect sphere
   - **Calculation**: 
     
     .. math::
     
        \mathrm{Sphericity}=\frac{\pi^{1/3}\times\left(6V\right)^{2/3}}{A}
     
     where V is molecular volume and A is surface area
   - **Interpretation**: 
     - 1.0 = perfect sphere
     - 0.0-0.5 = increasing deviation from spherical geometry
   - **Application**: Characterizes molecular compactness

**Farthest Distance**
   - **Units**: Å
   - **Definition**: Maximum separation between any two atomic nuclei
   - **Calculation**: 
     
     .. math::
     
        \mathrm{Farthest\ Distance}=\max_{i,j}{\left|r_i-r_j\right|}
     
     where r_i, r_j are nuclear coordinates
   - **Application**: Defines maximal spatial extent of molecule

**Molecular Radius**
   - **Units**: Å
   - **Definition**: Maximum distance from any atom to the molecule's center of mass
   - **Calculation**: 
     
     .. math::
     
        \mathrm{Molecular\ Radius}=\max_{i}{\left|r_i-r_{cm}\right|}
     
     where r_cm is center of mass
   - **Application**: Effective spherical boundary of molecule

**Principal Dimensions (3 Features)**

Molecular dimensions are determined through axis-aligned bounding box approach after orienting molecule along principal axes:

**Molecular Size Short**
   - **Units**: Å
   - **Definition**: Shortest bounding box dimension
   - **Application**: Minimum molecular extent

**Molecular Size Medium**
   - **Units**: Å
   - **Definition**: Intermediate bounding box dimension
   - **Application**: Intermediate molecular extent

**Molecular Size Long**
   - **Units**: Å
   - **Definition**: Longest bounding box dimension
   - **Application**: Maximum molecular extent

.. note::
   
   **Principal Axis Alignment**
   
   Molecules are automatically rotated to align with principal axes of inertia before dimension calculation:
   
   - Longest axis: Smallest moment of inertia (easiest to spin)
   - Shortest axis: Largest moment of inertia (hardest to spin)
   - Ensures reproducible, standardized orientation

**Long/Sum Size Ratio**
   - **Units**: unitless
   - **Definition**: Normalized measure of molecular elongation
   - **Calculation**:
     
     .. math::
     
        \mathrm{Long/Sum\ Ratio}=\frac{L_{long}}{L_{short}+L_{medium}+L_{long}}
     
   - **Range**: ~0.33 (spherical) to ~1.0 (linear)
   - **Application**: Quantifies aspect ratio anisotropy

**Length/Diameter Ratio**
   - **Units**: unitless
   - **Definition**: Alternative aspect ratio measurement
   - **Calculation**:
     
     .. math::
     
        \mathrm{Length/Diameter}=\frac{L_{long}}{2\times R}
     
     where R is molecular radius
   - **Application**: Insight into molecular anisotropy and elongation

1.3 Planarity (2 Features)
-------------------------

Molecular planarity is a critical factor in conjugated and aromatic systems, assessed quantitatively through geometric fitting.

**Molecular Planarity Parameter (MPP)**
   - **Units**: Å
   - **Definition**: Root mean square deviation of atomic positions from fitted plane
   - **Calculation**: 
     
     .. math::
     
        MPP=\sqrt{\frac{1}{N_{atom}}\times\sum_{i} d_i^2}
     
     where d_i is perpendicular distance to fitted plane:
     
     .. math::
     
        d_i=\frac{\left|Ax_i+By_i+Cz_i+D\right|}{\sqrt{A^2+B^2+C^2}}
     
   - **Application**: 
     - 0.0 = perfectly planar
     - Larger values = increasing deviation from planarity
     - Important for aromaticity and conjugation analysis

**Span of Deviation from Plane (SDP)**
   - **Units**: Å
   - **Definition**: Full range of atomic deviations from fitted plane
   - **Calculation**: Uses signed distances:
     
     .. math::
     
        d_s=\frac{Ax_i+By_i+Cz_i+D}{\sqrt{A^2+B^2+C^2}}
     
     .. math::
     
        SDP=d_{s,max}-d_{s,min}
     
   - **Application**: Captures extent of atomic distribution on both sides of fitted plane, complementary to MPP

1.4 Shape Descriptors (8 Features)
-------------------------------

Advanced shape characterization is achieved through descriptors derived from the molecule's mass distribution, quantified via the moment of inertia tensor.

**Principal Moments of Inertia (3 Features)**

**Principal_Moment_I1**
   - **Units**: amu·Å²
   - **Definition**: Minimum principal moment of inertia
   - **Application**: Smallest rotational inertia value

**Principal_Moment_I2**
   - **Units**: amu·Å²
   - **Definition**: Intermediate principal moment of inertia
   - **Application**: Intermediate rotational inertia value

**Principal_Moment_I3**
   - **Units**: amu·Å²
   - **Definition**: Maximum principal moment of inertia
   - **Application**: Largest rotational inertia value

.. math::

   The inertia tensor components are calculated as:
   
   I_{xx}=\sum_{i}{m_i\left(y_i^2+z_i^2\right)}
   
   I_{yy}=\sum_{i}{m_i\left(x_i^2+z_i^2\right)}
   
   I_{zz}=\sum_{i}{m_i\left(x_i^2+y_i^2\right)}
   
   I_{xy}=-\sum_{i} m_i\times x_i\times y_i
   
   I_{xz}=-\sum_{i} m_i\times x_i\times z_i
   
   I_{zy}=-\sum_{i} m_i\times x_z\times z_y
   
   where m_i is atomic mass, (x_i, y_i, z_i) are center-of-mass coordinates
   
   Diagonalization yields I₁ ≤ I₂ ≤ I₃ (invariant under rotation)

**Shape_Asphericity**
   - **Units**: unitless
   - **Definition**: Deviation from perfect spherical symmetry in terms of mass distribution
   - **Calculation**:
     
     .. math::
     
        \mathrm{Shape_Asphericity}=\frac{1}{2}\times\frac{\left(I_1-I_2\right)^2+\left(I_1-I_3\right)^2+\left(I_2-I_3\right)^2}{I_1^2+I_2^2+I_3^2}
     
   - **Range**: 0 to 0.5
     - 0 = perfect sphere (I₁ = I₂ = I₃)
     - 0.5 = perfect line (I₁ = I₂ = 0)
     - 0.25 = perfect disk (I₁ = 0, I₂ = I₃)
   - **Application**: Mass-based shape characterization

**Radius of Gyration Related Descriptors (3 Features)**

**Shape_Gyradius**
   - **Units**: Å
   - **Definition**: Mass-weighted root mean square distance of atoms from center of mass
   - **Calculation**:
     
     .. math::
     
        R_g=\sqrt{\frac{\sum_{i} m_i\times r_i^2}{\sum_{i} m_i}}
     
     where r_i is distance from atom i to center of mass
   - **Application**: Classical descriptor from polymer physics, reflects molecular compactness

**Shape_Relative_Gyradius**
   - **Units**: unitless
   - **Definition**: Size-normalized measure of compactness compared to equivalent-volume sphere
   - **Calculation**:
     
     Multi-step process:
     
     1. Calculate bounding box volume: V_{box}=L_{short}×L_{medium}×L_{long}
     
     2. Determine equivalent sphere radius: R_{equiv}=√[3]{\frac{3×V_{box}}{4×\pi}}
     
     3. Calculate theoretical spherical gyradius: R_{g,sphere}=√{\frac{3}{5}}×R_{equiv}
     
     4. Compute relative gyradius: RGR=\frac{R_g}{R_{g,sphere}}
   
   - **Interpretation**:
     - < 1 = more compact than sphere (typical of linear molecules)
     - ≈ 1 = sphere-like geometry
     - > 1 = extended or hollow structures

**Shape_Waist_Variance**
   - **Units**: Å²
   - **Definition**: Cross-sectional size variation along principal molecular axis
   - **Calculation**:
     
     .. math::
     
        \mathrm{Waist\ Variance}=\mathrm{Var}{w_1,w_2,\ldots,w_n}
     
     where w_i represents maximum cross-sectional width in slice i along principal axis
   
   - **Algorithm**:
     1. Project all atoms onto principal axis
     2. Divide molecule into n slices
     3. Calculate variance in cross-sectional dimensions
   - **Application**: 
     - High variance = pronounced "dumbbell" or constricted shapes
     - Low variance = uniform cylindrical or spherical geometry
     - Inspired by bottleneck analysis in protein channels

**Shape_Geometric_Asphericity**
   - **Units**: unitless
   - **Definition**: Mass-independent geometric analog to traditional asphericity
   - **Calculation**:
     
     .. math::
     
        \mathrm{Geometric\ Asphericity}=\frac{1}{2}\times\frac{\left(L_s-L_m\right)^2+\left(L_s-L_l\right)^2+\left(L_m-L_l\right)^2}{L_s^2+L_m^2+L_l^2}
     
     where L_s ≤ L_m ≤ L_l are sorted molecular box dimensions
   - **Range**: 0 (perfect cube) to 0.5 (perfect line)
   - **Application**: Shape information independent of atomic masses

Applications
------------

**QSAR/QSPR Modeling**
   - Steric descriptors for activity prediction
   - Shape complementarity for binding affinity
   - Transport property modeling

**Drug Discovery**
   - Oral bioavailability prediction
   - Blood-brain barrier penetration
   - Solubility estimation

**Materials Science**
   - Polymer property prediction
   - Molecular packing analysis
   - Crystal structure design

**Computational Chemistry**
   - Conformation analysis
   - Molecular dynamics validation
   - Geometry optimization monitoring

References
----------

- **Classic Textbook**: "Molecular Descriptors for QSAR/QSPR" by Todeschini & Consonni
- **Shape Analysis**: Connolly (1983) on molecular surface analysis
- **Principal Moments**: Classic text on rotational dynamics

See Also
--------

- :doc:`electronic_properties`: Electronic descriptor definitions
- :doc:`mqsa_modes`: Multi-scale analysis approaches
- :doc:`../api/descriptors`: API reference
