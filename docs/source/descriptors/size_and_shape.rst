Size and Shape Descriptors
==========================

Size and shape descriptors quantify how a molecule occupies space, how compact it is, and how strongly it deviates from simple spherical or planar geometries.

Overview
--------

These descriptors are useful when a property depends on:

- steric accessibility
- exposed surface extent
- compactness versus extension
- molecular anisotropy
- planarity

Descriptor Families
-------------------

Basic Composition
~~~~~~~~~~~~~~~~~

**Atom Number**
  - Number of atoms in the molecule
  - Unitless

**Molecule Weight**
  - Total molecular mass
  - Units: Da

**Occupied Orbitals**
  - Number of occupied molecular orbitals
  - Unitless

Surface and Dimension Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Isosurface Area**
  - Surface area of the 0.01 a.u. electron-density isosurface
  - Units: Å²

**Sphericity**
  - Measures how close the molecular surface is to a sphere
  - Unitless

**Farthest Distance**
  - Maximum separation between two atoms
  - Units: Å

**Molecular Size Short / Medium / Long**
  - Bounding-box dimensions after alignment to the principal axes
  - Units: Å

**Long/Sum Size Ratio**
  - Normalized elongation descriptor
  - Unitless

**Length/Diameter**
  - Aspect-ratio-like descriptor for anisotropy
  - Unitless

**Molecular Radius**
  - Effective outer radial size from the center of mass
  - Units: Å

Planarity Descriptors
~~~~~~~~~~~~~~~~~~~~~

**Molecular Planarity Parameter (MPP)**
  - Root-mean-square deviation from the best-fit plane
  - Units: Å

**Span of Deviation from Plane (SDP)**
  - Full signed spread of atomic deviations around the fitted plane
  - Units: Å

Shape Descriptors
~~~~~~~~~~~~~~~~~

**Principal Moments of Inertia (I1, I2, I3)**
  - Rotation-invariant measures of mass distribution
  - Units: amu·Å²

**Shape_Asphericity**
  - Deviation from spherical symmetry based on mass distribution
  - Unitless
  - Typical range: 0 to 0.5

**Shape_Gyradius**
  - Radius of gyration
  - Units: Å
  - Larger values generally indicate more extended structures

**Shape_Relative_Gyradius**
  - Radius of gyration normalized against an equivalent sphere
  - Unitless
  - Useful for comparing molecules of different sizes

**Shape_Waist_Variance**
  - Cross-sectional variation along the principal molecular axis
  - Units: Å²
  - High values often indicate dumbbell-like or constricted shapes

**Shape_Geometric_Asphericity**
  - Geometric version of asphericity based on bounding-box dimensions
  - Unitless
  - Useful when you want shape information independent of atomic masses

How to Interpret Them Quickly
-----------------------------

- larger ``Isosurface Area`` often means more exposed surface for interaction
- larger ``Molecular Radius`` usually means a bulkier molecular envelope
- lower ``Sphericity`` generally means less compact geometry
- larger ``Shape_Asphericity`` or ``Shape_Geometric_Asphericity`` means stronger anisotropy
- larger ``Shape_Waist_Variance`` suggests stronger shape variation along the long axis
- smaller ``MPP`` and ``SDP`` usually mean a more planar structure

Why They Matter
---------------

These descriptors are often useful for interpreting:

- solubility and exposed surface effects
- steric control in catalysis
- scaffold compactness versus extension
- shape complementarity in recognition problems

In practice, Surfacia often combines them with electronic and surface descriptors so that shape effects are interpreted together with ESP, ALIE, and LEAE.
