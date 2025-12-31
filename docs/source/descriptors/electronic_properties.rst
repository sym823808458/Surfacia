Electronic Properties Descriptors
==============================

The electronic properties of a molecule, which govern its reactivity and interaction potential, are described by a suite of quantum mechanical descriptors. These parameters capture electron donor-acceptor behavior, charge distribution, and stability.

Overview
--------

Electronic descriptors (28 features total) characterize:

- **Frontier orbitals**: HOMO, LUMO energies and gap
- **Orbital delocalization**: Spatial distribution of electrons
- **Surface analysis**: Local properties on vdW surface
- **Dipole moment**: Overall charge separation

2.1 Frontier Molecular Orbitals (3 Features)
-----------------------------------------

According to frontier molecular orbital theory, chemical reactivity is primarily dictated by highest occupied and lowest unoccupied molecular orbitals.

**HOMO (Highest Occupied Molecular Orbital)**
   - **Units**: a.u. (atomic units)
   - **Definition**: Energy level of the highest occupied molecular orbital
   - **Application**: 
     - High energy = good electron donor (nucleophile)
     - Low energy = poor electron donor
     - Correlates with oxidation potential

**LUMO (Lowest Unoccupied Molecular Orbital)**
   - **Units**: a.u.
   - **Definition**: Energy level of the lowest unoccupied molecular orbital
   - **Application**: 
     - Low energy = good electron acceptor (electrophile)
     - High energy = poor electron acceptor
     - Correlates with reduction potential

**HOMO-LUMO Gap**
   - **Units**: a.u.
   - **Definition**: Energy difference between HOMO and LUMO
   - **Formula**: Gap = E(LUMO) - E(HOMO)
   - **Application**: 
     - Large gap = chemically stable, low reactivity
     - Small gap = high reactivity, soft molecules
     - Key indicator of kinetic stability

2.2 Orbital Delocalization Index (6 Features)
------------------------------------------

Orbital delocalization indices provide quantitative measures of electron delocalization within molecular orbitals, offering insights into electron mobility and chemical bonding character.

**The ODI Concept**

The Orbital Delocalization Index (ODI) was defined by Tian Lu to quantify extent of orbital spatial delocalization. For orbital i, the ODI value is calculated using the following formula:

.. math::

   {\rm ODI}_i=0.01\times\sum_{A}{\left(\Theta_{A,i}\right)^2}

where Θ_(A,i) represents the composition of atom A in orbital i.

**Interpretation**

- **Low ODI values** (0.1-0.3): Strong delocalization
  - Characteristic of π-conjugated systems
  - Aromatic compounds
  - Extended bonding networks
  
- **High ODI values** (0.5-1.0): Localized orbitals
  - Characteristic of σ-bonds
  - Isolated functional groups
  - Non-conjugated systems

**Specific Descriptors**

**ODI_HOMO_1**
   - **Units**: unitless
   - **Definition**: ODI value for HOMO-1 orbital
   - **Application**: Delocalization of second-highest occupied orbital

**ODI_HOMO**
   - **Units**: unitless
   - **Definition**: ODI value for HOMO orbital
   - **Application**: Electron donor orbital delocalization, critical for reactivity

**ODI_LUMO**
   - **Units**: unitless
   - **Definition**: ODI value for LUMO orbital
   - **Application**: Electron acceptor orbital delocalization, influences electrophilicity

**ODI_LUMO+1**
   - **Units**: unitless
   - **Definition**: ODI value for LUMO+1 orbital
   - **Application**: Delocalization of second-lowest unoccupied orbital

**ODI_Mean**
   - **Units**: unitless
   - **Definition**: Average orbital delocalization index across all considered orbitals
   - **Calculation**: Mean of ODI_HOMO_1, ODI_HOMO, ODI_LUMO, ODI_LUMO+1
   - **Application**: Overall measure of electron delocalization within the molecule

**ODI_Standard_Deviation**
   - **Units**: unitless
   - **Definition**: Quantifies heterogeneity in delocalization patterns
   - **Calculation**: Standard deviation of ODI values
   - **Application**: 
     - Large values = significant variation in electron mobility
     - Small values = uniform delocalization across orbitals
     - Indicates mixed bonding character

2.3 Molecular Quantitative Surface Analysis (18 Features)
----------------------------------------------------

Local electronic properties provide detailed insights into molecular reactivity and interaction sites through surface-based analysis of electron density-derived quantities.

### 2.3.1 Theoretical Foundation

The analysis of local electronic properties relies on partitioning the molecular van der Waals surface into atom-specific regions using a weight-based assignment algorithm.

**Surface Point Assignment**

Surface points are attributed to atoms based on distance-weighted calculations:

.. math::

   w_A=1-\frac{\left|r-r_A\right|}{R_A}

where r represents a surface point, r_A is the atomic coordinate, and R_A is the atomic radius. Each surface point is assigned to the atom with the highest weight value, naturally accommodating atomic size differences.

### 2.3.2 Local Electron Attachment Energy (LEAE)

Local Electron Attachment Energy characterizes the local propensity for electron acceptance across the molecular surface.

**LEAE Formula**

.. math::

   E_{att}\left(r\right)=\frac{\sum_{i=LUMO}^{\varepsilon_i<0}\left|\phi_i\left(r\right)\right|^2\times\varepsilon_i}{\rho\left(r\right)}

where ρ(r) represents total electron density, |φ_i(r)|² is the probability density of the i-th unoccupied molecular orbital, and ε_i is the corresponding orbital energy.

**Descriptors (4 Features)**

**LEAE_Minimal_Value**
   - **Units**: eV
   - **Definition**: Minimum LEAE value on molecular surface
   - **Application**: Identifies the hardest electron-accepting region (most stable toward reduction)

**LEAE_Maximal_Value**
   - **Units**: eV
   - **Definition**: Maximum LEAE value on molecular surface
   - **Application**: Locates the most electrophilic site (easiest to reduce)

**LEAE_Average_Value**
   - **Units**: eV
   - **Definition**: Average LEAE across molecular surface
   - **Application**: Overall measure of electron-accepting capability

**LEAE_Variance**
   - **Units**: eV²
   - **Definition**: Variance in LEAE values across surface
   - **Application**: Quantifies heterogeneity in electron affinity

### 2.3.3 Electrostatic Potential (ESP)

The molecular electrostatic potential describes the interaction energy between the molecule and a unit positive test charge.

**ESP Formula**

.. math::

   V\left(r\right)=\sum_{A}\frac{Z_A}{\left|r-R_A\right|}-\int\frac{\rho\left(r^\prime\right)}{\left|r-r^\prime\right|}dr^\prime

where Z_A represents nuclear charges and R_A denotes nuclear coordinates.

**Basic Descriptors (3 Features)**

**ESP_Minimal_Value**
   - **Units**: kcal/mol
   - **Definition**: Minimum ESP on molecular surface
   - **Application**: Identifies the most electron-rich (nucleophilic) region

**ESP_Maximal_Value**
   - **Units**: kcal/mol
   - **Definition**: Maximum ESP on molecular surface
   - **Application**: Locates the most electron-deficient (electrophilic) site

**ESP_Overall_Average_Value**
   - **Units**: kcal/mol
   - **Definition**: Average ESP across molecular surface
   - **Application**: Characterizes the general electrostatic environment

**Statistical Descriptors (7 Features)**

**ESP_Overall_Variance**
   - **Units**: (kcal/mol)²
   - **Definition**: σ²_tot, quantifies electrostatic potential heterogeneity
   - **Application**: High values indicate diverse electrostatic environment

**Balance_of_Charges (ν)**
   - **Units**: unitless
   - **Definition**: Measure of charge balance
   - **Formula**:
     
     .. math::
     
        \nu=\frac{\sigma_+^2\times\sigma_-^2}{\left(\sigma_{tot}^2\right)^2}
   
   - **Application**: 
     - ν = 0: One charged region dominates
     - ν = 1: Perfect balance between positive and negative regions
     - Indicates polarity character

**Product_of_Sigma_and_Nu**
   - **Units**: (kcal/mol)²
   - **Definition**: Composite charge indicator
   - **Formula**: Product of σ²_tot and ν
   - **Application**: Combined measure of polarity magnitude and balance

**Internal_Charge_Separation (Π)**
   - **Units**: kcal/mol
   - **Definition**: Average absolute deviation from surface ESP mean
   - **Formula**:
     
     .. math::
     
        \Pi=\frac{1}{t}\sum_{t}\left|V\left(r_k\right)-\bar{V_s}\right|
   
   - **Application**: Quantifies internal charge distribution

**Molecular_Polarity_Index (MPI)**
   - **Units**: kcal/mol
   - **Definition**: Average absolute electrostatic potential
   - **Formula**:
     
     .. math::
     
        MPI=\frac{1}{t}\sum_{t}\left|V\left(r_k\right)\right|
   
   - **Application**: Overall polarity measure, correlates with solvation

**Polar_Surface_Area**
   - **Units**: Å²
   - **Definition**: Surface area with |ESP| > 10 kcal/mol
   - **Application**: Regions of significant electrostatic character

**Polar_Surface_Area_Percent**
   - **Units**: %
   - **Definition**: Percentage of polar surface relative to total surface area
   - **Application**: Important for drug-likeness and membrane permeability

### 2.3.4 Average Local Ionization Energy (ALIE)

Average Local Ionization Energy identifies sites susceptible to electrophilic attack.

**ALIE Formula**

.. math::

   \bar{I}\left(r\right)=\frac{\sum_{i}{\rho_i\left(r\right)\left|\varepsilon_i\right|}}{\rho\left(r\right)}

where ρ_i(r) and ε_i represent the electron density and orbital energy of the i-th occupied molecular orbital.

**Descriptors (4 Features)**

**ALIE_Minimal_Value**
   - **Units**: eV
   - **Definition**: Minimum ALIE on molecular surface
   - **Application**: Identifies the most nucleophilic site (easiest to oxidize)

**ALIE_Maximal_Value**
   - **Units**: eV
   - **Definition**: Maximum ALIE on molecular surface
   - **Application**: Locates the least nucleophilic site (hardest to oxidize)

**ALIE_Average_Value**
   - **Units**: eV
   - **Definition**: Average ALIE across molecular surface
   - **Application**: Overall measure of electron-donating ability

**ALIE_Variance**
   - **Units**: eV²
   - **Definition**: Variance in ALIE values across surface
   - **Application**: Quantifies heterogeneity in ionization energy

2.4 Dipole Moment (1 Feature)
-------------------------------

**Dipole_Moment**
   - **Units**: a.u. (atomic units)
   - **Definition**: Overall charge separation within the molecule
   - **Formula**: μ = Σ(q_i × r_i) where q_i is atomic charge and r_i is position
   - **Application**: 
     - Influences intermolecular interactions
     - Affects solvation properties
     - Correlates with dielectric constant

Applications
------------

**Chemical Reactivity**
   - Predicting reaction sites and mechanisms
   - Understanding electron transfer processes
   - Designing redox-active compounds

**Drug Design**
   - Pharmacophore identification
   - ADMET property prediction
   - Structure-activity relationships

**Materials Science**
   - Electronic property engineering
   - Charge transport prediction
   - Surface chemistry optimization

**Theoretical Chemistry**
   - Validating quantum chemical calculations
   - Benchmarking computational methods
   - Understanding chemical bonding

References
----------

- **Frontier Orbital Theory**: Fukui (1952) "The role of frontier orbitals in chemical reactions"
- **Electrostatic Potential**: Murray et al. (1978) "Electrostatic potential"
- **Multiwfn Documentation**: http://sobereva.com/multiwfn/

See Also
--------

- :doc:`size_and_shape`: Geometric descriptor definitions
- :doc:`mqsa_modes`: Multi-scale analysis approaches
- :doc:`../api/descriptors`: API reference
