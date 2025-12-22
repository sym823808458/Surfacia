Molecular Descriptors Reference
===============================

Surfacia generates comprehensive molecular descriptors across three hierarchical levels, providing a complete characterization of molecular properties from quantum mechanical calculations and surface analysis.

.. toctree::
   :maxdepth: 2

   size_and_shape
   electronic_properties
   surface_analysis

Overview
--------

Surfacia's descriptor framework is built on the principle that molecular function is fundamentally determined by surface-mediated interactions. Our multi-scale approach systematically captures:

- **Atomic Level**: Individual atom surface properties and local reactivity
- **Functional Group Level**: Chemical fragment characteristics and modular behavior
- **Molecular Level**: Global molecular properties and bulk characteristics

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 📏 Size and Shape
      :link: size_and_shape
      :link-type: doc

      22 descriptors covering molecular dimensions, geometry, and shape characteristics

   .. grid-item-card:: ⚡ Electronic Properties
      :link: electronic_properties
      :link-type: doc

      28 descriptors from quantum mechanical calculations and orbital analysis

   .. grid-item-card:: 🌊 Surface Analysis
      :link: surface_analysis
      :link-type: doc

      13-32 descriptors from multi-scale quantitative surface analysis

Descriptor Categories
---------------------

**Total Descriptors Generated**: 50-100+ features depending on analysis mode

**1. Size and Shape (22 Features)**

Basic molecular characteristics that define the physical presence and geometric properties:

- **Basic Properties** (3 features): Atom count, molecular weight, orbital occupancy
- **Dimensional Analysis** (9 features): Molecular size, sphericity, surface area
- **Planarity Measures** (2 features): Deviation from planarity, geometric flatness
- **Advanced Shape** (8 features): Moment of inertia, asphericity, compactness

**2. Electronic Properties (28 Features)**

Quantum mechanical descriptors that govern chemical reactivity and interactions:

- **Frontier Orbitals** (3 features): HOMO, LUMO, and energy gap
- **Orbital Delocalization** (6 features): Electron mobility and bonding character
- **Surface Electronics** (19 features): Local reactivity and charge distribution

**3. Multi-Scale Surface Analysis (Variable Features)**

Surface-based descriptors that directly encode chemical reactivity patterns:

- **Element-Specific** (13 features): Properties of specific atomic types
- **Fragment-Specific** (18 features): User-defined chemical fragment analysis
- **LOFFI Automated** (32 features): Comprehensive automated surface analysis

Scientific Foundation
---------------------

**Quantum Mechanical Basis**

All descriptors are derived from high-level quantum mechanical calculations:

.. mermaid::

   graph TD
       A[SMILES Input] --> B[3D Structure]
       B --> C[Gaussian 16 DFT]
       C --> D[Wavefunction Analysis]
       D --> E[Surface Property Mapping]
       E --> F[Multi-Scale Descriptors]
       
       style C fill:#e1f5fe
       style F fill:#f3e5f5

**Surface-Based Analysis**

Surface properties are calculated using the molecular van der Waals surface:

- **LEAE (Local Electron Attachment Energy)**: Electron-accepting capability
- **ESP (Electrostatic Potential)**: Charge distribution and polarity
- **ALIE (Average Local Ionization Energy)**: Electron-donating ability

**Multi-Scale Integration**

The framework systematically analyzes properties at different organizational levels:

1. **Atomic Level**: Individual surface atoms and their local environments
2. **Functional Group Level**: Chemical fragments and their collective behavior
3. **Molecular Level**: Global properties and overall molecular character

Chemical Interpretability
-------------------------

**Direct Chemical Meaning**

Every descriptor has clear chemical interpretation:

.. csv-table::
   :header: "Descriptor", "Chemical Meaning", "Interpretation"
   :widths: 25, 35, 40

   "ALIE_min", "Most nucleophilic site", "Lower values = stronger electron donor"
   "ESP_max", "Most electrophilic site", "Higher values = stronger electron acceptor"
   "HOMO", "Electron-donating orbital", "Higher values = easier electron donation"
   "Sphericity", "Molecular compactness", "Values near 1 = spherical shape"

**Structure-Activity Relationships**

Descriptors directly relate to molecular function:

- **Drug-Target Binding**: Surface complementarity and electronic matching
- **Catalytic Activity**: Active site electronic properties and accessibility
- **Transport Properties**: Size, shape, and surface characteristics
- **Stability**: Electronic structure and molecular geometry

**Design Insights**

SHAP analysis of descriptors provides actionable design principles:

- **Which atoms** contribute most to desired properties
- **How modifications** would affect molecular behavior
- **Where to focus** synthetic efforts for optimization
- **What changes** to avoid based on negative contributions

Descriptor Validation
---------------------

**Chemical Consistency**

All descriptors are validated against known chemical principles:

- **Periodic Trends**: Descriptors follow expected periodic behavior
- **Functional Group Effects**: Consistent with known group contributions
- **Stereochemical Effects**: Capture stereochemical influences correctly
- **Size Dependencies**: Scale appropriately with molecular size

**Statistical Properties**

Descriptors exhibit appropriate statistical characteristics:

- **Dynamic Range**: Sufficient variation across molecular space
- **Correlation Structure**: Reasonable correlation patterns
- **Outlier Behavior**: Appropriate handling of unusual molecules
- **Numerical Stability**: Robust calculation across diverse structures

**Predictive Power**

Validation across multiple chemical systems demonstrates:

- **High Accuracy**: Strong predictive performance for target properties
- **Transferability**: Consistent performance across different chemical spaces
- **Interpretability**: SHAP analysis provides meaningful chemical insights
- **Robustness**: Stable performance with different molecular sets

Usage Guidelines
----------------

**Descriptor Selection**

Choose appropriate descriptor sets based on your analysis goals:

.. tabs::

   .. tab:: General Analysis

      Use **LOFFI Automated Analysis** (32 features) for:
      
      - Exploratory analysis of diverse molecular libraries
      - Unbiased feature discovery
      - General QSAR model development
      - Comprehensive molecular characterization

   .. tab:: Element-Focused

      Use **Element-Specific Analysis** (13 features) for:
      
      - Studying specific atomic contributions (e.g., sulfur, fluorine)
      - Element-centric hypothesis testing
      - Focused medicinal chemistry optimization
      - Understanding heteroatom effects

   .. tab:: Fragment-Focused

      Use **Fragment-Specific Analysis** (18 features) for:
      
      - Pharmacophore analysis and optimization
      - Core scaffold modification studies
      - Fragment-based drug design
      - Systematic R-group analysis

**Best Practices**

1. **Start Comprehensive**: Begin with LOFFI analysis for full characterization
2. **Focus Gradually**: Use element/fragment-specific analysis for targeted studies
3. **Validate Results**: Cross-check findings with chemical knowledge
4. **Interpret Carefully**: Consider descriptor correlations and interactions

**Common Applications**

- **Drug Discovery**: ADMET property prediction and optimization
- **Materials Science**: Property prediction for functional materials
- **Catalysis**: Active site characterization and design
- **Environmental Chemistry**: Fate and transport modeling

Integration with Machine Learning
---------------------------------

**Feature Engineering**

Surfacia descriptors are designed for optimal ML performance:

- **Numerical Stability**: All descriptors are numerically well-behaved
- **Appropriate Scaling**: Features span reasonable dynamic ranges
- **Low Redundancy**: Intelligent selection minimizes correlation
- **Chemical Relevance**: Each feature has clear chemical meaning

**Model Interpretability**

SHAP analysis with Surfacia descriptors provides:

- **Feature Importance**: Ranking of molecular properties by impact
- **Directional Effects**: Understanding of positive/negative contributions
- **Chemical Context**: Translation of mathematical results to chemical insights
- **Design Guidance**: Actionable recommendations for molecular optimization

**Performance Optimization**

- **Feature Selection**: Automated selection maintains predictive power
- **Dimensionality Reduction**: Intelligent reduction preserves chemical meaning
- **Cross-Validation**: Robust evaluation across molecular diversity
- **Transfer Learning**: Descriptors transfer well across chemical spaces

See Also
--------

- :doc:`size_and_shape` - Detailed size and shape descriptor reference
- :doc:`electronic_properties` - Electronic and quantum mechanical descriptors
- :doc:`surface_analysis` - Multi-scale surface analysis descriptors
- :doc:`../getting_started/basic_concepts` - Theoretical foundation
- :doc:`../commands/workflow` - How descriptors are calculated