Molecular Structure Drawing (mol-drawer)
========================================

Interactive molecular structure drawing and editing tool for creating and modifying molecular inputs.

Overview
--------

The [`mol-drawer`](../commands/mol_drawer.rst:1) command provides an interactive graphical interface for drawing, editing, and visualizing molecular structures. This tool is essential for creating input files, modifying existing structures, and preparing molecules for analysis.

Command Syntax
---------------

.. code-block:: bash

   surfacia mol-drawer [OPTIONS]

Basic Usage Examples
--------------------

Launch Interactive Drawing Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start the molecular drawing interface
   surfacia mol-drawer
   
   # Launch with a specific molecule loaded
   surfacia mol-drawer --input benzene.xyz
   
   # Start with a template molecule
   surfacia mol-drawer --template benzene

Save and Export Options
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Draw molecule and save to specific format
   surfacia mol-drawer --output molecule.xyz
   
   # Export to multiple formats
   surfacia mol-drawer --output molecule --formats xyz,sdf,mol
   
   # Save with specific naming convention
   surfacia mol-drawer --output-dir structures/ --prefix drawn_

Command-Line Options
---------------------

Input Options
~~~~~~~~~~~~~

``--input, -i FILE``
  Load an existing molecular structure file
  
  - **Supported formats**: XYZ, SDF, MOL, PDB
  - **Default**: Start with empty canvas
  - **Example**: ``--input caffeine.sdf``

``--template MOLECULE``
  Start with a predefined molecular template
  
  - **Available templates**: benzene, cyclohexane, water, methane
  - **Custom templates**: User-defined template library
  - **Example**: ``--template benzene``

Output Options
~~~~~~~~~~~~~~

``--output, -o FILE``
  Specify output file for the drawn structure
  
  - **Format detection**: Based on file extension
  - **Default**: Interactive save dialog
  - **Example**: ``--output new_molecule.xyz``

``--output-dir DIR``
  Directory for saving output files
  
  - **Default**: Current working directory
  - **Auto-creation**: Creates directory if it doesn't exist
  - **Example**: ``--output-dir molecules/``

``--formats LIST``
  Comma-separated list of export formats
  
  - **Available formats**: xyz, sdf, mol, pdb, mol2
  - **Multiple exports**: Save in multiple formats simultaneously
  - **Example**: ``--formats xyz,sdf,mol``

Display Options
~~~~~~~~~~~~~~~

``--theme THEME``
  Set the visual theme for the interface
  
  - **Available themes**: light, dark, high-contrast
  - **Default**: System theme
  - **Example**: ``--theme dark``

``--size WIDTHxHEIGHT``
  Set the window size for the drawing interface
  
  - **Format**: Width x Height in pixels
  - **Default**: 1024x768
  - **Example**: ``--size 1200x900``

``--fullscreen``
  Launch in fullscreen mode
  
  - **Toggle**: F11 key during use
  - **Useful for**: Detailed molecular editing
  - **Example**: ``--fullscreen``

Advanced Options
~~~~~~~~~~~~~~~~

``--precision DIGITS``
  Set coordinate precision for output files
  
  - **Range**: 1-8 decimal places
  - **Default**: 6
  - **Example**: ``--precision 4``

``--validate``
  Enable structure validation during drawing
  
  - **Checks**: Bond lengths, angles, stereochemistry
  - **Warnings**: Real-time feedback on unusual structures
  - **Example**: ``--validate``

``--auto-save INTERVAL``
  Enable automatic saving at specified intervals
  
  - **Units**: Minutes
  - **Default**: Disabled
  - **Example**: ``--auto-save 5``

Interface Features
------------------

Drawing Tools
~~~~~~~~~~~~~

**Atom Tools**
  - Add atoms by clicking
  - Change atom types with keyboard shortcuts
  - Delete atoms with right-click or Delete key
  - Move atoms by dragging

**Bond Tools**
  - Create bonds by clicking between atoms
  - Change bond orders (single, double, triple)
  - Delete bonds with right-click
  - Automatic bond length optimization

**Structure Tools**
  - Add common ring systems
  - Insert functional groups
  - Fragment library access
  - Template insertion

**Selection Tools**
  - Select individual atoms or bonds
  - Rectangle selection for multiple atoms
  - Select all (Ctrl+A)
  - Invert selection

Editing Features
~~~~~~~~~~~~~~~~

**Geometry Optimization**
  - Real-time structure cleanup
  - Force field-based optimization
  - Constraint-based positioning
  - Energy minimization

**Stereochemistry**
  - Wedge and dash bond notation
  - Chiral center identification
  - R/S configuration assignment
  - Stereoisomer generation

**Measurement Tools**
  - Distance measurements
  - Angle measurements
  - Dihedral angle display
  - Real-time geometry feedback

**Validation**
  - Structure connectivity check
  - Valence validation
  - Unusual geometry warnings
  - Chemical reasonableness assessment

Keyboard Shortcuts
------------------

Drawing Shortcuts
~~~~~~~~~~~~~~~~~

- **C**: Carbon atom
- **N**: Nitrogen atom
- **O**: Oxygen atom
- **S**: Sulfur atom
- **P**: Phosphorus atom
- **F**: Fluorine atom
- **Cl**: Chlorine atom
- **Br**: Bromine atom
- **I**: Iodine atom
- **H**: Hydrogen atom

Bond Shortcuts
~~~~~~~~~~~~~~

- **1**: Single bond
- **2**: Double bond
- **3**: Triple bond
- **4**: Aromatic bond
- **W**: Wedge bond (up)
- **D**: Dash bond (down)

General Shortcuts
~~~~~~~~~~~~~~~~~

- **Ctrl+N**: New molecule
- **Ctrl+O**: Open file
- **Ctrl+S**: Save file
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **Ctrl+A**: Select all
- **Delete**: Delete selected
- **Escape**: Clear selection

File Format Support
-------------------

Input Formats
~~~~~~~~~~~~~

**XYZ Files**
  - Cartesian coordinates
  - Element symbols
  - Comment lines supported
  - Multiple conformations

**SDF Files**
  - Structure-data format
  - Property data included
  - Multiple molecules
  - Stereochemistry information

**MOL Files**
  - MDL Molfile format
  - Bond connectivity
  - Atom properties
  - Charge information

**PDB Files**
  - Protein Data Bank format
  - Biological macromolecules
  - Coordinate records
  - Metadata support

Output Formats
~~~~~~~~~~~~~~

All input formats plus:

**MOL2 Files**
  - Tripos format
  - Atom types
  - Partial charges
  - Substructure information

**SMILES Strings**
  - Simplified notation
  - Canonical SMILES
  - Stereochemistry encoding
  - Fragment representation

Integration with Workflow
-------------------------

Preparation for Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Draw molecule and prepare for workflow
   surfacia mol-drawer --output molecule.xyz
   
   # Run complete analysis pipeline
   surfacia workflow --input molecule.xyz --output results/

Structure Modification
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Load existing structure for modification
   surfacia mol-drawer --input original.xyz --output modified.xyz
   
   # Compare original and modified structures
   surfacia mol-viewer --file original.xyz --compare modified.xyz

Template Creation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create custom template
   surfacia mol-drawer --output template.xyz
   
   # Use template for multiple molecules
   surfacia mol-drawer --template custom --output series_1.xyz

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Structure Not Displaying**
  - Check file format compatibility
  - Verify coordinate validity
  - Try different input format
  - Check for file corruption

**Slow Performance**
  - Reduce structure complexity
  - Disable real-time optimization
  - Close other applications
  - Check system resources

**Export Problems**
  - Verify output directory permissions
  - Check available disk space
  - Validate structure before export
  - Try different output format

**Interface Issues**
  - Update graphics drivers
  - Try different theme
  - Adjust window size
  - Restart application

Best Practices
~~~~~~~~~~~~~~

1. **Save frequently** during complex drawing sessions
2. **Validate structures** before using in calculations
3. **Use templates** for common molecular frameworks
4. **Check stereochemistry** for chiral molecules
5. **Optimize geometry** before exporting

Advanced Features
-----------------

Scripting Interface
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Python API for automated drawing
   from surfacia.tools import MolDrawer
   
   drawer = MolDrawer()
   drawer.load_template('benzene')
   drawer.add_substituent('methyl', position=1)
   drawer.save('toluene.xyz')

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Process multiple templates
   for template in benzene cyclohexane pyridine; do
       surfacia mol-drawer --template $template --output ${template}.xyz
   done

Custom Templates
~~~~~~~~~~~~~~~~

Create and manage custom molecular templates:

- Template library management
- User-defined fragments
- Parameterized structures
- Template sharing and import

Integration with External Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ChemDraw**: Import/export compatibility
- **Avogadro**: Structure exchange
- **OpenBabel**: Format conversion
- **RDKit**: Chemical informatics integration

See Also
--------

- :doc:`mol_viewer` - Molecular structure visualization
- :doc:`workflow` - Complete analysis pipeline
- :doc:`../getting_started/quick_start` - Getting started guide
- :doc:`../examples/index` - Usage examples