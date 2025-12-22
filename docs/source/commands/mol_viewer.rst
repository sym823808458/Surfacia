Molecular Structure Viewer (mol-viewer)
=======================================

Advanced molecular structure visualization and analysis tool for examining molecular geometries and properties.

Overview
--------

The [`mol-viewer`](../commands/mol_viewer.rst:1) command provides a powerful 3D visualization interface for molecular structures, surfaces, and properties. This tool is essential for visual analysis, result interpretation, and presentation of molecular data.

Command Syntax
---------------

.. code-block:: bash

   surfacia mol-viewer [OPTIONS] --file STRUCTURE_FILE

Basic Usage Examples
--------------------

View Molecular Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View a basic molecular structure
   surfacia mol-viewer --file molecule.xyz
   
   # View with different representation
   surfacia mol-viewer --file molecule.xyz --style ball-stick
   
   # View multiple conformations
   surfacia mol-viewer --file conformers.xyz --multi-frame

Surface Visualization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View molecular surface
   surfacia mol-viewer --file molecule.wfn --surface
   
   # View surface with properties
   surfacia mol-viewer --file molecule.wfn --surface --property esp
   
   # Custom surface transparency
   surfacia mol-viewer --file molecule.wfn --surface --transparency 0.7

Property Mapping
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Map electrostatic potential
   surfacia mol-viewer --file molecule.wfn --property esp --color-scale rwb
   
   # Map electron density
   surfacia mol-viewer --file molecule.wfn --property density --iso-value 0.001
   
   # Multiple property comparison
   surfacia mol-viewer --file molecule.wfn --properties esp,density --split-view

Command-Line Options
---------------------

Input Options
~~~~~~~~~~~~~

``--file, -f FILE``
  **Required.** Molecular structure or wavefunction file to visualize
  
  - **Supported formats**: XYZ, SDF, MOL, PDB, WFN, CUBE
  - **Multiple files**: Use multiple ``--file`` flags
  - **Example**: ``--file caffeine.xyz``

``--format FORMAT``
  Explicitly specify input file format
  
  - **Auto-detection**: Usually not needed
  - **Override**: For ambiguous extensions
  - **Example**: ``--format xyz``

``--multi-frame``
  Enable multi-frame mode for trajectory files
  
  - **Animation controls**: Play, pause, step through frames
  - **Frame selection**: Jump to specific frames
  - **Example**: ``--multi-frame``

Display Options
~~~~~~~~~~~~~~~

``--style STYLE``
  Set molecular representation style
  
  - **Available styles**: wireframe, ball-stick, space-fill, ribbon
  - **Default**: ball-stick
  - **Example**: ``--style space-fill``

``--color-scheme SCHEME``
  Set atom coloring scheme
  
  - **Available schemes**: element, residue, chain, custom
  - **Default**: element (CPK colors)
  - **Example**: ``--color-scheme residue``

``--background COLOR``
  Set background color
  
  - **Formats**: name, hex, rgb
  - **Default**: white
  - **Example**: ``--background black``

``--size WIDTHxHEIGHT``
  Set viewer window size
  
  - **Format**: Width x Height in pixels
  - **Default**: 1024x768
  - **Example**: ``--size 1200x900``

Surface Options
~~~~~~~~~~~~~~~

``--surface``
  Enable molecular surface display
  
  - **Surface types**: Van der Waals, solvent-accessible, molecular
  - **Default**: Molecular surface
  - **Example**: ``--surface``

``--surface-type TYPE``
  Specify surface type
  
  - **Types**: vdw, sas, molecular, custom
  - **Default**: molecular
  - **Example**: ``--surface-type vdw``

``--transparency ALPHA``
  Set surface transparency
  
  - **Range**: 0.0 (opaque) to 1.0 (transparent)
  - **Default**: 0.3
  - **Example**: ``--transparency 0.5``

``--probe-radius RADIUS``
  Set probe radius for surface calculation
  
  - **Units**: Angstroms
  - **Default**: 1.4 (water)
  - **Example**: ``--probe-radius 1.2``

Property Visualization
~~~~~~~~~~~~~~~~~~~~~~

``--property PROP``
  Map property onto surface or structure
  
  - **Available properties**: esp, density, fukui, lie, hardness
  - **Data source**: From wavefunction file
  - **Example**: ``--property esp``

``--properties LIST``
  Display multiple properties
  
  - **Format**: Comma-separated list
  - **Split view**: Side-by-side comparison
  - **Example**: ``--properties esp,density,fukui``

``--color-scale SCALE``
  Set color scale for property mapping
  
  - **Available scales**: rwb, bwr, rainbow, viridis, plasma
  - **Default**: rwb (red-white-blue)
  - **Example**: ``--color-scale viridis``

``--iso-value VALUE``
  Set isosurface value for property display
  
  - **Units**: Property-dependent
  - **Multiple values**: Comma-separated
  - **Example**: ``--iso-value 0.001,0.01``

Comparison Options
~~~~~~~~~~~~~~~~~~

``--compare FILE``
  Compare with another structure
  
  - **Overlay mode**: Superimpose structures
  - **Difference analysis**: Highlight changes
  - **Example**: ``--compare optimized.xyz``

``--align``
  Enable structure alignment for comparison
  
  - **Algorithm**: Kabsch algorithm
  - **RMSD calculation**: Root-mean-square deviation
  - **Example**: ``--align``

``--split-view``
  Use split-screen view for comparisons
  
  - **Side-by-side**: Left and right panels
  - **Synchronized**: Linked rotation and zoom
  - **Example**: ``--split-view``

Output Options
~~~~~~~~~~~~~~

``--output, -o FILE``
  Save rendered image to file
  
  - **Formats**: PNG, JPG, SVG, PDF
  - **High resolution**: Vector formats for publications
  - **Example**: ``--output molecule_view.png``

``--resolution WIDTHxHEIGHT``
  Set output image resolution
  
  - **High-DPI**: For publication quality
  - **Default**: Window size
  - **Example**: ``--resolution 2400x1800``

``--dpi DPI``
  Set dots per inch for output
  
  - **Publication quality**: 300+ DPI
  - **Default**: 150
  - **Example**: ``--dpi 300``

Interactive Features
--------------------

Navigation Controls
~~~~~~~~~~~~~~~~~~~

**Mouse Controls**
  - **Left drag**: Rotate structure
  - **Right drag**: Translate (pan)
  - **Scroll wheel**: Zoom in/out
  - **Middle click**: Reset view

**Keyboard Shortcuts**
  - **R**: Reset view to default
  - **F**: Fit structure to window
  - **C**: Center structure
  - **S**: Toggle surface display
  - **P**: Toggle property mapping
  - **Space**: Play/pause animation

Selection and Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~

**Atom Selection**
  - **Click**: Select individual atoms
  - **Ctrl+Click**: Multiple selection
  - **Shift+Click**: Range selection
  - **Double-click**: Select residue/molecule

**Measurement Tools**
  - **Distance**: Click two atoms
  - **Angle**: Click three atoms
  - **Dihedral**: Click four atoms
  - **Clear**: Right-click to clear measurements

Property Analysis
~~~~~~~~~~~~~~~~~

**Interactive Probing**
  - **Hover**: Show property values
  - **Click**: Pin property display
  - **Color mapping**: Real-time updates
  - **Statistics**: Min, max, average values

**Contour Lines**
  - **Toggle**: Show/hide contours
  - **Levels**: Adjustable contour levels
  - **Labels**: Property value labels
  - **Export**: Save contour data

Visualization Styles
--------------------

Molecular Representations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Wireframe**
  - Bonds as lines
  - Minimal visual clutter
  - Fast rendering
  - Good for large systems

**Ball-and-Stick**
  - Atoms as spheres
  - Bonds as cylinders
  - Standard representation
  - Good balance of detail and clarity

**Space-Filling**
  - Van der Waals spheres
  - Shows molecular volume
  - Surface accessibility
  - Good for packing analysis

**Ribbon**
  - Protein secondary structure
  - Alpha helices and beta sheets
  - Cartoon representation
  - Good for macromolecules

Surface Representations
~~~~~~~~~~~~~~~~~~~~~~

**Solid Surface**
  - Opaque surface rendering
  - Clear boundary definition
  - Property mapping support
  - Good for shape analysis

**Transparent Surface**
  - See-through surface
  - Internal structure visible
  - Adjustable transparency
  - Good for cavity analysis

**Mesh Surface**
  - Wireframe surface
  - Topology visualization
  - Low memory usage
  - Good for large surfaces

**Contour Surface**
  - Isosurface rendering
  - Property-based surfaces
  - Multiple contour levels
  - Good for property analysis

File Format Support
-------------------

Structure Formats
~~~~~~~~~~~~~~~~~

**XYZ Files**
  - Cartesian coordinates
  - Multiple frames supported
  - Comment line information
  - Simple format

**SDF Files**
  - Structure-data format
  - Property information
  - Multiple molecules
  - Chemical database format

**PDB Files**
  - Protein Data Bank format
  - Biological macromolecules
  - Secondary structure
  - Experimental metadata

**MOL Files**
  - MDL Molfile format
  - Connection tables
  - Stereochemistry
  - Chemical structure

Wavefunction Formats
~~~~~~~~~~~~~~~~~~~

**WFN Files**
  - Gaussian wavefunction
  - Electron density
  - Molecular orbitals
  - Property calculation

**CUBE Files**
  - Volumetric data
  - Property grids
  - Isosurface generation
  - Visualization ready

**FCHK Files**
  - Formatted checkpoint
  - Gaussian output
  - Complete wavefunction
  - Property extraction

Advanced Features
-----------------

Animation and Dynamics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View molecular dynamics trajectory
   surfacia mol-viewer --file trajectory.xyz --multi-frame --animate
   
   # Control animation speed
   surfacia mol-viewer --file trajectory.xyz --fps 10
   
   # Export animation
   surfacia mol-viewer --file trajectory.xyz --export-animation movie.mp4

Property Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Analyze property statistics
   surfacia mol-viewer --file molecule.wfn --property esp --statistics
   
   # Export property data
   surfacia mol-viewer --file molecule.wfn --property esp --export-data esp_data.csv
   
   # Custom property mapping
   surfacia mol-viewer --file molecule.wfn --custom-property custom_data.txt

Scripting Interface
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Python API for automated visualization
   from surfacia.tools import MolViewer
   
   viewer = MolViewer()
   viewer.load_structure('molecule.xyz')
   viewer.add_surface(transparency=0.5)
   viewer.map_property('esp', color_scale='rwb')
   viewer.save_image('molecule_esp.png', dpi=300)

Integration with Workflow
-------------------------

Result Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View workflow results
   surfacia mol-viewer --file results/molecule_surface.wfn --surface --property esp
   
   # Compare before and after optimization
   surfacia mol-viewer --file original.xyz --compare results/optimized.xyz --align

Quality Control
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check structure quality
   surfacia mol-viewer --file molecule.xyz --validate --measurements
   
   # Verify surface generation
   surfacia mol-viewer --file molecule.wfn --surface --quality-check

Presentation
~~~~~~~~~~~~

.. code-block:: bash

   # Generate publication-quality images
   surfacia mol-viewer --file molecule.wfn --surface --property esp \
                      --output figure.png --resolution 2400x1800 --dpi 300
   
   # Create multi-panel figure
   surfacia mol-viewer --file molecule.wfn --properties esp,density,fukui \
                      --split-view --output multi_panel.png

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**File Loading Problems**
  - Check file format and extension
  - Verify file integrity
  - Try different format specification
  - Check file permissions

**Performance Issues**
  - Reduce surface resolution
  - Use simpler representations
  - Close other applications
  - Check system memory

**Display Problems**
  - Update graphics drivers
  - Try different rendering options
  - Adjust window size
  - Check OpenGL support

**Property Mapping Issues**
  - Verify wavefunction file completeness
  - Check property availability
  - Try different color scales
  - Validate property ranges

Best Practices
~~~~~~~~~~~~~~

1. **Use appropriate representations** for different analysis types
2. **Save high-resolution images** for publications
3. **Validate structures** before detailed analysis
4. **Use consistent color schemes** for comparisons
5. **Document visualization settings** for reproducibility

See Also
--------

- :doc:`mol_drawer` - Molecular structure drawing
- :doc:`workflow` - Complete analysis pipeline
- :doc:`shap_viz` - SHAP visualization
- :doc:`../getting_started/quick_start` - Getting started guide
- :doc:`../examples/index` - Usage examples