utilities
=========

Surfacia provides a comprehensive set of utility commands for molecular visualization, batch processing, and error recovery. These tools complement the main analysis workflow and can be used independently for specific tasks.

.. toctree::
   :maxdepth: 2
   :caption: Utility Commands

   mol_drawer
   mol_viewer
   rerun_gaussian

Utility Commands Overview
--------------------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 🎨 mol-drawer
      :link: #mol-drawer
      :link-type: ref

      Generate 2D molecular structure visualizations from SMILES strings

   .. grid-item-card:: 🔍 mol-viewer
      :link: #mol-viewer
      :link-type: ref

      Interactive 3D molecular structure viewer for XYZ files

   .. grid-item-card:: 🔄 rerun-gaussian
      :link: #rerun-gaussian
      :link-type: ref

      Recover and rerun failed Gaussian quantum chemistry calculations

.. _mol-drawer:

mol-drawer
----------

Generate high-quality 2D molecular structure images from SMILES strings.

**Synopsis**

.. code-block:: bash

   surfacia mol-drawer [OPTIONS] -i INPUT_FILE

**Description**

The mol-drawer command creates publication-quality 2D molecular structure diagrams:

- **Batch Processing**: Handle multiple molecules simultaneously
- **Customizable Output**: Various image formats and sizes
- **Chemical Accuracy**: Proper stereochemistry and bond representation
- **Organized Output**: Systematic file naming and directory structure

**Options**

.. option:: -i, --input PATH

   Input CSV file with SMILES data. Must contain ``Sample Name`` and ``SMILES`` columns.

.. option:: -o, --output PATH

   Output directory for generated images.
   
   **Default**: ``molecular_structures/``

.. option:: --format TEXT

   Image format for output.
   
   **Options**: ``png``, ``svg``, ``pdf``
   
   **Default**: ``png``

.. option:: --size TEXT

   Image dimensions in pixels (WIDTHxHEIGHT).
   
   **Default**: ``300x300``

.. option:: --dpi INTEGER

   Resolution for raster images.
   
   **Default**: 300

**Examples**

.. code-block:: bash

   # Basic 2D structure generation
   surfacia mol-drawer -i molecules.csv -o structures/
   
   # High-resolution SVG output
   surfacia mol-drawer -i molecules.csv --format svg --size 600x600 -o high_res/
   
   # Publication-quality PDF
   surfacia mol-drawer -i molecules.csv --format pdf --dpi 600 -o publication/

**Input File Format**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "Sample Name", "Unique identifier", "caffeine"
   "SMILES", "Valid SMILES string", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

**Output Structure**

.. code-block:: text

   molecular_structures/
   ├── caffeine.png
   ├── aspirin.png
   ├── ibuprofen.png
   └── generation_log.txt

.. _mol-viewer:

mol-viewer
----------

Interactive 3D molecular structure viewer for examining optimized geometries.

**Synopsis**

.. code-block:: bash

   surfacia mol-viewer [OPTIONS] -i INPUT_FILE

**Description**

The mol-viewer command provides interactive 3D visualization:

- **3D Rendering**: High-quality molecular visualization
- **Interactive Controls**: Rotate, zoom, and pan
- **Multiple Formats**: Support for XYZ, PDB, and other formats
- **Property Display**: Show atomic charges, bond orders, etc.

**Options**

.. option:: -i, --input PATH

   Input molecular structure file (XYZ, PDB, etc.).

.. option:: --style TEXT

   Visualization style.
   
   **Options**: ``ball_stick``, ``space_fill``, ``wireframe``
   
   **Default**: ``ball_stick``

.. option:: --background TEXT

   Background color.
   
   **Default**: ``white``

.. option:: --port INTEGER

   Port for web viewer.
   
   **Default**: 8051

**Examples**

.. code-block:: bash

   # View single molecule
   surfacia mol-viewer -i caffeine.xyz
   
   # Custom styling
   surfacia mol-viewer -i molecule.xyz --style space_fill --background black
   
   # Custom port
   surfacia mol-viewer -i structure.xyz --port 9000

**Interactive Features**

- **Mouse Controls**: Left-click drag to rotate, scroll to zoom
- **Keyboard Shortcuts**: R (reset view), S (screenshot), H (help)
- **Property Panels**: Display molecular properties and atom information
- **Export Options**: Save images and coordinate files

.. _rerun-gaussian:

rerun-gaussian
--------------

Recover and rerun failed Gaussian quantum chemistry calculations.

**Synopsis**

.. code-block:: bash

   surfacia rerun-gaussian [OPTIONS] -i INPUT_FILE

**Description**

The rerun-gaussian command handles calculation failures:

- **Error Detection**: Automatically identify failure types
- **Smart Recovery**: Apply appropriate fixes for common issues
- **Batch Processing**: Handle multiple failed calculations
- **Progress Tracking**: Monitor rerun progress and success rates

**Options**

.. option:: -i, --input PATH

   Input CSV file with failed molecules or directory with failed calculations.

.. option:: --max-attempts INTEGER

   Maximum number of retry attempts per molecule.
   
   **Default**: 3

.. option:: --timeout INTEGER

   Timeout in seconds for each calculation.
   
   **Default**: 7200 (2 hours)

.. option:: --method TEXT

   Gaussian method to use for reruns.
   
   **Default**: ``B3LYP/6-31G(d)``

.. option:: --memory TEXT

   Memory allocation for Gaussian.
   
   **Default**: ``4GB``

**Examples**

.. code-block:: bash

   # Rerun failed calculations
   surfacia rerun-gaussian -i failed_molecules.csv
   
   # Custom parameters
   surfacia rerun-gaussian -i failures.csv --max-attempts 5 --timeout 10800
   
   # Different method
   surfacia rerun-gaussian -i failures.csv --method "M06-2X/6-311G(d,p)"

**Common Failure Types and Solutions**

.. csv-table::
   :header: "Error Type", "Automatic Fix", "Description"
   :widths: 30, 30, 40

   "SCF Convergence", "Add SCF=XQC", "Improve convergence"
   "Geometry Optimization", "Reduce step size", "More careful optimization"
   "Memory Error", "Increase memory", "Allocate more RAM"
   "Disk Space", "Clean temp files", "Free up disk space"

**Output Files**

.. code-block:: text

   rerun_results/
   ├── successful_reruns/          # Successfully completed calculations
   ├── still_failed/              # Still failing after max attempts
   ├── rerun_summary.csv          # Summary of all rerun attempts
   └── error_analysis.txt         # Detailed error analysis

Common Utility Workflows
------------------------

**Workflow 1: Structure Visualization Pipeline**

.. code-block:: bash

   # Generate 2D structures
   surfacia mol-drawer -i molecules.csv -o 2d_structures/
   
   # Run workflow to get 3D structures
   surfacia workflow -i molecules.csv --resume
   
   # View optimized 3D structures
   surfacia mol-viewer -i Surfacia_3.0_*/xyz_files/caffeine.xyz

**Workflow 2: Error Recovery Pipeline**

.. code-block:: bash

   # Initial workflow run
   surfacia workflow -i molecules.csv --resume
   
   # Check for failures and rerun
   surfacia rerun-gaussian -i failed_molecules.csv
   
   # Resume workflow with recovered calculations
   surfacia workflow -i molecules.csv --resume

**Workflow 3: Publication Figure Generation**

.. code-block:: bash

   # High-quality 2D structures for publication
   surfacia mol-drawer -i key_molecules.csv --format svg --size 800x800 --dpi 600 -o figures/
   
   # Generate 3D renderings
   for mol in key_molecules/*.xyz; do
       surfacia mol-viewer -i "$mol" --style ball_stick --background white
       # Save screenshot manually
   done

Integration with Main Workflow
------------------------------

**Pre-Workflow Visualization**

.. code-block:: bash

   # Visualize input molecules before analysis
   surfacia mol-drawer -i input_molecules.csv -o preview/

**Post-Workflow Analysis**

.. code-block:: bash

   # View optimized structures
   surfacia mol-viewer -i Surfacia_3.0_*/xyz_files/molecule.xyz
   
   # Handle any failures
   surfacia rerun-gaussian -i Surfacia_3.0_*/failed_calculations/

**Quality Control**

.. code-block:: bash

   # Check structure quality
   surfacia mol-viewer -i optimized_structures/*.xyz
   
   # Rerun problematic calculations
   surfacia rerun-gaussian -i problematic_molecules.csv --method "M06-2X/6-311G(d,p)"

Best Practices
--------------

**Molecular Visualization**

1. **Format Selection**: Use SVG for scalable graphics, PNG for web display
2. **Resolution**: Use high DPI (600+) for publication figures
3. **Batch Processing**: Process multiple molecules efficiently
4. **File Organization**: Use descriptive names and organized directories

**3D Viewing**

1. **Structure Quality**: Verify optimized geometries before analysis
2. **Comparison**: Compare input and optimized structures
3. **Property Analysis**: Use viewer to understand molecular features
4. **Documentation**: Save important views and annotations

**Error Recovery**

1. **Systematic Approach**: Analyze error patterns before rerunning
2. **Parameter Adjustment**: Modify method/basis set for difficult cases
3. **Resource Management**: Monitor memory and disk usage
4. **Documentation**: Keep records of successful recovery strategies

**Performance Tips**

1. **Parallel Processing**: Use multiple cores for batch operations
2. **Resource Monitoring**: Watch memory and disk usage
3. **Incremental Processing**: Process large datasets in chunks
4. **Backup Strategy**: Keep copies of important intermediate files

Troubleshooting
---------------

**Common Issues**

.. admonition:: Image generation fails
   :class: warning

   **Symptoms**: Empty or corrupted image files
   
   **Solutions**:
   
   - Verify SMILES validity
   - Check output directory permissions
   - Try different image format
   - Reduce image size if memory limited

.. admonition:: 3D viewer not loading
   :class: warning

   **Symptoms**: Blank viewer or connection errors
   
   **Solutions**:
   
   - Check if port is available
   - Try different port number
   - Verify XYZ file format
   - Check browser compatibility

.. admonition:: Rerun calculations still failing
   :class: warning

   **Symptoms**: Repeated failures after rerun attempts
   
   **Solutions**:
   
   - Try different quantum chemistry method
   - Increase memory allocation
   - Check molecular structure validity
   - Use simpler basis set

**Performance Issues**

.. code-block:: bash

   # For large datasets
   surfacia mol-drawer -i large_dataset.csv --format png --size 200x200
   
   # For memory-limited systems
   surfacia rerun-gaussian -i failures.csv --memory 2GB --timeout 3600

Advanced Usage
--------------

**Custom Scripting**

.. code-block:: bash

   #!/bin/bash
   # Automated structure generation and viewing
   
   INPUT_FILE="molecules.csv"
   
   # Generate 2D structures
   surfacia mol-drawer -i "$INPUT_FILE" -o 2d_structures/
   
   # Run workflow
   surfacia workflow -i "$INPUT_FILE" --resume
   
   # Handle failures
   if [ -f "failed_molecules.csv" ]; then
       surfacia rerun-gaussian -i failed_molecules.csv
       surfacia workflow -i "$INPUT_FILE" --resume
   fi
   
   # Generate final visualizations
   surfacia mol-drawer -i "$INPUT_FILE" --format svg --size 600x600 -o final_structures/

**Integration with External Tools**

.. code-block:: python

   # Python script for automated processing
   import subprocess
   import os
   
   def process_molecules(input_file):
       # Generate structures
       subprocess.run([
           "surfacia", "mol-drawer", 
           "-i", input_file, 
           "-o", "structures/"
       ])
       
       # Run analysis
       subprocess.run([
           "surfacia", "workflow",
           "-i", input_file,
           "--resume"
       ])
       
       # Check for failures and rerun
       if os.path.exists("failed_molecules.csv"):
           subprocess.run([
               "surfacia", "rerun-gaussian",
               "-i", "failed_molecules.csv"
           ])

See Also
--------

- :doc:`workflow` - Complete analysis pipeline
- :doc:`ml_analysis` - Machine learning analysis
- :doc:`shap_viz` - SHAP visualization
- :doc:`../getting_started/quick_start` - Quick start guide