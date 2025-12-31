Visualization Modules
======================

This section documents the visualization modules of Surfacia that provide interactive molecular visualization, SHAP analysis interfaces, and molecular drawing capabilities.

.. module:: surfacia.visualization.interactive_shap_viz

Interactive SHAP Visualization
----------------------------

.. automodule:: surfacia.visualization.interactive_shap_viz
   :members:
   :undoc-members:
   :show-inheritance:

The interactive SHAP visualization module provides a web-based interface for exploring molecular features, SHAP values, and 3D molecular structures with AI-powered analysis assistance.

Classes
~~~~~~~

MultiwfnPDBGenerator
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: surfacia.visualization.interactive_shap_viz.MultiwfnPDBGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Utility class for generating PDB files from Multiwfn calculations for visualization purposes.

**Key Methods:**

- ``create_leae_pdb_content()``: Create LEAE surface PDB content
- ``create_esp_pdb_content()``: Create ESP surface PDB content
- ``create_alie_pdb_content()``: Create ALIE surface PDB content
- ``generate_xyz_pdb_file()``: Generate XYZ PDB file from .fchk
- ``generate_surface_pdb_file()``: Generate surface PDB file
- ``generate_all_pdb_files()``: Generate all required PDB files

InteractiveSHAPAnalyzer
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: surfacia.visualization.interactive_shap_viz.InteractiveSHAPAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Main class for interactive SHAP visualization with 3D molecular viewer and AI assistant.

**Key Methods:**

- ``load_data()``: Load training and test data
- ``load_test_data()``: Load test set data
- ``check_and_generate_surface_files()``: Ensure surface PDB files exist
- ``get_initial_guidance()``: Get initial AI guidance for analysis
- ``call_deepseek_llm()``: Call LLM API for AI assistance
- ``create_features_shap_table()``: Create SHAP value table
- ``load_xyz_file()``: Load molecular coordinates
- ``load_surface_pdb_file()``: Load surface PDB file
- ``create_3d_molecule_viewer()``: Create 3D molecular viewer
- ``create_combined_surface_xyz_viewer()``: Create combined surface and XYZ viewer
- ``create_surfacia_mode_viewer()``: Create Surfacia mode molecular viewer
- ``create_feature_shap_plot()``: Create SHAP scatter plot
- ``create_dash_app()``: Create Dash web application
- ``run_app()``: Run the web application server

Functions
~~~~~~~~~

.. autofunction:: surfacia.visualization.interactive_shap_viz.interactive_shap_viz_main

.. autofunction:: surfacia.visualization.interactive_shap_viz.run_interactive_shap_viz

Examples
~~~~~~~~

**Basic SHAP visualization:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import InteractiveSHAPAnalyzer
   
   # Initialize with training data
   analyzer = InteractiveSHAPAnalyzer(
       csv_file_path="Training_Set_Detailed.csv",
       xyz_folder_path="./xyz_files"
   )
   
   # Run the web application
   analyzer.run_app(debug=False, port=8052)

**With test set and AI assistant:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import InteractiveSHAPAnalyzer
   
   analyzer = InteractiveSHAPAnalyzer(
       csv_file_path="Training_Set_Detailed.csv",
       xyz_folder_path="./xyz_files",
       test_csv_path="Test_Set_Detailed.csv",
       api_key="your_zhipuai_api_key"
   )
   
   analyzer.run_app(port=8080, host="localhost")

**Using convenience function:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import run_interactive_shap_viz
   
   run_interactive_shap_viz(
       csv_path="Training_Set_Detailed.csv",
       xyz_path="./xyz_files",
       test_csv_path="Test_Set_Detailed.csv",
       api_key="your_api_key",
       port=8052
   )

**Generate PDB files:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import MultiwfnPDBGenerator
   
   # Generate all PDB files for visualization
   success = MultiwfnPDBGenerator.generate_all_pdb_files(
       input_path="./calculations"
   )
   
   if success:
       print("All PDB files generated successfully")

**Generate specific surface PDB:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import MultiwfnPDBGenerator
   
   # Generate ESP surface PDB for a specific molecule
   MultiwfnPDBGenerator.generate_surface_pdb_file(
       fchk_file="molecule_000001.fchk",
       surface_type="ESP",
       output_dir="./pdb_files"
   )

Web Interface Features
~~~~~~~~~~~~~~~~~~~~~

The interactive web application provides:

**Dashboard:**
- Overview of analysis results
- Quick access to key statistics
- Navigation controls

**SHAP Explorer:**
- Interactive scatter plots
- Feature selection dropdown
- SHAP value distribution
- Test set overlay option

**3D Molecular Viewer:**
- Interactive 3D molecular structures
- Surface property isosurfaces
- Multiple display modes (Surfacia, Surface Only, XYZ Only)
- Adjustable point size and opacity
- Color range controls

**Data Tables:**
- Detailed numerical data
- Sortable columns
- Search functionality
- Export options

**AI Assistant:**
- Chat interface with LLM
- Image upload support
- Molecular analysis insights
- Feature interpretation

Display Modes
~~~~~~~~~~~~~

**Surfacia Mode:**
- Shows both surface and XYZ structures
- Combined 3D visualization
- Complete molecular picture

**Surface Only Mode:**
- Displays only surface properties
- Clear view of isosurfaces
- Focus on electronic properties

**XYZ Only Mode:**
- Shows only molecular skeleton
- Atomic coordinates
- Basic structure view

Surface Types
~~~~~~~~~~~~~~

Supported surface types for visualization:

- **ESP**: Electrostatic Potential
- **ALIE**: Average Local Ionization Energy
- **LEAE**: Local Electron Affinity Energy
- **XYZ**: Molecular coordinates (structure)

Customization Options
~~~~~~~~~~~~~~~~~~~~~

**Viewer Settings:**

.. code-block:: python

   analyzer.run_app(
       debug=True,           # Debug mode with auto-reload
       port=8052,           # Server port
       host='0.0.0.0'      # Server host
   )

**Surface Visualization Parameters:**

.. code-block:: python

   # When creating viewers
   viewer = analyzer.create_surfacia_mode_viewer(
       sample_name="000001",
       smiles="CCO",
       target_value=5.2,
       feature_value=1.5,
       shap_value=0.8,
       selected_feature="ESP_mean",
       surface_type="ESP",
       point_size=25,          # Adjust point size (10-50)
       opacity=100,            # Opacity percentage (0-100)
       height="600px"          # Viewer height
   )

**Color Range Controls:**

- Auto-range: Automatically determine color range
- Manual range: Set custom min/max values
- Reset: Restore default color mapping

AI Assistant Integration
~~~~~~~~~~~~~~~~~~~~~~~

The AI assistant provides intelligent analysis:

**Features:**
- Explain SHAP value patterns
- Suggest molecular optimizations
- Interpret surface properties
- Provide domain insights

**Usage:**

1. Navigate to AI Chat tab
2. Type questions or descriptions
3. Upload images of molecular structures (optional)
4. Receive AI-powered analysis

**API Requirements:**

- ZhipuAI API key required
- Secure key storage recommended
- Rate limiting may apply

**Example Interactions:**

- "Explain why this molecule has high ESP values"
- "Suggest modifications to improve activity"
- "What regions are most reactive?"
- "Interpret the SHAP value distribution"

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Resource Requirements:**

- CPU: 4+ cores recommended
- Memory: 8+ GB RAM
- Browser: Modern browser with WebGL support

**Loading Times:**

- Small datasets (<50 samples): <5 seconds
- Medium datasets (50-200 samples): 10-30 seconds
- Large datasets (200+ samples): 30-60 seconds

**Optimization Tips:**

1. Pre-generate PDB files for faster loading
2. Use test set option to reduce initial load
3. Enable browser caching
4. Use modern browsers (Chrome, Firefox, Edge)

.. module:: surfacia.visualization.mol_drawer

Molecular Drawer
----------------

.. automodule:: surfacia.visualization.mol_drawer
   :members:
   :undoc-members:
   :show-inheritance:

The molecular drawer module provides functions for generating 2D molecular structure images from SMILES strings using RDKit.

Functions
~~~~~~~~~

.. autofunction:: surfacia.visualization.mol_drawer.draw_molecules_from_csv

.. autofunction:: surfacia.visualization.mol_drawer.draw_single_molecule

.. autofunction:: surfacia.visualization.mol_drawer.batch_draw_molecules

Examples
~~~~~~~~

**Draw single molecule:**

.. code-block:: python

   from surfacia.visualization.mol_drawer import draw_single_molecule
   
   # Draw ethanol
   draw_single_molecule(
       smiles="CCO",
       output_path="ethanol.png",
       size=(800, 800)
   )

**Draw molecules from CSV:**

.. code-block:: python

   from surfacia.visualization.mol_drawer import draw_molecules_from_csv
   
   # Draw all molecules from CSV
   draw_molecules_from_csv(
       csv_file="molecules.csv",
       output_dir="molecule_images"
   )

**Batch drawing:**

.. code-block:: python

   from surfacia.visualization.mol_drawer import batch_draw_molecules
   
   # Draw multiple SMILES strings
   smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
   
   batch_draw_molecules(
       smiles_list=smiles_list,
       output_dir="batch_images",
       prefix="molecule"
   )

**High resolution drawing:**

.. code-block:: python

   from surfacia.visualization.mol_drawer import draw_single_molecule
   
   draw_single_molecule(
       smiles="complex_molecule",
       output_path="high_res.png",
       size=(1200, 1200)
   )

**Custom styling:**

The drawer uses RDKit's default styling but can be customized:

- 300 DPI resolution
- Clean bond rendering
- Atom labels
- Stereochemistry representation

**Output Format:**

- PNG format
- High quality (300 DPI)
- Transparent background
- Ready for publications

.. module:: surfacia.visualization.mol_viewer

Molecular Viewer
----------------

.. automodule:: surfacia.visualization.mol_viewer
   :members:
   :undoc-members:
   :show-inheritance:

The molecular viewer module provides interactive 3D molecular structure viewing capabilities using NGL Viewer.

Functions
~~~~~~~~~

.. autofunction:: surfacia.visualization.mol_viewer.read_xyz_file

.. autofunction:: surfacia.visualization.mol_viewer.view_molecule

.. autofunction:: surfacia.visualization.mol_viewer.see_xyz_interactive

Examples
~~~~~~~~

**View single molecule:**

.. code-block:: python

   from surfacia.visualization.mol_viewer import see_xyz_interactive
   
   # Launch interactive viewer
   see_xyz_interactive()

**View specific molecule:**

.. code-block:: python

   from surfacia.visualization.mol_viewer import view_molecule
   
   # View in existing Jupyter widget
   view_molecule("molecule_000001.xyz", view_widget)

**Read XYZ file:**

.. code-block:: python

   from surfacia.visualization.mol_viewer import read_xyz_file
   
   # Read molecular coordinates
   atoms, coords = read_xyz_file("molecule.xyz")
   
   # atoms: list of element symbols
   # coords: list of coordinate arrays

**Interactive Features:**

- Rotate: Click and drag
- Zoom: Scroll wheel
- Pan: Shift + click and drag
- Reset view: Double-click
- Change representation: Use built-in controls

**Supported Representations:**

- Ball and Stick
- Space Fill
- Ribbon (for proteins)
- Wireframe
- Licorice

.. module:: surfacia.visualization.surface_calc

Surface Calculation
-------------------

.. automodule:: surfacia.visualization.surface_calc
   :members:
   :undoc-members:
   :show-inheritance:

The surface calculation module provides functions for creating surface property content for Multiwfn analysis and generating PDB files for visualization.

Functions
~~~~~~~~~

.. autofunction:: surfacia.visualization.surface_calc.create_leae_content

.. autofunction:: surfacia.visualization.surface_calc.create_esp_content

.. autofunction:: surfacia.visualization.surface_calc.create_alie_content

.. autofunction:: surfacia.visualization.surface_calc.run_multiwfn_surface_calculations

.. autofunction:: surfacia.visualization.surface_calc.check_surface_files_completeness

.. autofunction:: surfacia.visualization.surface_calc.cleanup_temp_files

Examples
~~~~~~~~

**Run surface calculations:**

.. code-block:: python

   from surfacia.visualization.surface_calc import run_multiwfn_surface_calculations
   
   # Run surface analysis on all .fchk files
   run_multiwfn_surface_calculations(
       input_path="./calculations"
   )

**Check surface file completeness:**

.. code-block:: python

   from surfacia.visualization.surface_calc import check_surface_files_completeness
   
   # Verify all required surface files exist
   completeness = check_surface_files_completeness(
       input_path="./calculations"
   )
   
   if completeness:
       print("All surface files are complete")
   else:
       print("Some surface files are missing or incomplete")

**Create surface content manually:**

.. code-block:: python

   from surfacia.visualization.surface_calc import (
       create_esp_content,
       create_alie_content,
       create_leae_content
   )
   
   # Create ESP surface calculation input
   esp_content = create_esp_content()
   
   # Create ALIE surface calculation input
   alie_content = create_alie_content()
   
   # Create LEAE surface calculation input
   leae_content = create_leae_content("molecule_name")

**Cleanup temporary files:**

.. code-block:: python

   from surfacia.visualization.surface_calc import cleanup_temp_files
   
   # Remove temporary Multiwfn files
   cleanup_temp_files(input_path="./calculations")

Surface Types
~~~~~~~~~~~~~~

**ESP (Electrostatic Potential):**
- Measures electrostatic potential on molecular surface
- Important for understanding charge distribution
- Relevant for reactivity and binding

**ALIE (Average Local Ionization Energy):**
- Measures local ionization energy on surface
- Indicates sites susceptible to electrophilic attack
- Useful for predicting chemical reactivity

**LEAE (Local Electron Affinity Energy):**
- Measures local electron affinity on surface
- Indicates sites susceptible to nucleophilic attack
- Complementary to ALIE analysis

Integration and Workflow
========================

The visualization modules integrate seamlessly with the main workflow:

**Complete visualization pipeline:**

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   from surfacia.visualization.interactive_shap_viz import run_interactive_shap_viz
   
   # Step 1: Run complete workflow
   workflow = SurfaciaWorkflow("molecules.csv")
   workflow.run_full_workflow(
       max_features=5,
       test_samples="79,22,82"
   )
   
   # Step 2: Launch interactive visualization
   run_interactive_shap_viz(
       csv_path="Training_Set_Detailed.csv",
       xyz_path="./Surfacia_3.0_*/xyz_files",
       test_csv_path="Test_Set_Detailed.csv",
       api_key="your_api_key"
   )

**Generate molecular images:**

.. code-block:: python

   from surfacia.visualization.mol_drawer import draw_molecules_from_csv
   
   # Create 2D structure images
   draw_molecules_from_csv(
       csv_file="sample_mapping.csv",
       output_dir="molecule_images"
   )

**View 3D structures:**

.. code-block:: python

   from surfacia.visualization.mol_viewer import see_xyz_interactive
   
   # Launch interactive 3D viewer
   see_xyz_interactive()

Best Practices
==============

**1. Pre-generate PDB files:**

Before launching interactive visualization, generate all required PDB files:

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import MultiwfnPDBGenerator
   
   MultiwfnPDBGenerator.generate_all_pdb_files()

**2. Organize data properly:**

Ensure proper directory structure:

.. code-block:: text

   analysis_folder/
   ├── Training_Set_Detailed.csv
   ├── Test_Set_Detailed.csv (optional)
   └── xyz_files/
       ├── 000001.xyz
       ├── 000001_ESP.pdb
       ├── 000001_ALIE.pdb
       ├── 000001_LEAE.pdb
       ├── 000002.xyz
       └── ...

**3. Use appropriate port:**

Avoid conflicts with other services:

.. code-block:: python

   # Common ports to avoid: 80, 443, 3000, 5000, 8000
   analyzer.run_app(port=8052)  # Default, usually safe

**4. Secure API keys:**

Never hardcode API keys in scripts:

.. code-block:: python

   import os
   
   # Use environment variables
   api_key = os.environ.get('ZHIPUAI_API_KEY')
   
   analyzer = InteractiveSHAPAnalyzer(
       csv_file_path="data.csv",
       xyz_folder_path="./xyz",
       api_key=api_key
   )

**5. Optimize for performance:**

For large datasets:

- Use test set option
- Pre-generate all PDB files
- Enable browser caching
- Consider sampling for initial preview

**6. Interactive exploration tips:**

**For SHAP analysis:**
- Start with feature importance rankings
- Examine high-impact features first
- Use test set overlay for validation
- Compare training and test distributions

**For molecular viewing:**
- Try different display modes
- Adjust point size and opacity
- Use color range to highlight regions
- Zoom into specific molecular regions

**7. Export results:**

Save important visualizations:

- Export SHAP plots as PNG
- Save 3D viewer screenshots
- Download data tables as CSV
- Record AI insights in notes

**8. Combine visualization tools:**

Use multiple visualization approaches:

1. Interactive SHAP viewer for exploration
2. 2D molecular drawer for publications
3. 3D viewer for detailed structural analysis
4. Surface calculations for property mapping

**9. Document findings:**

Keep records of visualizations:

- Screenshot important observations
- Annotate SHAP patterns
- Note AI assistant insights
- Track feature correlations

**10. Troubleshooting:**

**If visualization is slow:**
- Reduce dataset size (use test set)
- Close other browser tabs
- Clear browser cache
- Use a faster computer

**If PDB files don't load:**
- Check file paths are correct
- Verify file permissions
- Ensure Multiwfn completed successfully
- Regenerate PDB files

**If AI assistant fails:**
- Verify API key is valid
- Check internet connection
- Review API rate limits
- Try simpler queries first

Advanced Usage
~~~~~~~~~~~~~~

**Custom SHAP visualizations:**

.. code-block:: python

   from surfacia.visualization.interactive_shap_viz import InteractiveSHAPAnalyzer
   
   analyzer = InteractiveSHAPAnalyzer(
       csv_file_path="Training_Set_Detailed.csv",
       xyz_folder_path="./xyz_files"
   )
   
   # Load data
   analyzer.load_data()
   
   # Create custom SHAP plot
   fig = analyzer.create_feature_shap_plot(
       selected_feature="ESP_mean",
       show_test_set=True
   )
   
   # Save plot
   fig.write_html("custom_shap_plot.html")

**Batch molecular drawing:**

.. code-block:: python

   from surfacia.visualization.mol_drawer import batch_draw_molecules
   
   import pandas as pd
   
   # Read SMILES from CSV
   df = pd.read_csv("molecules.csv")
   smiles_list = df['smiles'].tolist()
   
   # Draw all molecules
   batch_draw_molecules(
       smiles_list=smiles_list,
       output_dir="all_molecules",
       prefix="mol"
   )

**Custom viewer configuration:**

.. code-block:: python

   from surfacia.visualization.mol_viewer import see_xyz_interactive
   
   # Modify viewer settings before launching
   see_xyz_interactive()

**Surface analysis workflow:**

.. code-block:: python

   from surfacia.visualization.surface_calc import (
       run_multiwfn_surface_calculations,
       check_surface_files_completeness
   )
   
   # Run calculations
   run_multiwfn_surface_calculations("./calculations")
   
   # Verify completeness
   if check_surface_files_completeness("./calculations"):
       print("Surface analysis complete!")
   else:
       print("Some calculations failed")

These visualization modules provide comprehensive tools for exploring molecular structures, surface properties, and SHAP-based feature importance through interactive interfaces.
