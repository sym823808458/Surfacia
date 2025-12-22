shap-viz
========

The ``shap-viz`` command creates interactive SHAP (SHapley Additive exPlanations) visualizations with AI-powered explanations for interpretable machine learning analysis. This command transforms complex SHAP values into intuitive, chemically meaningful insights.

Synopsis
--------

.. code-block:: bash

   surfacia shap-viz [OPTIONS] -i INPUT_FILE

Description
-----------

The shap-viz command provides comprehensive interpretability analysis through:

- **Interactive SHAP Visualizations**: Web-based dashboard for exploring SHAP values
- **AI-Powered Explanations**: Natural language interpretation of molecular features
- **Chemical Context**: Descriptor knowledge base for chemical understanding
- **Multiple Plot Types**: Summary plots, waterfall plots, force plots, and more
- **Export Capabilities**: Save results as images, reports, or data files

This command is essential for understanding:

- **Why** predictions were made
- **Which** molecular features drive activity
- **How** structural modifications would affect properties
- **Where** to focus synthetic efforts

.. mermaid::

   graph TD
       A[SHAP Values Input] --> B[Interactive Dashboard]
       B --> C[Summary Plots]
       B --> D[Individual Explanations]
       B --> E[Feature Analysis]
       C --> F[AI Interpretation]
       D --> F
       E --> F
       F --> G[Chemical Insights]

Options
-------

**Required Parameters**

.. option:: -i, --input PATH

   Input CSV file containing SHAP values and molecular data. Must include:
   
   - ``Sample Name``: Unique identifier for each molecule
   - SHAP value columns: Feature contributions to predictions
   - Optional: Actual and predicted values

**AI Integration**

.. option:: --api-key TEXT

   ZhipuAI API key for AI-powered explanations. Can also be set via ``ZHIPUAI_API_KEY`` environment variable.

**Server Configuration**

.. option:: --host TEXT

   Host address for the visualization server.
   
   **Default**: ``localhost``

.. option:: --port INTEGER

   Port number for the visualization server.
   
   **Default**: 8050

.. option:: --debug

   Enable debug mode for development.

**Analysis Options**

.. option:: --top-features INTEGER

   Number of top features to display in summary plots.
   
   **Default**: 20

.. option:: --sample-limit INTEGER

   Maximum number of samples to include in interactive plots.
   
   **Default**: 100

**Export Options**

.. option:: --export-format TEXT

   Format for exporting visualizations.
   
   **Options**:
   
   - ``html`` - Interactive HTML dashboard (default)
   - ``png`` - Static PNG images
   - ``pdf`` - PDF report
   - ``json`` - Raw data export

.. option:: -o, --output PATH

   Output directory for exported files.
   
   **Default**: ``SHAP_Analysis_YYYYMMDD_HHMMSS/``

Examples
--------

**Basic SHAP Visualization**

.. code-block:: bash

   # Launch interactive SHAP dashboard
   surfacia shap-viz -i training_data_with_shap.csv --api-key YOUR_API_KEY

**Custom Server Configuration**

.. code-block:: bash

   # Run on specific host and port
   surfacia shap-viz -i shap_results.csv --host 0.0.0.0 --port 8080 --api-key YOUR_KEY

**Export Static Visualizations**

.. code-block:: bash

   # Export as PNG images
   surfacia shap-viz -i shap_data.csv --export-format png -o shap_plots/

**Limited Feature Analysis**

.. code-block:: bash

   # Focus on top 10 features with 50 samples
   surfacia shap-viz -i shap_data.csv --top-features 10 --sample-limit 50 --api-key YOUR_KEY

**Development Mode**

.. code-block:: bash

   # Enable debug mode for development
   surfacia shap-viz -i shap_data.csv --debug --api-key YOUR_KEY

Input File Format
-----------------

The input file should contain SHAP values and related data:

**Required Columns**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "Sample Name", "Unique identifier", "caffeine"

**SHAP Value Columns**

SHAP values for each feature (named as ``shap_FeatureName``):

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 30, 50, 20

   "shap_HOMO", "SHAP value for HOMO", "-0.15"
   "shap_LUMO", "SHAP value for LUMO", "0.23"
   "shap_ALIE_min", "SHAP value for ALIE_min", "0.08"
   "shap_ESP_max", "SHAP value for ESP_max", "-0.12"

**Optional Columns**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "Actual", "Actual target values", "1.23"
   "Predicted", "Predicted values", "1.18"
   "SMILES", "Molecular SMILES", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

**Example Input File**

.. code-block:: text

   Sample Name,shap_HOMO,shap_LUMO,shap_ALIE_min,shap_ESP_max,Actual,Predicted,SMILES
   caffeine,-0.15,0.23,0.08,-0.12,-0.07,-0.05,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   aspirin,0.12,-0.18,-0.05,0.09,1.19,1.15,CC(=O)OC1=CC=CC=C1C(=O)O
   ibuprofen,0.28,0.15,0.12,0.18,3.97,3.89,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O

Interactive Dashboard Features
-------------------------------

**Main Dashboard Components**

1. **Summary Plot**: Overview of feature importance across all samples
2. **Waterfall Plot**: Detailed breakdown for individual molecules
3. **Force Plot**: Visual representation of feature contributions
4. **Partial Dependence**: Feature effect analysis
5. **Feature Interaction**: Two-way feature interactions

**AI Assistant Panel**

- **Initial Guidance**: Automated analysis overview
- **Feature Explanations**: Chemical meaning of important features
- **Pattern Recognition**: Identification of molecular patterns
- **Design Suggestions**: Recommendations for molecular optimization

**Interactive Controls**

- **Sample Selection**: Choose specific molecules for detailed analysis
- **Feature Filtering**: Focus on subsets of molecular features
- **Plot Customization**: Adjust colors, scales, and display options
- **Export Functions**: Save plots and generate reports

Visualization Types
--------------------

**Summary Plot**

.. code-block:: python

   # Shows feature importance across all samples
   # - Features ranked by mean absolute SHAP value
   # - Color indicates feature value (high/low)
   # - Horizontal spread shows impact distribution

**Waterfall Plot**

.. code-block:: python

   # Individual sample explanation
   # - Shows how each feature contributes to prediction
   # - Starts from base value, adds/subtracts contributions
   # - Final bar shows predicted value

**Force Plot**

.. code-block:: python

   # Visual force diagram
   # - Features pushing prediction higher (red)
   # - Features pushing prediction lower (blue)
   # - Base value in center

**Partial Dependence Plot**

.. code-block:: python

   # Shows feature effect across value range
   # - X-axis: feature values
   # - Y-axis: SHAP values (marginal contribution)
   # - Reveals non-linear relationships

**Feature Interaction Plot**

.. code-block:: python

   # Two-way feature interactions
   # - Heatmap showing interaction strength
   # - Scatter plots for specific pairs
   # - Identifies synergistic effects

AI-Powered Explanations
-----------------------

**Descriptor Knowledge Base**

The system includes comprehensive knowledge about molecular descriptors:

.. code-block:: python

   descriptor_knowledge = {
       "HOMO": {
           "name": "Highest Occupied Molecular Orbital",
           "meaning": "Electron-donating ability",
           "interpretation": "Lower values indicate stronger electron donors"
       },
       "ALIE_min": {
           "name": "Minimum Average Local Ionization Energy",
           "meaning": "Most nucleophilic site",
           "interpretation": "Lower values indicate stronger nucleophilic character"
       }
   }

**Automated Analysis**

The AI assistant provides:

1. **Feature Ranking**: Identification of most important molecular features
2. **Chemical Interpretation**: Translation of SHAP values to chemical meaning
3. **Pattern Recognition**: Detection of molecular patterns and trends
4. **Design Insights**: Suggestions for molecular optimization

**Natural Language Explanations**

Example AI-generated explanation:

.. code-block:: text

   "The model prediction for caffeine is primarily driven by its electronic 
   properties. The HOMO energy (-0.25 a.u.) contributes negatively to LogP, 
   consistent with its electron-rich aromatic system. The minimum ALIE value 
   (8.5 eV) indicates strong nucleophilic sites around the nitrogen atoms, 
   which decrease lipophilicity. To increase LogP, consider reducing the 
   number of hydrogen bond acceptors or adding hydrophobic substituents."

Export and Reporting
---------------------

**HTML Export**

.. code-block:: bash

   # Generate standalone HTML report
   surfacia shap-viz -i data.csv --export-format html -o report/

**Static Images**

.. code-block:: bash

   # Export all plots as PNG images
   surfacia shap-viz -i data.csv --export-format png -o images/

**PDF Report**

.. code-block:: bash

   # Generate comprehensive PDF report
   surfacia shap-viz -i data.csv --export-format pdf -o report.pdf

**Data Export**

.. code-block:: bash

   # Export processed data as JSON
   surfacia shap-viz -i data.csv --export-format json -o data_export/

Output Files
------------

**Interactive Dashboard**

.. code-block:: text

   SHAP_Analysis_20241201_143022/
   ├── dashboard.html                  # Interactive web dashboard
   ├── assets/                         # Dashboard assets
   │   ├── style.css
   │   └── script.js
   └── data/
       └── processed_shap_data.json    # Processed SHAP data

**Static Exports**

.. code-block:: text

   ├── plots/
   │   ├── summary_plot.png            # Feature importance summary
   │   ├── waterfall_plots/            # Individual explanations
   │   │   ├── caffeine_waterfall.png
   │   │   └── aspirin_waterfall.png
   │   ├── force_plots/                # Force diagrams
   │   └── interaction_plots/          # Feature interactions
   └── reports/
       ├── analysis_summary.txt        # AI-generated summary
       └── feature_explanations.txt    # Detailed feature analysis

Best Practices
--------------

**Data Preparation**

1. **Quality SHAP Values**: Ensure SHAP values are properly calculated
2. **Feature Names**: Use descriptive, consistent feature names
3. **Sample Selection**: Include representative molecules in analysis
4. **Data Validation**: Check for missing or invalid SHAP values

**Visualization Strategy**

1. **Start with Summary**: Begin with overall feature importance
2. **Drill Down**: Examine individual molecules of interest
3. **Compare Samples**: Look for patterns across similar molecules
4. **Validate Insights**: Cross-check with chemical knowledge

**AI Assistant Usage**

1. **Provide Context**: Include molecular structures (SMILES) when possible
2. **Ask Specific Questions**: Focus on particular features or patterns
3. **Validate Explanations**: Check AI insights against domain knowledge
4. **Iterate Analysis**: Use AI suggestions to guide further investigation

**Interpretation Guidelines**

1. **Chemical Meaning**: Always connect SHAP values to chemical properties
2. **Magnitude Matters**: Focus on features with large absolute SHAP values
3. **Sign Interpretation**: Understand positive vs negative contributions
4. **Context Dependency**: Consider feature interactions and correlations

Troubleshooting
---------------

**Common Issues**

.. admonition:: Dashboard not loading
   :class: warning

   **Symptoms**: Blank page or connection errors
   
   **Solutions**:
   
   - Check if port is available
   - Try different port: ``--port 8051``
   - Verify firewall settings
   - Use ``--host 0.0.0.0`` for network access

.. admonition:: AI explanations not working
   :class: warning

   **Symptoms**: No AI responses or error messages
   
   **Solutions**:
   
   - Verify API key is correct
   - Check internet connection
   - Set environment variable: ``export ZHIPUAI_API_KEY=your_key``
   - Try without AI features first

.. admonition:: Memory issues with large datasets
   :class: warning

   **Symptoms**: Slow performance or crashes
   
   **Solutions**:
   
   - Reduce sample limit: ``--sample-limit 50``
   - Limit features: ``--top-features 10``
   - Export static plots instead of interactive dashboard
   - Process data in smaller chunks

**Performance Optimization**

.. code-block:: bash

   # For large datasets
   surfacia shap-viz -i large_data.csv --sample-limit 100 --top-features 15

   # For slow connections
   surfacia shap-viz -i data.csv --export-format png  # Static export

   # For development
   surfacia shap-viz -i data.csv --debug --sample-limit 20

Integration with Other Commands
--------------------------------

**From ML Analysis**

.. code-block:: bash

   # Use ML analysis results
   surfacia shap-viz -i ML_Analysis_*/predictions.csv --api-key YOUR_KEY

**From Workflow**

.. code-block:: bash

   # Use workflow SHAP results
   surfacia shap-viz -i Surfacia_3.0_*/Training_Set_Detailed*.csv --api-key YOUR_KEY

**Iterative Analysis**

.. code-block:: bash

   # Export for further analysis
   surfacia shap-viz -i data.csv --export-format json -o exported_data/
   
   # Use exported data for custom analysis
   python custom_analysis.py exported_data/processed_shap_data.json

Advanced Features
------------------

**Custom Styling**

Modify dashboard appearance by editing CSS files in the assets directory.

**API Integration**

The dashboard provides REST API endpoints for programmatic access:

.. code-block:: python

   # Get SHAP data for specific sample
   GET /api/shap/{sample_name}
   
   # Get feature importance ranking
   GET /api/features/importance
   
   # Get AI explanation for feature
   POST /api/explain/feature

**Batch Processing**

.. code-block:: bash

   # Process multiple files
   for file in *.csv; do
       surfacia shap-viz -i "$file" --export-format png -o "plots_${file%.csv}/"
   done

See Also
--------

- :doc:`workflow` - Complete analysis pipeline
- :doc:`ml_analysis` - Machine learning analysis
- :doc:`utilities` - Supporting tools
- :doc:`../getting_started/basic_concepts` - SHAP theory and concepts