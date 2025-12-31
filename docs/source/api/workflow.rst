Workflow Command Reference
==========================

The ``workflow`` command is the main entry point for running complete Surfacia analysis pipelines. It orchestrates the entire computational workflow from SMILES input to interactive SHAP visualization.

.. important::
   The workflow command manages all intermediate steps automatically, ensuring proper file handling and data flow between different computational stages.

Basic Usage
------------

Run a complete workflow with default settings:

.. code-block:: bash

   surfacia workflow -i input.csv

Run with custom parameters:

.. code-block:: bash

   surfacia workflow -i input.csv --max-features 5 --epoch 128 --test-samples "1,2,3"

Resume an interrupted workflow:

.. code-block:: bash

   surfacia workflow --resume -i input.csv --test-samples "79,22,82"

Command Line Arguments
----------------------

Required Arguments
~~~~~~~~~~~~~~~~~~~

``-i, --input PATH``
    Input CSV file containing molecular data with SMILES strings and target values.
    
    **Format**: CSV with columns for SMILES strings and experimental values
    **Example**: ``-i /home/data/molecules.csv``
    **Note**: The file should contain at least 10 molecules for meaningful ML analysis

Optional Arguments
~~~~~~~~~~~~~~~~~~

``--resume``
    Resume a previously interrupted workflow from the last successful step.
    
    **Use case**: When workflow was interrupted due to system failure or manual stop
    **Behavior**: Automatically detects the last completed step and continues from there

``--extract-mode INT``
    Molecular descriptor extraction mode.
    
    **Choices**:
    - ``1``: Sulfur atom properties only
    - ``2``: Element-specific properties
    - ``3``: Complete LOFFI analysis (default)
    
    **Example**: ``--extract-mode 1`` for sulfur-focused analysis

``--extract-element TEXT``
    Target element for extraction when using mode 1.
    
    **Default**: ``"S"``
    **Supported**: Any chemical element symbol
    **Example**: ``--extract-element "O"`` for oxygen analysis

``--max-features INT``
    Maximum number of features to select in machine learning models.
    
    **Default**: ``5``
    **Range**: 1-50
    **Impact**: Controls model complexity and interpretability
    **Example**: ``--max-features 10`` for more complex models

``--stepreg-runs INT``
    Number of stepwise regression iterations.
    
    **Default**: ``3``
    **Range**: 1-10
    **Purpose**: Refines feature selection through iterative improvement
    **Example**: ``--stepreg-runs 5`` for more thorough feature selection

``--epoch INT``
    Number of training epochs for neural network models.
    
    **Default**: ``64``
    **Range**: 10-1000
    **Trade-off**: Higher values may improve accuracy but increase training time
    **Example**: ``--epoch 128`` for better convergence

``--test-samples TEXT``
    Comma-separated list of samples to hold out for testing.
    
    **Format**: Index numbers or sample identifiers
    **Example**: ``--test-samples "1,2,3,25,11,20,31"``
    **Note**: These samples will not be used in model training

``--port INT``
    Port number for the interactive visualization dashboard.
    
    **Default**: ``8052``
    **Range**: 1024-65535
    **Example**: ``--port 8080`` for custom port configuration

``--api-key TEXT``
    API key for external visualization services.
    
    **Security**: **Never share your API key publicly**
    **Example**: ``--api-key "your-api-key-here"``
    **Alternative**: Set via environment variable ``SURFACIA_API_KEY``

Computational Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

``--gaussian-keywords TEXT``
    Gaussian calculation method and basis set.
    
    **Default**: ``"# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3"``
    **Common choices**:
    - ``"# B3LYP/6-31G(d)"``: DFT with standard basis
    - ``"# MP2/cc-pVTZ"``: Post-Hartree-Fock method
    - ``"# HF/6-311++G**"``: Hartree-Fock with diffuse functions
    
``--nproc INT``
    Number of CPU cores for parallel computations.
    
    **Default**: ``32``
    **Optimization**: Set to number of available cores for best performance
    **Example**: ``--nproc 64`` for high-performance computing

``--memory TEXT``
    Memory allocation for Gaussian calculations.
    
    **Default**: ``"30GB"``
    **Format**: Number with unit (MB, GB, TB)
    **Example**: ``--memory 50GB`` for large systems

``--charge INT``
    Molecular charge for quantum calculations.
    
    **Default**: ``0`` (neutral)
    **Range**: Any integer
    **Example**: ``--charge 1`` for cations

``--multiplicity INT``
    Spin multiplicity for quantum calculations.
    
    **Default**: ``1`` (singlet)
    **Common values**: 1 (singlet), 2 (doublet), 3 (triplet)
    **Example**: ``--multiplicity 2`` for radicals

Advanced Options
~~~~~~~~~~~~~~~~

``--train-test-split FLOAT``
    Fraction of data for training (0.0-1.0).
    
    **Default**: ``0.85``
    **Range**: 0.5-0.95
    **Impact**: Higher values give more training data but less validation

``--shap-fit-threshold FLOAT``
    Minimum R² threshold for SHAP value fitting.
    
    **Default**: ``0.3``
    **Range**: 0.0-1.0
    **Purpose**: Controls quality of SHAP model explanations

``--generate-fitting``
    Generate additional fitting plots and statistics.
    
    **Default**: ``True``
    **Output**: Regression plots, residual analysis
    **Use**: For detailed model diagnostics

``--cores INT``
    Number of CPU cores for machine learning algorithms.
    
    **Default**: ``32``
    **Note**: Different from ``--nproc`` which is for quantum calculations
    **Example**: ``--cores 16`` for shared computing resources

Workflow Steps
---------------

The workflow command executes the following steps sequentially:

1. **SMILES to XYZ Conversion**
   - Converts SMILES strings to 3D molecular structures
   - Generates initial geometry files
   - Output: ``*.xyz`` files and mapping

2. **XTB Optimization** (optional)
   - Pre-optimizes molecular geometries
   - Reduces Gaussian computation time
   - Method: GFN2-xTB with tight convergence

3. **Gaussian Input Generation**
   - Creates Gaussian input files with specified method
   - Includes solvation and dispersion corrections
   - Output: ``*.gjf`` files

4. **Gaussian Calculation**
   - Performs quantum chemical calculations
   - Generates wavefunction files
   - Output: ``*.fchk`` and ``*.log`` files

5. **Multiwfn Analysis**
   - Calculates molecular surface properties
   - Extracts electrostatic and topological descriptors
   - Output: ``FullOption*.csv`` files

6. **Feature Extraction**
   - Processes surface analysis results
   - Computes atomic and molecular descriptors
   - Output: ``FinalFull*.csv`` files

7. **Machine Learning Analysis**
   - Performs feature selection and model training
   - Generates regression models with performance metrics
   - Output: Training and test set analysis files

8. **Interactive Visualization**
   - Launches SHAP-based interactive dashboard
   - Provides molecular-level interpretation
   - Output: Web-based visualization interface

Examples
--------

Basic Workflow
~~~~~~~~~~~~~~

Run a complete analysis with default settings:

.. code-block:: bash

   surfacia workflow -i molecules.csv

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Custom workflow with specific computational parameters:

.. code-block:: bash

   surfacia workflow -i data.csv \
       --extract-mode 1 \
       --extract-element "S" \
       --max-features 5 \
       --stepreg-runs 3 \
       --epoch 128 \
       --test-samples "25,11,20,31" \
       --port 8054

High-Performance Computing
~~~~~~~~~~~~~~~~~~~~~~~~

Workflow optimized for HPC environments:

.. code-block:: bash

   surfacia workflow -i large_dataset.csv \
       --nproc 64 \
       --memory 50GB \
       --cores 32 \
       --epoch 256 \
       --max-features 10

Resume Interrupted Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continue from the last successful step:

.. code-block:: bash

   surfacia workflow --resume -i molecules.csv \
       --test-samples "79,22,82,36,70,80" \
       --api-key "your-api-key"

Real-World Usage Examples
-------------------------

Based on your command history, here are typical usage patterns:

Sulfur-Containing Molecule Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   surfacia workflow -i Shenyan80.csv \
       --resume \
       --extract-mode 1 \
       --extract-element "S" \
       --max-features 5 \
       --stepreg-runs 3 \
       --epoch 128 \
       --test-samples "1,2" \
       --port 8054

Large Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   surfacia workflow -i liuxinru.csv \
       --resume \
       --extract-xyz1 sub.xyz1 \
       --max-features 5 \
       --stepreg-runs 3 \
       --epoch 128 \
       --test-samples "25,11,20,31" \
       --port 8054 \
       --api-key "your-api-key"

Production Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   surfacia workflow -i dataset.csv \
       --max-features 5 \
       --stepreg-runs 1 \
       --epoch 64 \
       --test-samples "1,2" \
       --port 8054

Output Files
------------

The workflow generates several types of output files:

**Main Output Directory**: ``Surfacia_Workflow_YYYYMMDD_HHMMSS/``

**Structure Files**:
- ``*.xyz``: 3D molecular coordinates
- ``*.gjf``: Gaussian input files
- ``*.fchk``: Formatted checkpoint files
- ``*.log``: Gaussian calculation logs

**Analysis Files**:
- ``FullOption*.csv``: Multiwfn surface analysis results
- ``FinalFull*.csv``: Processed feature matrices
- ``Training_Set_Detailed*.csv``: Machine learning training data
- ``Test_Set_Detailed*.csv``: Machine learning test data

**Visualization**:
- Interactive dashboard accessible via web browser
- SHAP plots and molecular visualizations
- Performance metrics and model statistics

Troubleshooting
---------------

Common Issues and Solutions:

**Memory Errors**:
- Reduce ``--memory`` parameter
- Use smaller dataset for testing
- Increase system RAM or use cluster computing

**Convergence Problems**:
- Try different ``--gaussian-keywords``
- Use ``--extract-mode 1`` for simpler analysis
- Check molecular structures for errors

**Performance Optimization**:
- Adjust ``--nproc`` to match available cores
- Use ``--resume`` for interrupted calculations
- Monitor disk space for large outputs

**Visualization Issues**:
- Check ``--port`` availability
- Verify ``--api-key`` configuration
- Ensure network connectivity for dashboard

Technical Notes
---------------

**Memory Requirements**:
- Small molecules (<50 atoms): 8-16 GB RAM
- Medium systems (50-100 atoms): 16-32 GB RAM
- Large systems (>100 atoms): 32+ GB RAM

**Computational Time**:
- Small datasets (<20 molecules): 2-6 hours
- Medium datasets (20-50 molecules): 6-24 hours
- Large datasets (>50 molecules): 24+ hours

**Disk Space**:
- Allocate 5-10 GB per 100 molecules
- Additional space for intermediate files
- Consider compression for long-term storage

**Parallel Processing**:
- ``--nproc`` affects quantum calculations
- ``--cores`` affects machine learning
- Both can be optimized independently

See Also
--------

- :doc:`ml` - Machine learning analysis details
- :doc:`visualization` - SHAP visualization reference
- :doc:`tutorials/basic_workflow` - Step-by-step workflow tutorial
- :doc:`../user_guide/workflows` - Workflow configuration guide
