Installation
============

Surfacia is a comprehensive computational chemistry framework that requires careful setup of both Python dependencies and external quantum chemistry software. This guide will walk you through the complete installation process.

System Requirements
-------------------

**Operating Systems**
   - Linux (recommended for best performance)
   - macOS (Intel/Apple Silicon)
   - Windows (with WSL2 recommended)

**Hardware Requirements**
   - Minimum: 8 GB RAM, 4 CPU cores
   - Recommended: 16+ GB RAM, 8+ CPU cores
   - Storage: 10+ GB free space for calculations
   - GPU: Optional for ML acceleration

**Software Prerequisites**
   - Python 3.9 or newer
   - Git (for cloning repository)

External Software Dependencies
-----------------------------

Surfacia integrates with several established quantum chemistry packages. These must be installed separately:

**XTB (Semi-empirical Methods)**
   - Download from: `https://xtb-docs.readthedocs.io/en/latest/setup.html`
   - Recommended version: 6.5.0 or newer
   - Installation: Follow official documentation for your OS

**Gaussian (Quantum Chemistry)**
   - Commercial software requiring license
   - Recommended: Gaussian 16
   - Must be accessible via system PATH
   - Installation guide: `http://gaussian.com/`

**Multiwfn (Wavefunction Analysis)**
   - Download from: `http://sobereva.com/multiwfn/`
   - Recommended: Multiwfn 3.8 or newer
   - Must be executable and in PATH

Python Installation
------------------

**Method 1: From PyPI (Recommended)**

Install the latest stable version:

.. code-block:: bash

   pip install surfacia

**Method 2: From Source (Development Version)**

For the latest features and bug fixes:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/sym823808458/Surfacia.git
   cd Surfacia
   
   # Create virtual environment (recommended)
   python -m venv surfacia_env
   source surfacia_env/bin/activate  # Linux/macOS
   # surfacia_env\Scripts\activate  # Windows
   
   # Install in editable mode
   pip install -e .

**Method 3: Conda Installation**

.. code-block:: bash

   # Using conda-forge
   conda install -c conda-forge surfacia
   
   # Or create dedicated environment
   conda create -n surfacia python=3.9
   conda activate surfacia
   conda install -c conda-forge surfacia

Python Dependencies
-----------------

Surfacia automatically manages these Python packages:

**Core Dependencies**
   - ``numpy>=1.21.0`` - Numerical computing
   - ``pandas>=1.3.0`` - Data manipulation
   - ``scipy>=1.7.0`` - Scientific computing

**Chemistry & Molecular Modeling**
   - ``openbabel-wheel>=3.1.1`` - Chemical file formats
   - ``rdkit-pypi`` - Chemical informatics (install via conda-forge)

**Machine Learning**
   - ``scikit-learn>=1.0.0`` - ML algorithms
   - ``xgboost>=1.6.0`` - Gradient boosting
   - ``shap>=0.41.0`` - Model interpretability

**Visualization & Web**
   - ``matplotlib>=3.5.0`` - Plotting
   - ``seaborn>=0.11.0`` - Statistical visualization
   - ``plotly>=5.0.0`` - Interactive plots
   - ``dash>=2.0.0`` - Web dashboard
   - ``py3Dmol>=2.0.0`` - 3D molecular visualization
   - ``ipywidgets>=7.6.0`` - Interactive widgets

**Quantum Chemistry Interface**
   - ``cclib>=1.7.2`` - Quantum chemistry file parsing

**CLI & Utilities**
   - ``click>=8.0.0`` - Command line interface
   - ``pyyaml>=6.0`` - Configuration files
   - ``toml>=0.10.2`` - TOML parsing
   - ``tqdm>=4.64.0`` - Progress bars
   - ``pathlib2>=2.3.7`` - Path utilities
   - ``joblib>=1.1.0`` - Parallel processing

**AI Integration**
   - ``zhipuai>=2.0.0`` - AI assistant integration

**Optional Development Tools**
   - ``jupyter>=1.0.0`` - Jupyter notebooks
   - ``pytest>=6.0.0`` - Testing
   - ``black>=21.0.0`` - Code formatting
   - ``flake8>=4.0.0`` - Linting

Configuration
-------------

**Environment Variables**

Set these for optimal performance:

.. code-block:: bash

   # Add to ~/.bashrc, ~/.zshrc, or Windows Environment Variables
   export GAUSSIAN_ROOT="/path/to/gaussian16"
   export MULTIWFN_ROOT="/path/to/multiwfn"
   export OMP_NUM_THREADS=8  # Adjust based on CPU cores
   export SURFACIA_CACHE_DIR="/path/to/cache"  # Optional cache directory

**PATH Configuration**

Ensure all external tools are accessible:

.. code-block:: bash

   # Test each installation
   which g16 && echo "✅ Gaussian found"
   which Multiwfn && echo "✅ Multiwfn found"
   which xtb && echo "✅ XTB found"
   
   # Verify Python installation
   python -c "import surfacia; print('✅ Surfacia installed')"

RDKit Installation (Special Cases)
------------------------------

RDKit can be challenging to install. Recommended approaches:

**Conda (Easiest)**
.. code-block:: bash

   conda install -c conda-forge rdkit

**Pip with Conda Environment**
.. code-block:: bash

   # Create conda environment first
   conda create -n surfacia python=3.9
   conda activate surfacia
   
   # Then install RDKit
   conda install -c conda-forge rdkit
   pip install surfacia

**Docker Installation**
--------------------

For isolated environments:

.. code-block:: bash

   # Use our pre-built Docker image
   docker pull sym823808458/surfacia:latest
   
   # Run with mounted volumes
   docker run -v $(pwd):/workspace -it sym823808458/surfacia:latest

Verification
------------

**Basic Installation Test**

.. code-block:: bash

   # Check Surfacia version
   surfacia --version
   
   # List available commands
   surfacia --help
   
   # Test workflow help
   surfacia workflow --help

**Complete Functionality Test**

Create a simple test to verify all components:

.. code-block:: bash

   # Create test data
   echo "name,smiles,target" > test.csv
   echo "caffeine,CN1C=NC2=C1C(=O)N(C=O)N2C,5.2" >> test.csv
   
   # Test workflow (dry run)
   surfacia workflow -i test.csv --test-samples "0" --help

Troubleshooting
---------------

**Common Installation Issues**

.. admonition:: Gaussian not found
   :class: warning

   **Error**: ``Command 'g16' not found`` or similar
   
   **Solution**: 
   1. Verify Gaussian installation path
   2. Add to system PATH or GAUSSIAN_ROOT
   3. Test with ``g16 < test.com``

.. admonition:: RDKit import error
   :class: warning

   **Error**: ``ImportError: No module named 'rdkit'``
   
   **Solution**: 
   1. Use conda-forge: ``conda install -c conda-forge rdkit``
   2. Ensure same Python environment for RDKit and Surfacia
   3. Check conda environment: ``conda list rdkit``

.. admonition:: Permission denied
   :class: warning

   **Error**: ``PermissionError`` during calculations
   
   **Solution**: 
   1. Check directory permissions: ``ls -la``
   2. Ensure write access to working directory
   3. Consider using user directory for calculations

.. admonition:: Out of memory
   :class: warning

   **Error**: Calculations fail with memory errors
   
   **Solution**: 
   1. Increase Gaussian memory: ``--memory 32GB``
   2. Reduce parallel jobs: ``--nproc 4``
   3. Use XTB for pre-optimization

**Performance Optimization**

.. code-block:: bash

   # Optimize for your system
   export OMP_NUM_THREADS=$(nproc)  # Use all CPU cores
   export MKL_NUM_THREADS=$(nproc)  # Intel MKL optimization
   
   # For Gaussian calculations
   surfacia workflow -i data.csv --nproc 8 --memory 16GB

Next Steps
----------

After successful installation:

1. 📖 Read the :doc:`quick_start` guide
2. 🧠 Learn about :doc:`basic_concepts`
3. 🚀 Try your first :doc:`../tutorials/basic_workflow`
4. 📊 Explore the :doc:`../api/index` reference

**Getting Help**

If you encounter issues:

1. 📖 Check the :doc:`../user_guide/troubleshooting` guide
2. 🔍 Search `GitHub Issues <https://github.com/sym823808458/Surfacia/issues>`_
3. 📧 Create new issue with system information:
   - OS and version
   - Python version
   - Surfacia version
   - Complete error message
   - Steps to reproduce
