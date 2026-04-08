Installation
============

System Requirements
-------------------

Surfacia requires the following software to be installed on your system:

**Required Software**
   - Python 3.8 or higher
   - Gaussian 16 (for quantum chemistry calculations)
   - Multiwfn (for wavefunction analysis)

**Operating Systems**
   - Linux (recommended)
   - macOS
   - Windows (with WSL recommended)

**Hardware Requirements**
   - Minimum: 8 GB RAM, 4 CPU cores
   - Recommended: 16+ GB RAM, 8+ CPU cores
   - Storage: 10+ GB free space for calculations

Installing Surfacia
-------------------

.. tabs::

   .. tab:: PyPI (Recommended)

      Install the latest stable version from PyPI:

      .. code-block:: bash

         pip install surfacia

   .. tab:: From Source

      Install the development version from GitHub:

      .. code-block:: bash

         git clone https://github.com/sym823808458/Surfacia.git
         cd Surfacia
         pip install -e .

   .. tab:: Conda

      Install from conda-forge:

      .. code-block:: bash

         conda install -c conda-forge surfacia

Installing Dependencies
-----------------------

**Gaussian 16**

Gaussian 16 must be purchased and installed separately. Follow the official installation guide from Gaussian Inc.

After installation, ensure Gaussian is in your PATH:

.. code-block:: bash

   # Test Gaussian installation
   g16 --version

**Multiwfn**

Download and install Multiwfn from the official website:

.. code-block:: bash

   # Download Multiwfn (Linux example)
   wget http://sobereva.com/multiwfn/misc/Multiwfn_3.8_dev_bin_Linux.zip
   unzip Multiwfn_3.8_dev_bin_Linux.zip
   
   # Add to PATH (add to ~/.bashrc)
   export PATH="/path/to/Multiwfn:$PATH"
   
   # Test installation
   Multiwfn

**Python Dependencies**

Surfacia automatically installs required Python packages:

.. code-block:: text

   numpy>=1.21.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   matplotlib>=3.5.0
   seaborn>=0.11.0
   plotly>=5.0.0
   dash>=2.0.0
   xgboost>=2.1.4,<3.0.0
   shap>=0.48.0,<0.49.0
   rdkit-pypi>=2022.3.0
   openpyxl>=3.0.0
   zhipuai>=1.0.0

.. admonition:: Version Compatibility Note
   :class: warning

   If you run into ``could not convert string to float: '[-3.xxxE0]'`` during ``surfacia ml-analysis``,
   check your installed ``xgboost`` and ``shap`` versions first. This is usually an environment compatibility issue.

Verification
------------

Verify your installation by running:

.. code-block:: bash

   # Check Surfacia installation
   surfacia --version
   
   # Test basic functionality
   surfacia --help

You should see the Surfacia version and help information.

**Test Complete Installation**

Create a test file to verify all components work together:

.. code-block:: bash

   # Create test SMILES file
   echo "Sample Name,SMILES" > test_molecules.csv
   echo "caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C" >> test_molecules.csv
   
   # Test workflow (dry run)
   surfacia workflow -i test_molecules.csv --help

Configuration
-------------

**Environment Variables**

Set up environment variables for optimal performance:

.. code-block:: bash

   # Add to ~/.bashrc or ~/.zshrc
   export GAUSSIAN_ROOT="/path/to/gaussian"
   export MULTIWFN_ROOT="/path/to/multiwfn"
   export OMP_NUM_THREADS=4  # Adjust based on your CPU cores

**API Keys**

For AI assistant features, configure your API key:

.. code-block:: bash

   # Set ZhipuAI API key
   export ZHIPUAI_API_KEY="your_api_key_here"

Troubleshooting
---------------

**Common Issues**

.. admonition:: Gaussian not found
   :class: warning

   **Error**: ``gaussian: command not found``
   
   **Solution**: Ensure Gaussian is properly installed and in your PATH. Check with ``which g16``.

.. admonition:: Multiwfn not found
   :class: warning

   **Error**: ``Multiwfn: command not found``
   
   **Solution**: Download Multiwfn and add it to your PATH. Verify with ``which Multiwfn``.

.. admonition:: Permission denied
   :class: warning

   **Error**: ``Permission denied`` when running calculations
   
   **Solution**: Ensure you have write permissions in the working directory and temporary directories.

**Getting Help**

If you encounter issues:

1. Check the :doc:`../user_guide/troubleshooting` guide
2. Search existing issues on `GitHub <https://github.com/sym823808458/Surfacia/issues>`_
3. Create a new issue with detailed error information

Next Steps
----------

Once installation is complete:

1. Try the :doc:`quick_start` tutorial
2. Learn about :doc:`basic_concepts`
3. Explore the :doc:`../commands/index` reference
