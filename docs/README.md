# Surfacia Documentation

This directory contains the complete Sphinx documentation for the Surfacia framework.

## 🚀 Quick Start - Building and Viewing the Documentation

### Prerequisites

1. **Install documentation dependencies**:
   ```bash
   cd docs/
   pip install -r requirements.txt
   ```

2. **Ensure Sphinx is installed**:
   ```bash
   pip install sphinx
   ```

### Building the Documentation

#### Option 1: Using Make (Linux/macOS)

```bash
# Navigate to docs directory
cd docs/

# Build HTML documentation
make html

# Clean and rebuild
make clean-build

# Build with live reload (for development)
make livehtml
```

#### Option 2: Using make.bat (Windows)

```cmd
# Navigate to docs directory
cd docs\

# Build HTML documentation
make.bat html

# Clean build directory
rmdir /s build
make.bat html
```

#### Option 3: Direct Sphinx Commands

```bash
# Navigate to docs directory
cd docs/

# Build HTML documentation
sphinx-build -b html source build/html

# Build with clean slate
rm -rf build/
sphinx-build -b html source build/html
```

### 📖 Viewing the Documentation

#### Method 1: Open in Browser (Recommended)

After building, open the main page:

**Windows:**
```cmd
# Open in default browser
start build\html\index.html
```

**Linux/macOS:**
```bash
# Open in default browser
open build/html/index.html
# or
xdg-open build/html/index.html
```

#### Method 2: Local HTTP Server

```bash
# Navigate to build directory
cd build/html/

# Start local server (Python 3)
python -m http.server 8000

# Or use the make target
cd ../../  # back to docs/
make serve
```

Then open your browser and go to: `http://localhost:8000`

#### Method 3: Live Development Server

For real-time editing and preview:

```bash
# Install sphinx-autobuild if not already installed
pip install sphinx-autobuild

# Start live server
make livehtml
# or directly:
sphinx-autobuild source build/html
```

This will start a server at `http://localhost:8000` that automatically rebuilds and refreshes when you edit source files.

## 🌐 Publish the Latest Docs to Web

Surfacia is configured for Read the Docs via `.readthedocs.yaml`.

### Standard release flow (recommended)

1. Commit your documentation changes and push to GitHub:
   ```bash
   git add docs/ .readthedocs.yaml
   git commit -m "docs: update troubleshooting and release notes"
   git push origin main
   ```
2. Open Read the Docs project dashboard and trigger **Build** for `main`.
3. After build success, verify your web docs URL:
   - `https://surfacia.readthedocs.io/`

### If the website still shows old content

1. In Read the Docs, open **Builds** and check latest build status/log.
2. Trigger **Build** again for `main`.
3. Hard refresh browser cache (`Ctrl+F5`).
4. Confirm the RTD default branch is `main`.

### Versioned docs for releases (optional)

For release-specific docs (for example `v3.0.2`):

```bash
git tag v3.0.2
git push origin v3.0.2
```

Then activate the new version in the Read the Docs admin panel.

## 📁 Documentation Structure

```
docs/
├── source/                          # Source files
│   ├── conf.py                     # Sphinx configuration
│   ├── index.rst                   # Main documentation page
│   ├── getting_started/            # Getting started guide
│   │   ├── index.rst
│   │   ├── installation.rst
│   │   ├── quick_start.rst
│   │   └── basic_concepts.rst
│   ├── commands/                   # Command reference
│   │   ├── index.rst
│   │   └── workflow.rst
│   └── _static/                    # Static assets
│       └── css/
│           └── custom.css
├── build/                          # Generated documentation
│   └── html/                       # HTML output
├── requirements.txt                # Documentation dependencies
├── Makefile                        # Build commands (Linux/macOS)
├── make.bat                        # Build commands (Windows)
└── README.md                       # This file
```

## 🛠️ Development Workflow

### Adding New Pages

1. **Create new .rst file** in appropriate directory:
   ```bash
   # Example: Add new command documentation
   touch source/commands/new_command.rst
   ```

2. **Add to table of contents** in parent index.rst:
   ```rst
   .. toctree::
      :maxdepth: 2
      
      existing_page
      new_command
   ```

3. **Build and test**:
   ```bash
   make html
   ```

### Editing Existing Content

1. **Edit .rst files** in `source/` directory
2. **Use live reload** for immediate feedback:
   ```bash
   make livehtml
   ```
3. **Check for warnings** during build

### Adding Images and Assets

1. **Place files** in `source/_static/`
2. **Reference in documentation**:
   ```rst
   .. image:: _static/images/diagram.png
      :alt: Description
      :width: 600px
   ```

## 🎨 Customization

### Themes and Styling

- **Theme**: Currently using Furo theme
- **Custom CSS**: Edit `source/_static/css/custom.css`
- **Theme options**: Modify in `source/conf.py`

### Extensions

Current extensions in `conf.py`:
- `sphinx.ext.autodoc` - Auto-generate API docs
- `sphinx_copybutton` - Copy code blocks
- `sphinx_tabs` - Tabbed content
- `sphinxcontrib.mermaid` - Diagrams
- `nbsphinx` - Jupyter notebook integration

## 🔧 Troubleshooting

### Common Issues

**Build fails with import errors:**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**Mermaid diagrams not rendering:**
```bash
# Check if sphinxcontrib-mermaid is installed
pip install sphinxcontrib-mermaid
```

**Live reload not working:**
```bash
# Install sphinx-autobuild
pip install sphinx-autobuild
```

**CSS changes not visible:**
```bash
# Clear browser cache or use incognito mode
# Or force rebuild:
make clean-build
```

### Build Warnings

- **Missing references**: Check internal links
- **Image not found**: Verify paths in `_static/`
- **Syntax errors**: Check .rst formatting

## 📚 Writing Documentation

### reStructuredText Basics

```rst
# Main heading
===============

## Section heading
-----------------

### Subsection
~~~~~~~~~~~~~~

**Bold text**
*Italic text*
``Code text``

.. code-block:: python

   # Python code block
   print("Hello, world!")

.. note::
   This is a note admonition.

.. warning::
   This is a warning admonition.
```

### Sphinx Directives

```rst
# Cross-references
:doc:`other_page`
:ref:`section-label`

# API documentation
.. automodule:: surfacia.core
   :members:

# Tables of contents
.. toctree::
   :maxdepth: 2
   
   page1
   page2
```

## 🚀 Deployment

### GitHub Pages

1. **Build documentation**:
   ```bash
   make html
   ```

2. **Copy to gh-pages branch**:
   ```bash
   # Copy build/html/* to gh-pages branch
   ```

### Read the Docs

1. **Connect repository**: `https://github.com/sym823808458/Surfacia`
2. **Read the Docs config**: use the repository root file `.readthedocs.yaml`
3. **Sphinx config**: `docs/source/conf.py`
4. **Dependency file**: `docs/requirements.txt`
5. **Homepage source**: `docs/source/index.rst`
6. **Custom styles**: `docs/source/_static/css/custom.css`
7. **Automatic builds**: every push to the tracked branch can trigger a rebuild

### Repository Paths That Control the Docs Site

The main files you will usually edit are:

```text
Surfacia/
├── .readthedocs.yaml              # Read the Docs build entry
├── docs/
│   ├── requirements.txt           # Docs build dependencies
│   └── source/
│       ├── conf.py                # Sphinx theme, repo links, extensions
│       ├── index.rst              # Homepage and the main card layout
│       └── _static/css/custom.css # Homepage and card styling
```

## 📞 Support

For documentation issues:
1. Check build logs for specific errors
2. Verify all dependencies are installed
3. Test with clean build: `make clean-build`
4. Check Sphinx documentation for advanced features

## 🎯 Next Steps

After setting up the documentation:

1. **Complete remaining sections**:
   - Command references (ml-analysis, shap-viz, utilities)
   - Descriptor documentation
   - API reference
   - Tutorial examples

2. **Add more content**:
   - Screenshots and diagrams
   - Video tutorials
   - FAQ section
   - Troubleshooting guides

3. **Set up automation**:
   - Continuous integration builds
   - Automatic deployment
   - Link checking
   - Spell checking
