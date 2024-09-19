# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
# sys.path.insert(0, os.path.abspath('../src/WPSProtocol'))

# parent = Path(conf.py).parent
# sys.paths = Path(confy.py).parents[1]

# try:
#   import WPSProtocol
# except ImportError:
#   print("oh no 1")

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src"))) # Works on local systems



# -- Project information -----------------------------------------------------

project = 'WalshPulSeqProtocol'
copyright = '2024, Jessica John Britto'
author = 'Jessica John Britto'

# The full version, including alpha/beta/rc tags
release = '1.0.post4'


# import mock

# MOCK_MODULES = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate', 'scipy.special', 'math', '__future__', 'toolboxutilities']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',            # Support for TODO notes
    # 'sphinx.ext.napoleon',        # Supports Google and NumPy style docstrings
    'sphinx.ext.viewcode',        # Add links to highlighted source code
    'sphinx.ext.autodoc',         # Automatically document from docstrings
    # 'sphinx.ext.intersphinx',     # Link to other project’s documentation
    # 'sphinx.ext.mathjax',         # For rendering LaTeX-style math
    # 'sphinx.ext.githubpages',
    # "sphinx_wagtail_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'English'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',           # Directory where the output files are generated
    'Thumbs.db',        # Windows thumbnail cache files
    '.DS_Store',        # macOS directory metadata files
    # 'venv',             # Virtual environment directory (if present)
    # 'env',              # Alternative name for a virtual environment
    # 'docs/modules.rst', # If you want to exclude a specific file like modules.rst
    '**.ipynb_checkpoints',  # Jupyter notebook checkpoints
    '*.egg-info',       # Metadata for Python packages
    # 'build',            # General build directory (if present)
    # '.tox',             # Test environments (like tox directories)
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_wagtail_theme'
# html_logo = "_static/custom_logo.png"
# html_title = "My Custom Documentation"
# html_theme_options = { 'logo_only': False, }
# html_logo = '_static/block2-logo-white.png'
html_logo = ""
html_title = "WalshPulseSeqProtocol"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']