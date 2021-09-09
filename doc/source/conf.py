# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Author: Aakash Kaushik 

# -- Project information -----------------------------------------------------

project = 'mlpack'
copyright = '2021, mlpack'
author = 'mlpack'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------
# Sphinx Extensions
extensions = [ 'breathe', 'exhale', 'm2r2' ]

# Breathe configuration
breathe_projects = { "mlpack": "./doxyoutput/xml/" }

breathe_default_project = "mlpack"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Models API",
    "doxygenStripFromPath":  "./",
    # Suggested optional arguments
    "createTreeView":        True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": True
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

html_logo = "./assets/mlpack-logo.png"
html_favicon = "./assets/mlpack-favicon.ico"

# Theme specific HTML configuration
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'Models',

    # Set the color and the accent color
    'color_primary': 'light-blue',
    'color_accent': 'deep-orange',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/mlpack/models',
    'repo_name': 'Models',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 3,
    # If False, expand all TOC entries
    'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
