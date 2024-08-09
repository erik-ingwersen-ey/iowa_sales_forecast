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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys
import os
from pathlib import Path
import importlib
import datetime

try:
    import sphinx.ext.imgmath  # noqa
except ImportError:
    img_ext = 'sphinx.ext.pngmath'
else:
    img_ext = 'sphinx.ext.imgmath'

# -- Project information -----------------------------------------------------

project = 'Iowa Liquor Sales Forecast'
copyright = '2024, Erik Ingwersen'
author = 'Erik Ingwersen'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

PACKAGE_NAME = "iowa_forecast"

current_dir = Path.cwd()
package_dirs = list(current_dir.glob(f"**/{PACKAGE_NAME}/__init__.py"))
max_parents = 2

while len(package_dirs) == 0 and max_parents > 0:
    current_dir = current_dir.parent
    package_dirs = list(current_dir.glob(f"**/{PACKAGE_NAME}/__init__.py"))
    max_parents -= 1

if len(package_dirs) > 0:
    package_dir = package_dirs[0].parent
    sys.path.insert(0, str(package_dir))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # If you're using Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode',  # Optional: Add links to highlighted source code
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_click',
    'sphinx_inline_tabs',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_pyreverse',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    img_ext,
]

autodoc_typehints = "both"
autodoc_typehints_format = "short"
autodoc_member_order = 'bysource'

autodoc_default_options = {
    'members': True,
    # 'undoc-members': True,
    'show-inheritance': True,
    # 'lines-between-items': 1,  # Adds space between parameter entries
}

autodocgen_config = {
    'modules': [PACKAGE_NAME],
    'generated_source_dir': './source/',
    # if module matches this then it and any of its submodules will be skipped
    # 'skip_module_regex': 'supply.allocation_model.etl.validation',
    # produce a text file containing a list of everything documented.
    # you can use this in a test to notice when you've
    # intentionally added/removed/changed a documented API
    'write_documented_items_output_file': 'autodocgen_documented_items.txt',
    'autodoc_options_decider': {
        PACKAGE_NAME: {'inherited-members': True},
    },
    'module_title_decider': lambda modulename: 'API Reference' if modulename == PACKAGE_NAME
    else modulename,
}

# Add any paths that contain templates here, relative to this directory.
default_role = 'code'

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {'optional', 'type_without_description', 'BadException'}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "furo"

html_theme_options = {
    # "github_url": "https://github.com/ingwersen-erik/peloptimize",
    # "show_prev_next": True,
    # "navbar_end": ["search-field.html", "navbar-icon-links.html"],
    "navigation_with_keys": True,
    "sidebar_hide_name": True,
    # "light_logo": "<LOGO-FILEPATH>.jpg",
    # "dark_logo": "<LOGO-FILEPATH>.jpg",
}
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
    ]
}
# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

napoleon_use_param = True  # Disables the Napoleon pre-processing of params
napoleon_use_rtype = False  # Disables the Napoleon pre-processing of a return type

source_suffix = {
    '.rst': 'restructuredtext', '.txt': 'restructuredtext', '.md': 'markdown'
}

# Example configuration for inter-sphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/devdocs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
autosummary_generate = True
