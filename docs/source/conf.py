# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cleanvision'
copyright = f"{datetime.datetime.now().year}, Cleanlab Inc."
author = 'Cleanlab'
release = '2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "autodocsumm",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_tabs.tabs",
    "sphinx_multiversion",
    "sphinx_copybutton",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

# --------- Autodoc configuration --------------------------------------------

# Default options to an ..autoXXX directive.
autodoc_default_options = {
    "autosummary": True,
    "members": None,
    "inherited-members": None,
    "show-inheritance": None,
    "special-members": "__call__",
}

# Subclasses should show parent classes docstrings if they don't override them.
autodoc_inherit_docstrings = True

# Order functions displayed by the order of source code
autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# https://stackoverflow.com/questions/12206334/sphinx-autosummary-toctree-contains-reference-to-nonexisting-document-warnings
numpydoc_show_class_members = True
