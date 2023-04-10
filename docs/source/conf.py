# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cleanvision"
copyright = f"{datetime.datetime.now().year}, Cleanlab Inc."
author = "Cleanlab Inc."
release = "2023"

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

templates_path = ["_templates"]

exclude_patterns = ["cleanvision/issue_managers/*", "cleanvision/utils/*"]

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

# -- Options for copybutton extension -----------------------------------------

# Strip input prompts when copying code blocks. Supports:
# - Python Repl + continuation prompt
# - Bash prompt
# - ipython + continuation prompt
# - jupyter-console + continuation prompt
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = ""
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanlab_logo_only.png"

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/cleanlab/cleanvision",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "brand.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
napoleon_use_ivar = True
# https://stackoverflow.com/questions/12206334/sphinx-autosummary-toctree-contains-reference-to-nonexisting-document-warnings
numpydoc_show_class_members = True
