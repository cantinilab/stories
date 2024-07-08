# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../"))
print(sys.path)

import stories

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "stories-jax"
copyright = "2024, Geert-Jan Huizing, Gabriel Peyré, Laura Cantini"
author = "Geert-Jan Huizing, Gabriel Peyré, Laura Cantini"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser"]
extensions += ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "nbsphinx", "sphinx_design"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_permalinks = False
html_title = "STORIES"
html_short_title = "STORIES"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_static_path = ["_static"]

pygments_style = "sphinx"
