# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'viral-nucleotide-transformers'
copyright = '2023, karlo.lee'
author = 'karlo.lee'
release = "dev"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    'sphinxcontrib.mermaid',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ["style.css"]

add_module_names = True  # Remove namespaces from class/method signatures

# sphinx.ext.todo
todo_include_todos = True  # Todo 출력

# sphinx.ext.autodoc
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
# autosummary_mock_imports = ["src.catcher.config", "src.tag.config"]  # Prevent config modules converted into rst

html_show_sphinx = False
html_theme_options = {
  "use_sidenotes": True,
  "show_toc_level": 2,
  "logo": {
      "text": "Viral Nucleotide Transformers",
      "alt_text": "PyData Theme",
  },
  "show_toc_level": 1,
  "navbar_align": "left",
}

# -- Extension configuration -------------------------------------------------
html_js_files = [
  'https://cdn.jsdelivr.net/npm/mermaid@10.2.4/dist/mermaid.min.js'
]