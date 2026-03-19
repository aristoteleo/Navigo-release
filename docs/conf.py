from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent

# -- Project information -----------------------------------------------------
project = "Navigo"
author = "Navigo Team"
copyright = f"{datetime.now():%Y}, {author}."
version = "main"
release = "main"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "substitution",
]
myst_url_schemes = ("http", "https", "mailto")

nb_execution_mode = "off"
nb_merge_streams = True
nb_output_stderr = "remove"

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = project
html_logo = "_static/logo_navigo.png"
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["css/override.css"]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7fb4ff",
        "color-brand-content": "#7fb4ff",
    },
}

pygments_style = "tango"
pygments_dark_style = "monokai"
