"""Sphinx configuration for the VIPER documentation."""

import shutil
from importlib.metadata import version as _dist_version
from pathlib import Path

# symlinks break myst relative links
_DOCS_DIR = Path(__file__).resolve().parent
for _nb in ("example_pwn_fit.ipynb", "example_pwn_gammapy.ipynb"):
    _src = _DOCS_DIR.parent / "examples" / _nb
    if _src.exists():
        shutil.copyfile(_src, _DOCS_DIR / "tutorials" / _nb)

project = "VIPER"
author = "Jason Alford"
copyright = "2026, Jason Alford, Joseph Gelfand, and contributors"
release = _dist_version("viper-pwn")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# notebooks render from committed outputs
nb_execution_mode = "off"

myst_enable_extensions = ["dollarmath", "colon_fence"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# not documentation sources
exclude_patterns = ["_build", "dev", "scripts", "videos", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "VIPER"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/j-alford/viper-pwn",
            "icon": "fa-brands fa-github",
        },
    ],
}
