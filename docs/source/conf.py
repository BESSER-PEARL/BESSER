# Configuration file for the Sphinx documentation builder.
import datetime
import importlib
import inspect
import os
import sys
from configparser import ConfigParser
from pathlib import Path

# -- Add the project root directory to the path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

config: ConfigParser = ConfigParser()
config.read(REPO_ROOT / 'setup.cfg')

# -- Project information
project = config.get('metadata', 'description')
author = config.get('metadata', 'author')
release = config.get('metadata', 'version')
year = datetime.date.today().year
if year > 2023:
    year = '2023 - ' + str(year)
copyright = f'{year} {author}. All Rights Reserved'

# -- General configuration

extensions = [
    'sphinx.ext.duration',  # measure durations of Sphinx processing
    'sphinx.ext.doctest',  
    'sphinx.ext.autodoc',  # include documentation from docstrings
    'sphinx.ext.autosummary',  # generate autodoc summaries
    'sphinx.ext.intersphinx',  # link to other projects’ documentation
    'sphinx.ext.linkcode',  # add external links to source code
    'sphinx.ext.napoleon',  # support for Google (and also NumPy) style docstrings
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# Optional third-party libraries that some generators import lazily (e.g.
# ``deep_translator`` in the agent-personalization generator). They are not in
# the docs build environment, so mock them for autodoc — otherwise importing the
# module to read its docstrings fails with ModuleNotFoundError and the page is
# dropped from the build.
autodoc_mock_imports = ['deep_translator']

templates_path = ['_templates']

html_title = f"{project} {release}"

# -- Options for HTML output
html_favicon = "_static/besser_ico.ico"
html_theme = "sphinx_immaterial"
extensions.append("sphinx_immaterial")
html_logo = "_static/besser_logo_dark.png"
html_static_path = ["_static"]
html_css_files = ["docs.css"]
html_theme_options = {
    "font": False,
    "features": ["navigation.sections", "navigation.top", "search.highlight"],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "cyan",
            "accent": "cyan",
            "toggle": {"icon": "material/weather-night", "name": "Switch to dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "cyan",
            "accent": "cyan",
            "toggle": {"icon": "material/weather-sunny", "name": "Switch to light mode"},
        },
    ],
    "repo_url": "https://github.com/BESSER-PEARL/BESSER",
    "repo_name": "Source",
    "globaltoc_collapse": True,
    "toc_title": "On this page",
}
html_show_sourcelink = False
autodoc_member_order = 'bysource'
# Keep class attribute descriptions as fields; the properties below are indexed
# once by autodoc rather than a second time by Napoleon's Attributes sections.
napoleon_use_ivar = True
# Material already handles signature and code-block links. Keep descriptive
# parameter lists without indexing the same constructor parameters twice.
napoleon_use_param = False
# These sections describe properties and methods, not constructor arguments.
# Keep them as explanatory text; autodoc indexes the actual members below.
napoleon_custom_sections = [('Attributes', 'params_style'), ('Methods', 'params_style')]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for EPUB output
#epub_show_urls = 'footnote'


def linkcode_resolve(domain, info):
    """Generate links to module components."""
    if domain != 'py':
        return None
    if not info['module']:
        return None
    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        lines = inspect.getsourcelines(obj)
    except TypeError:
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1
    filename = info['module'].replace('.', '/')
    return f"https://github.com/BESSER-PEARL/BESSER/blob/master/{filename}.py#L{start}-L{end}"
