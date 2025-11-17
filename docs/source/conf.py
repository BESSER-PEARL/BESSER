# Configuration file for the Sphinx documentation builder.
import datetime
import importlib
import inspect
import os
import sys
from configparser import ConfigParser

# -- Add the project root directory to the path
sys.path.insert(0, os.path.abspath('../../'))

config: ConfigParser = ConfigParser()
config.read('../../setup.cfg')

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
    'sphinx_paramlinks',  # allows :param: directives within Python documentation to be linkable
    'sphinx.ext.linkcode',  # add external links to source code
    'sphinx_copybutton',  # add a little “copy” button to the right of the code blocks
    'sphinx.ext.napoleon',  # support for Google (and also NumPy) style docstrings
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

html_title = f"{project} {release}"

# -- Options for HTML output
html_favicon = "_static/besser_ico.ico"
html_theme_options = {
    "light_logo": "besser_logo_light.png",
    "dark_logo": "besser_logo_dark.png"
}
html_theme = 'furo'
html_static_path = ['_static']

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