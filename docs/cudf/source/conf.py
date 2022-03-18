#!/usr/bin/env python3
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
#
# cudf documentation build configuration file, created by
# sphinx-quickstart on Wed May  3 10:59:22 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from docutils.nodes import Text
from sphinx.addnodes import pending_xref
import cudf

sys.path.insert(0, os.path.abspath(cudf.__path__[0]))
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath("./_ext"))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "PandasCompat",
]

copybutton_prompt_text = ">>> "
autosummary_generate = True
ipython_mplbackend = "str"

html_use_modindex = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "cudf"
copyright = "2018-2021, NVIDIA"
author = "NVIDIA"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '22.06'
# The full version, including alpha/beta/rc tags.
release = '22.06.00'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['venv', "**/includes/**",]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/rapidsai/cudf",
    "twitter_url": "https://twitter.com/rapidsai",
    "show_toc_level": 1,
    "navbar_align": "right",
}
include_pandas_compat = True


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "pydata_sphinx_theme"
html_logo = "_static/RAPIDS-logo-purple.png"


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "cudfdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "cudf.tex",
        "cudf Documentation",
        "Continuum Analytics",
        "manual",
    )
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "cudf", "cudf Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "cudf",
        "cudf Documentation",
        author,
        "cudf",
        "One line description of project.",
        "Miscellaneous",
    )
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
}

# Config numpydoc
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False

autoclass_content = "class"

# Replace API shorthands with fullname
_reftarget_aliases = {
    "cudf.Series": ("cudf.core.series.Series", "cudf.Series"),
    "cudf.Index": ("cudf.core.index.Index", "cudf.Index"),
    "cupy.core.core.ndarray": ("cupy.ndarray", "cupy.ndarray"),
}

_internal_names_to_ignore = {"cudf.core.column.string.StringColumn"}


def resolve_aliases(app, doctree):
    pending_xrefs = doctree.traverse(condition=pending_xref)
    for node in pending_xrefs:
        alias = node.get("reftarget", None)
        if alias is not None and alias in _reftarget_aliases:
            real_ref, text_to_render = _reftarget_aliases[alias]
            node["reftarget"] = real_ref

            text_node = next(
                iter(node.traverse(lambda n: n.tagname == "#text"))
            )
            text_node.parent.replace(text_node, Text(text_to_render, ""))


def ignore_internal_references(app, env, node, contnode):
    name = node.get("reftarget", None)
    if name is not None and name in _internal_names_to_ignore:
        node["reftarget"] = ""
        return contnode

def process_class_docstrings(app, what, name, obj, options, lines):
    """
    For those classes for which we use ::
    :template: autosummary/class_without_autosummary.rst
    the documented attributes/methods have to be listed in the class
    docstring. However, if one of those lists is empty, we use 'None',
    which then generates warnings in sphinx / ugly html output.
    This "autodoc-process-docstring" event connector removes that part
    from the processed docstring.
    """
    if what == "class":
        if name in {"cudf.RangeIndex", "cudf.Int64Index", "cudf.UInt64Index", "cudf.Float64Index", "cudf.CategoricalIndex", "cudf.IntervalIndex", "cudf.MultiIndex", "cudf.DatetimeIndex", "cudf.TimedeltaIndex", "cudf.TimedeltaIndex"}:

            cut_index = lines.index('.. rubric:: Attributes')
            lines[:] = lines[:cut_index]




def setup(app):
    app.add_css_file("params.css")
    app.connect("doctree-read", resolve_aliases)
    app.connect("missing-reference", ignore_internal_references)
    app.connect("autodoc-process-docstring", process_class_docstrings)
