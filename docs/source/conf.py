# Copyright (C) 2024 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

project = html_title = "Anaconda AI"
copyright = "2025 Anaconda, Inc"  # noqa: A001
author = "Anaconda, Inc"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.images",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_sitemap",
    "notfound.extension",
]

myst_heading_anchors = 3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

pygments_style = "monokai"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

notfound_urls_prefix = ""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_logo = "_static/Anaconda_Icon.png"

# Serving the robots.txt since we want to point to the sitemap.xml file
html_extra_path = ["robots.txt"]

html_theme_options = {
    "navigation_depth": -1,
    "use_edit_page_button": False,
    "navbar_align": "right",
    "navbar_center": ["navbar-nav-override"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/anaconda/assistant-ai",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}

# Custom sidebar templates, maps document names to template names.

# html_sidebars = {
#     "**": [
#         "sidebar-nav-bs-override"
#     ],  # Use the newly made template instead of the default sidebar navigation
# }

html_context = {
    "github_user": "anaconda",
    "github_repo": "assistant-ai",
    "github_version": "main",
    "doc_path": "docs",
}

html_show_sourcelink = True

# We don't have a locale set, so we can safely ignore that for the sitemaps.
sitemap_locales = [None]
# We're hard-coding stable here since that's what we want Google to point to.
sitemap_url_scheme = "{link}"

# -- For sphinx_reredirects ------------------------------------------------

redirects: dict[str, str] = {}
