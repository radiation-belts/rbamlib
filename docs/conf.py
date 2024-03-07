# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

print(os.path.abspath(".."))
print(os.path.abspath("."))


project = 'rbamlib'
copyright = '2024, Alexander Drozdov'
author = 'Alexander Drozdov'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests']

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    'myst_parser'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

## -- Generate rts files from new pyton files --

# The directory relative to this file
source_dir = '../rbamlib/'
# The target directory for the generated rst files
target_dir = './'

def generate_rst_files(source_dir, target_dir):
    """
    Create rst files from the python modules which were not included as rst files

    # TODO: also add toctree for dub packages
    """
    for root, dirs, files in os.walk(source_dir):
        if "__init__.py" in files:
            pyfiles = [os.path.splitext(f)[0] for f in files if f.endswith(".py") and not f.startswith('__')]

            last_dir = os.path.split(root)[-1]

            # Split the `root` directory into parts
            path_parts = [p for p in os.path.normpath(root).split(os.sep) if ".." not in p and not p.startswith('__')]
            module_name = ".".join(path_parts)

            if last_dir == '':
                for f in pyfiles:
                    rst_filename = os.path.join(target_dir, f + '.rst')
                    print(f"root module file name {rst_filename}:\t", f"import {module_name}.{f}")
                    if os.path.isfile(rst_filename):
                        print(f"{rst_filename} exist. Skipping.")
                    else:
                        print(f"{rst_filename} not exist. Creating default.")
                        with open(rst_filename, 'w') as rst_file:
                            automodule = f"{module_name}.{f}"
                            rst_file.write(f"{automodule}\n")
                            rst_file.write(f"{"-" * len(automodule)}\n")
                            rst_file.write(f".. automodule:: {automodule}\n")
            else:
                if module_name.count(".") < 2:
                    rst_filename = os.path.join(target_dir, last_dir + '.rst')
                else:
                    last_dir_str = module_name.split('.')[1:-1]
                    # print(last_dir_str)
                    rst_foldername = os.path.join(target_dir, *last_dir_str)
                    os.makedirs(rst_foldername, exist_ok=True)
                    rst_filename = os.path.join(rst_foldername, last_dir + '.rst')

                print(f"module file name  {rst_filename}:\t", f"import {module_name}")
                if os.path.isfile(rst_filename):
                    print(f"{rst_filename} exist. Skipping.")
                else:
                    print(f"{rst_filename} not exist. Creating default.")
                    with open(rst_filename, 'w') as rst_file:
                        rst_file.write(f".. currentmodule:: {module_name}\n")
                        rst_file.write(f"{module_name}\n")
                        rst_file.write(f"{"-" * len(module_name)}\n")
                        rst_file.write(f".. automodule:: {module_name}\n")

                        if pyfiles:
                            rst_file.write(f"Functions\n")
                            rst_file.write(f"=========\n")

                            for f in pyfiles:
                                rst_file.write(f".. autofunction:: {os.path.splitext(f)[0]}\n")

generate_rst_files(source_dir, target_dir)