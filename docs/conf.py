# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import re
import ast
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

print(os.path.abspath(".."))
print(os.path.abspath("."))

project = 'rbamlib'
copyright = '2024-2026'
author = 'Alexander Drozdov'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_automodapi.automodapi',
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "md_alert_to_admonition",
    "md_link_adjust"
]

# List of documents where the md_alert_to_admonition should apply
md_alert_to_admonition_affected_docs = ["README"]

# List of documents where the LinkAdjustTransform should apply
# This plugin work only with "broken links", node type sphinx.addnodes.pending_xref
md_link_adjust_affected_docs = ['DEVELOPERS_GUIDE']  # Adjust these as needed

# List of link replacement rules for md_link_adjust
md_link_adjust_rule = [
    (r'^/docs/symbols\.rst$', 'symbols.html'),  # Replace `/docs/symbols.rst` with `symbols.html`. This allows link to work in both .md on GitHub and in documentation
    # Add other rules as needed
]


# myst_enable_extensions = [
#     "colon_fence",  # Fenced code blocks with ::
#     "dollarmath",   # Optional: allow inline math with $...$
# ]

myst_url_schemes = ["http", "https", ""]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static'] # Disable _static for now.

## -- Generate rts files from new python files --

# The directory relative to this file
source_dir = '../rbamlib/'
# The target directory for the generated rst files
target_dir = './'

def generate_rst_files(source_dir, target_dir, root_package_name='rbamlib'):
    """
    Generate RST files from Python modules in the source directory.

    This function traverses the source directory recursively, identifies Python packages (directories containing
    an `__init__.py` file), and generates corresponding RST files in the target directory. It ensures that
    each package and module is documented, and includes a toctree for subpackages to facilitate hierarchical
    documentation with Sphinx.

    The root package RST file is not generated to allow manual control over the main page (`index.rst`).

    Only functions imported in `__init__.py` are included in the documentation. Functions not present in
    `__init__.py` are ignored.

    Args:
        source_dir (str or Path): The root directory containing the Python source code.
        target_dir (str or Path): The directory where the generated RST files will be saved.
        root_package_name (str): The name of the root package (default is 'rbamlib').

    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        if '__init__.py' in files:
            process_package(root_path, source_dir, target_dir, dirs, files, root_package_name)

def process_package(root_path, source_dir, target_dir, dirs, files, root_package_name):
    """
    Process a Python package by generating its corresponding RST file.

    This function determines the module name based on the package's relative path to the source directory.
    It creates the necessary directories in the target directory and checks if an RST file already exists.
    If it exists, it checks for any missing functions and appends them. If it doesn't exist, it writes
    a new RST file.

    The root package RST file is not generated to allow manual control over the main page (`index.rst`).

    Args:
        root_path (Path): The current package directory path.
        source_dir (Path): The root directory of the Python source code.
        target_dir (Path): The directory where the RST files will be saved.
        dirs (list): A list of subdirectories in the current package directory.
        files (list): A list of files in the current package directory.
        root_package_name (str): The name of the root package.

    """
    # Calculate the relative path of the current package
    relative_root = root_path.relative_to(source_dir)
    # Build the module name by joining the relative path parts
    module_parts = [part for part in relative_root.parts if not part.startswith('__')]
    if module_parts:
        module_parts = [root_package_name] + module_parts
        module_name = '.'.join(module_parts)
    else:
        # Skip generating RST for the root package
        logging.info(f"Skipping root package: {root_package_name}")
        return

    # Determine the RST file path
    rst_dir = target_dir / relative_root.parent
    rst_filename = rst_dir / (module_parts[-1] + '.rst')

    # Create the necessary directories in the target directory
    rst_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing module: {module_name}")

    # Get the list of functions imported in __init__.py
    imported_functions = get_imported_functions(root_path)

    if rst_filename.exists():
        # If the RST file exists, check for missing functions
        logging.info(f"{rst_filename} exists. Checking for missing functions.")
        existing_functions = extract_documented_functions(rst_filename)
        logging.debug(f"Documented: {existing_functions}.")
        logging.debug(f"Imported: {imported_functions}.")
        missing_functions = [f for f in imported_functions if f not in existing_functions]
        if missing_functions:
            logging.info(f"Adding missing functions to {rst_filename}: {', '.join(missing_functions)}")
            append_functions_to_rst(rst_filename, module_name, missing_functions)
        else:
            logging.info(f"All functions are already documented in {rst_filename}.")
    else:
        # Write the RST file for the current package
        write_rst_file(rst_filename, module_name, imported_functions, dirs, root_path)

def get_imported_functions(root_path):
    """
    Extract the names of functions imported in __init__.py.

    Args:
        root_path (Path): The path to the package directory.

    Returns:
        list: A list of function names imported in __init__.py.

    """
    init_file = root_path / '__init__.py'

    logging.debug(f"Pacakge file: {init_file}")

    if not init_file.exists():
        return []
    with init_file.open('r', encoding='utf-8') as f:
        source = f.read()
    imported_functions = []
    try:
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                # Consider only relative imports (level > 0)
                if node.level > 0:
                    for alias in node.names:
                        imported_functions.append(alias.name)
    except SyntaxError as e:
        logging.warning(f"Syntax error when parsing {init_file}: {e}")
    return imported_functions

def extract_documented_functions(rst_filename):
    """
    Extract the list of functions that are already documented in the RST file.

    Args:
        rst_filename (Path): The path to the RST file.

    Returns:
        list: A list of function names that are already documented.

    """
    documented_functions = []
    # Regular expression to match '.. autofunction:: function_name'
    pattern = re.compile(r'\.\.\s+autofunction::\s+([\w\.]+)')
    with rst_filename.open('r') as rst_file:
        for line in rst_file:
            match = pattern.search(line)
            if match:
                function_full_name = match.group(1)
                # Extract the function name (the last part after '.')
                function_name = function_full_name.split('.')[-1]
                documented_functions.append(function_name)
    return documented_functions

def append_functions_to_rst(rst_filename, module_name, missing_functions):
    """
    Append missing functions to the existing RST file.

    Args:
        rst_filename (Path): The path to the RST file.
        module_name (str): The fully qualified name of the module.
        missing_functions (list): A list of missing function names to append.

    """
    with rst_filename.open('a') as rst_file:
        # Append a separator
        rst_file.write("\n")
        rst_file.write(".. # Added by generate_rst_files\n")
        rst_file.write("\n")
        for function in missing_functions:
            rst_file.write(f".. autofunction:: {module_name}.{function}\n")
    logging.info(f"Appended missing functions to {rst_filename}")

def write_rst_file(rst_filename, module_name, imported_functions, dirs, root_path):
    """
    Write the content of the RST file for a given Python package or module.

    This function creates an RST file that includes the module's title, a toctree for subpackages,
    and documentation directives for the module's contents and functions.

    Args:
        rst_filename (Path): The path to the RST file to be created.
        module_name (str): The fully qualified name of the module (e.g., 'rbamlib.subpackage').
        imported_functions (list): A list of function names imported in __init__.py.
        dirs (list): A list of subdirectories (subpackages) in the current package directory.
        root_path (Path): The path to the current package directory.

    """
    with rst_filename.open('w') as rst_file:
        # Write the module title
        title = module_name

        rst_file.write(f"..currentmodule:: {title}\n\n")
        rst_file.write(f"{title}\n")
        rst_file.write(f"{'-' * len(title)}\n\n")

        # Add the first line of documentation from __init__.py
        docstring = get_module_docstring(root_path)
        if docstring:
            rst_file.write(f"{docstring}\n\n")

        # Write the automodule directive
        rst_file.write(f".. automodule:: {module_name}\n\n")

        # Add automodsumm to generate function summary
        if imported_functions:
            rst_file.write(f".. automodsumm:: {module_name}\n\n")

        # Add a toctree for subpackages
        subpackage_dirs = [d for d in dirs if (root_path / d / '__init__.py').exists()]
        if subpackage_dirs:
            rst_file.write(".. toctree::\n")
            rst_file.write("   :maxdepth: 2\n")
            rst_file.write("   :caption: Subpackages: \n\n")
            for subdir in sorted(subpackage_dirs):
                if not subdir.startswith('__'):
                    rst_file.write(f"   {subdir}/{subdir}.rst\n")
            rst_file.write("\n")

        # Document each function in the module using autofunction directive
        if imported_functions:
            rst_file.write(".. rubric:: Functions\n")
            rst_file.write("   :heading-level: 2\n\n")
            for f in imported_functions:
                rst_file.write(f".. autofunction:: {module_name}.{f}\n")

def get_module_docstring(root_path):
    """
    Extract the first line of the module docstring from __init__.py.

    Args:
        root_path (Path): The path to the package directory.

    Returns:
        str: The first line of the module docstring, or None if not found.

    """
    init_file = root_path / '__init__.py'
    if not init_file.exists():
        return None
    with init_file.open('r', encoding='utf-8') as f:
        source = f.read()
    try:
        docstring = ast.get_docstring(ast.parse(source))
        if docstring:
            # Return the first line
            return docstring.split('\n', 1)[0]
    except SyntaxError as e:
        logging.warning(f"Syntax error when parsing {init_file}: {e}")
    return None

generate_rst_files(source_dir, target_dir)
