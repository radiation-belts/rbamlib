"""
`rbamlib` is a lightweight, open-source Python library for the analysis and modeling of radiation belts.

The library is architected into Python packages, each acting as a Python module. The packages include functions in
separate Python files. Thus, the function is imported directly from the package without specifying the file name.

Please see README.md for more information.
"""

# Importing submodules into the package namespace for easier access.
# This allows users to access submodules directly via `rbamlib.<module_name>`
# without needing to import them separately.
# Example:
# import rbamlib
# rbamlib.utils  # Access
from rbamlib import (
    conv,
    models,
    motion,
    sim,
    utils,
    web,
    constants
)
