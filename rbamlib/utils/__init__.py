"""
The `utils` package provides supporting functions.

This package provides a collection of utility functions for various purposes.
It is designed to be expanded with more functions over time.

Main Features:
    - `idx`: Get the index of val from array arr
    - `parse_datetime`: Parses an input from various formats into a datetime
    - `storm_idx`: Identify storms in Dst based
    - `fixfill`: Fix invalid values in array
"""

from .idx import idx
from .storm_idx import storm_idx
from .parse_datetime import parse_datetime
from .fixfill import fixfill
