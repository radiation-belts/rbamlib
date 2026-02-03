"""
The `vf` (Various Functions) package us been renamed to `utils`.

Main Features:
- `idx`: Get the index of val from array arr
- `parse_datetime`: Parses an input from various formats into a datetime
- `storm_idx`: Identify storms in Dst based
- `fixfill`: Fix invalid values in array
"""

import warnings
import sys
import rbamlib.utils as utils  # Redirect to the new package

warnings.warn(
    "The 'vf' package has been renamed to 'utils'. Please update your imports.",
    DeprecationWarning,
    stacklevel=2
)

sys.modules['rbamlib.vf'] = utils  # Make `rbamlib.vf` an alias for `rbamlib.utils`
