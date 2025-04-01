"""
The `vf` (Various Functions) package us been renamed to `utils`.
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
