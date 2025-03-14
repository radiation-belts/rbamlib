"""
The `web` package provides an interface for retrieving data from online sources.

This package allows users to access geomagnetic indices, solar wind parameters, and other space weather data
from various providers, such as OMNIWeb.

Interfaces:
-----------
- `omni`: Retrieval of OMNIWeb data.
"""

from .omni import omni