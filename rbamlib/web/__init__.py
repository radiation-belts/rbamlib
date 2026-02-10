"""
The `web` package provides an interface for retrieving data from online sources.

This package allows users to access geomagnetic indices, solar wind parameters, and other space weather data
from various providers, such as OMNIWeb.

Main Features:
    - `omni`: Retrieval of OMNIWeb data.
    - `download_unzip`: Utility for downloading and extracting model data files.
"""

from .omni import omni
from .download_unzip import download_unzip
