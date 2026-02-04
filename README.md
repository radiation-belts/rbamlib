# Radiation Belts Analysis and Modeling Library (rbamlib)

## Overview

`rbamlib` is a lightweight, open-source Python library for the analysis and modeling of radiation belts.

The library aims to support the scientific community by providing essential functionalities for radiation belt studies, with detailed [documentation](https://rbamlib.readthedocs.io/).
This includes system properties calculations, empirical model collections, and modeling support.

> [!IMPORTANT]
> **This library is currently in active development.** 
> 
> Some functions are placeholders and may not yet have full implementations. Expect ongoing updates and new features as the library evolves.

[![Documentation Status](https://readthedocs.org/projects/rbamlib/badge/?version=latest)](https://rbamlib.readthedocs.io/latest/?badge=latest)

## Planned Key Features

- **System Properties**: Adiabatic invariants transformation, drift velocities, phase space density, and motion period calculations, adaptable for different planets.
- **Empirical Models Collection**: Radial diffusion coefficients, lifetimes, plasma densities, and more.
- **Modeling Support**: Conversion between simulation grids, boundary and initial condition characteristics, adiabatic transformation of boundary scaling factors, local diffusion coefficients scaling.

## Architecture
The library is architected into Python packages, acting as modules containing multiple functions organized in separate files. 
This design allows for direct function imports from the package, streamlining usage without the need to reference specific files.

For example, `pc2en` function from `conv` package is located in `conv/pc2en.py` file. To use it simply import it as follows:

```python
from rbamlib.conv import pc2en
```

The primary function that users should utilize is named after the file itself, ensuring intuitive access. If this main
function relies on any helper functions, they are located within the same file to maintain coherence. These helper
functions are kept distinct to facilitate targeted testing and validation.

## Development and Contribution

The library is being developed in compliance with the Heliophysics Community (PyHC) Standards and HP Data Policy. It
will be documented, tested with a planned release on Python Package Index (PyPI).

### How to Contribute

The contributions from the community are welcomed!
If you're interested in contributing, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Installation

The package is available on PyPI:

```bash
pip install rbamlib
```

Alternatively, you can install from source by cloning the repository:

```bash
git clone https://github.com/radiation-belts/rbamlib.git
cd rbamlib
pip install -e .
```

## Documentation

Please see documentation at [https://rbamlib.readthedocs.io/](https://rbamlib.readthedocs.io/).

## Acknowledgements

The original development of the library was supported by NASA grant 80NSSC24K0462.

Special thanks to the [PlasmaPy](https://github.com/PlasmaPy/PlasmaPy) for inspiring with their approach to building open-source scientific software. We are also grateful to the Python in Heliophysics Community (PyHC) for their guidance.

## License

`rbamlib` is released under the BSD-License (3-clause version). See the LICENSE and NOTICE files for details.
