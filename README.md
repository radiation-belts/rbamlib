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

- **System Properties**: Adiabatic invariants transformation, drift velocities, phase space density, and motion periods calculation, adaptable for different planets.
- **Empirical Models Collection**: Radial diffusion coefficients, lifetimes, local diffusion coefficients scaling, plasma densities, and more.
- **Modeling Support**: Conversion between simulation grids, boundary and initial condition characteristics, adiabatic transformation of boundary scaling factors.

## Architecture
The library is architected into Python packages, acting as modules containing multiple functions organized in separate files. 
This design allows for direct function imports from the package, streamlining usage without the need to reference specific files.

For example, `pc2en` function from `conv` package is located in `conv/pc2en.py` file. To use it simply import it as follows:

```python
import rmamlib.conv.pc2en
```

The primary function that users should utilize is named after the file itself, ensuring intuitive access. If this main
function relies on any helper functions, they are located within the same file to maintain coherence. These helper
functions are kept distinct to facilitate targeted testing and validation.

## Development and Contribution

The library is being developed in compliance with the Heliophysics Community (PyHC) Standards and HP Data Policy. It
will be documented, tested with a planned release on Python Package Index (PyPI).

### How to Contribute

The contributions from the community as welcomed!
If you're interested in contributing, please see CONTRIBUTING.md.

## Installation and Usage

Instructions on how to install and use `rbamlib` will be provided upon release.

At this moment, you can to install library you can clone the repository:

```bash
git clone https://github.com/radiation-belts/rbamlib.git
```

Recently, the package become available to install using PyPI.
```bash
pip install rbamlib
```

## Documentation
For more information, please see our documentation at: 

https://rbamlib.readthedocs.io/

## License

`rbamlib` is released under the BSD-License (3-clause version). See the LICENSE file for details.
