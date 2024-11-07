# Developer's Guide

Welcome to the [rbamlib project](README.md)! This guide provides information for contributors. 

## Contribution Process

For guidelines on contributing to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md).

Please ensure all code submissions and contributions adhere to the [PyHC Coding Standards](https://github.com/heliophysicsPy/standards/blob/v1.0/standards.md).

## Project Structure

The `rbamlib` library is organized into Python packages, each serving as a module containing related functions.
Note, the library is design to allow direct function imports from the package without referencing specific pythong files (e.g., Python modules).

**Core Components:**

- **`rbamlib/`**: Contains the main packages and sub-packages and modules of the library.
  - **Conversion Module (`conv/`)**: Functions for various physical conversions.
  - **Motion Module (`motion/`)**: Functions related to particle motion.
  - **Empirical Models (`models/`)**: Implementations of empirical models, such as radial diffusion coefficients.
  - **Simulation Support Module (`sim/`)**: Routine tasks for radiation bells models such as VERB code.
  - **Utilities (Various functions) (`vf/`)**: Helper functions and utilities used across the library.

**Support for different planets [TBD]**
- The core components are designed to be ether universal or related to Earth's radiation belts
- The support for other planets must follow similar core components structure but withing a subpackage of specific planet (`rbamlib/jupiter`)

**Adding New Empirical Models:**

When introducing a new empirical model, typically based on a specific research paper, follow these steps:

1. **Create a New Module**: Add a Python file in the appropriate sub-package (e.g., `models/lpp`), named after the authors and year of the paper (e.g., `CA1992.py` for Carpenter and Anderson, 1992).
2. **Implement the Model**: Define the model's functions within this Python files (e.g., Python module).
3. **Update sub-package `__init__.py`**:
   - Include a reference to the corresponding paper, such as "Carpenter and Anderson (1992)".
   - Import the new model to make it accessible at the package level:
    ```python
     from .CA1992 import CA1992
    ```
4. **Documentation**: Include documentation and updated corresponding .rst files located in `doc/` folder. 
5. **Testing**: Implement testing for the new functionality in `tests/` folder. Test-driven development approach is preferable.
     

## Naming Conventions

- **Packages**: Use lowercase letters (e.g., `conv`). The name of the packages and sub-packages should be designed to be short and relatable to the content. Use common definitions or abbreviations (e.g., `dll` for radial diffusion coefficent, 'lpp' for plasmapause location) 
- **Functions based on papers**: Use capital letter of the authors names and year of the paper (e.g., `def CA1992()` for Carpenter and Anderson, 1992). Use single capital letter for the paper with many authors (e.g., `def D2017()` for `Drozdov et al., (2017)`). Note, the corresponding Python module (file) must have the same name. The supporting function that related to the same paper can be defined locally within the same file.    
- **Symbol**: Refer to [`symbols.rst`](docs/symbols.rst). Use uppercase only for commonly defined physical variables with capital letters, such as `L` for L-shell, `K` for second adiabatic invariant or `B` for magnetic field. Use lowercase for other commonly defined variables, such as `p` for momentum. Use two or more lowercase letter for other variable to avoid ambiguity or common greek letter, such as `mu` - for first adiabatic invariant `en` - for energy, or `al` - for pitch angle or alpha.     
- **Convertion Functions**: Use lowercase or uppercase without underscores for general variables to convert from, follow by `2` and general variable to convert to (e.g., `Lal2K` or `enral2mu`). If one of physical variable can be omitted as an input (e.g., common default values) it can be also omitted in the functions name. However, creating an additional alias is preferable in such case. 
- **Aliases**: The corresponding alias can be created to use fewer variables in the name of the function (e.g., `enral2mu` and `en2mu` are equivalent), see example:
  ```python
    def en2mu(en, r, al=np.pi/2, *B0):
        r"""
        Convert energy in MeV to first adiabatic invariant, mu, in MeV/G.
    
        See Also
        --------
        enral2mu: Alias of the `enral2mu` function.
        """
    
        return enral2mu(en, r, al, *B0)
  ```
- **Other Functions and Variables [TBD]**: Use lowercase with underscores (e.g., `drift_v`). Similar to packages and sub-packages, the names should be designed to be short and relatable to the content. Use common definitions or abbreviations. 
- **Classes**: Use CamelCase (e.g., `RadiationModel`).

## Documentation

The **NumPy/SciPy** docstring style should be used. This format ensures automated documentation generation using Sphinx.

**Docstring Style:**
- **Docstrings**: Use triple double-quoted strings for module, class, and function docstrings. Use `r` strings (denoted by a prefix `r` before the string: `r"""`) to avoid conflicts with backslashes in equations. 
- **Summary**: A brief description of the function's purpose.
- **Parameters**: A list of input parameters with their types, descriptions and units.
- **Returns**: Details of the output, including type, description and units.
- **Notes**: A references to relevant paper.
- **Math Expressions**: Mathematical formulas rendered using LaTeX syntax.
- **See Also** (optional): Add this section for related functions or in aliases. 

**Example:**
```python
def T(al):
    r"""
    Approximation of the integral function T related to the bounce period, derived in the dipole approximation.

    Parameters
    ----------
    al : float or ndarray
         Equatorial pitch angle, in radians.

    Returns
    -------
    float or ndarray
         Value of T

    Notes
    -----
    See Schulz & Lanzerotti (1974) [#]_.

    .. math::
        T( \\alpha ) \\approx T_0 - \\frac{1}{2}(T_0 - T_1) \\cdot \\left( \\sin( \\alpha ) + \\sin( \\alpha)^1/2 \\right)
        
    Reference
    ---------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts (Vol. 7). Springer-Verlag Berlin Heidelberg. Retrieved from http://www.springer.com/physics/book/978-3-642-65677-4
    """
```

**Guidelines:**
- **Input and output**: Use commonly defined physical variables names and units. Refer to [`symbols.rst`](docs/symbols.rst). Always define units when applicable. 
- **Mathematical Expressions**: Use LaTeX syntax within the `.. math::` directive to render equations properly in Sphinx-generated documentation in the **Notes** section.
- **References**: When implementing functions based on specific research papers, include a citation in the **Reference** section, providing full bibliographic details. Use Sphinx directive ` [#]_ ` to define the reference number and `.. [#] ` directive to place the reference. Note, you can use multiple reference in the description of the function. 
- **Aliases**: For aliases, use only a one line summary (see example in Naming Conventions section). Add See Also section with the original function.  
- **`__init__.py`**: When describing the package or a sub-package, start with the name of the pacakge using `'`, explanation of its name and what it provides. Add the description and list of the **Main Features**. In the sub-package, include list of models using a short reference to the papers.   
**Examples**

Package:
```python
"""
The 'conv' (conversion) provides tools for the conversion between various physical quantities pertinent to particle physics.

This module includes handling  the system properties calculation and conversion such as
adiabatic invariant calculations and facilitating unit transformations in radiation belt studies.

Main Features:
    - Calculation of adiabatic invariants.
    - Energy and momentum conversion (en2pc, pc2en).
    ...
""" 
```

Sub-package of the models:
```python
"""
The `lpp` provides functionalities for calculating the plasmapause location.

Models:
    - Carpenter and Anderson (1992)
    - O’Brien and Moldwin (2003)
    - Moldwin et al., (2002)
"""
```

### Sphynx documentation
We utilize [Sphinx](https://www.sphinx-doc.org/) to generate documentation. This allows the automatic inclusion of docstrings from the code and synchronize documentation with the source code.

**Sphynx Documentation Organization**
  - **Location**: The Sphinx documentation is organized in `docs/` folder and mirrors the project's core structure.
  - **Structure**: Each sub-package is represented by corresponding reStructuredText (`.rst`) file with the same name and additional folder for the next level structure.  

**Guidelines:**
- **Update the Respective `.rst` File**:
  - Navigate to the appropriate `.rst` file within the `docs/` directory that corresponds to your sub-package.
  - If a corresponding `.rst` file does not exist (new sub-package), create one using the same name.
- **Add the Function to the `.rst` File**:
   - The `.rts` file should start with `currentmodule` directive of its sub-package:
     ```rst
     .. currentmodule:: rbamlib.your_package_name
     ```
   - Use the `automodule` directive to include description for new sub-packages. This will automatically pull docstring from `__init__.py`.
     ```rst
     Your Package Name
     -----------------
     .. automodule:: rbamlib.your_package_name
     ```
   - Use the `autofunction` directive for new functions. This will automatically pull in the function's docstring.
     ```rst
     Functions
     =========
    
     .. autofunction:: new_function
    
     Aliases
     =======
    
     .. autofunction:: new_function_alias 
     ```
- **Update the Table of Contents**:
   - If you've added a new `.rst` file (corresponding to your new sub-package), for example into `models` package, ensure it's included in the project's table of contents via `toctree` directive. 
   - Edit the corresponding `rst` file, for example `module.rst`, to include a reference to your new file `your_package_name.rst`:
     ```rst
      .. currentmodule:: rbamlib.models
      
      Models
      ------
      .. automodule:: rbamlib.models
      
      .. toctree::
         :maxdepth: 1
         :caption: Models
      
         models/dip
         models/your_package_name
         ...
     ```
   - If your sub-package has additional structure level, make sure that the corresponding table of content is also created. 

## Testing

We utilize Python's built-in `unittest` framework for writing and executing tests.

**Test Organization:**

- **Location**: All test modules are located in the `tests/` directory.
- **Structure**: The `tests` directory mirrors the structure of the `rbamlib` package, however Python test files corresponding to each sub-package (module).

**Writing Tests:**

- **Naming Conventions**:
  - **Test Files**: Name test files starting with `test_` followed by the package name (e.g., `test_conv.py`). The test should be located in the corresponding test sub-package mirroring the structure or the library (e.g., `tests/models/test_lpp.py`). 
  - **Test Classes**: Name test classes with `Test` followed by the package name but using CamelCase (e.g., `TestConv`).
  - **Test Methods**: Prefix test method names with `test_` (e.g., `test_en2pc_single_value`).
- **Content**:
  - **Methods**: Inspect and extend `setUp()` and `teadDown()` methods (if any) to prepare a common parameters. Inspect additional supporting methods that may be included in the class. Be consistent with the provided testing framework and workflow.  
  - **Test**: Call the function or method under test. The minimum test must include the value validation of the function's expected output. 
  - **Assertions**: Use `unittest`'s assertion methods to verify expected outcomes or local existing test methods. Provide the error message of the unexpected test.

**Examples of Test Class:**

Example of the single value test with existing method:
```python
import unittest
from rbamlib.conv import pc2en

class ConvTest(unittest.TestCase):
    def setUp(self):
        # Input values
        self.in_float = 0.1
        
        # Expected values
        self.res_pc_float = 0.3350
    
    def assertSingleValue(self, function, input, expected):
        """Assert function works for single float values"""
        result = function(input)
        self.assertIsInstance(result, float, "Result should be a float for single value input")
        self.assertAlmostEqual(result, expected, places=4, msg="Single value output incorrect")
    
    # Tests for en2pc
    def test_en2pc_single_value(self):
        self.assertSingleValue(en2pc, self.in_float, self.res_pc_float)

if __name__ == '__main__':
    unittest.main()
```

Example of the array values assertion:

```python
import unittest
import numpy as np
from rbamlib.models.lpp import CA1992


class TestModelsLpp(unittest.TestCase):
    def test_CA1992_values(self):
        """Test the CA1992 function with known values."""
        time = np.array([0, 1, 2])  # Example time in days
        kp = np.array([1, 2, 3])  # Example Kp-index values
        expected_output = np.array([5.14, 4.68, 4.22])  # Expected plasmapause locations

        # Call the CA1992 function
        result = CA1992(time, kp)

        # Assert that the result is as expected
        np.testing.assert_almost_equal(result, expected_output, decimal=2,
                                       err_msg="CA1992 did not return expected values.")

if __name__ == '__main__':
    unittest.main()
```

**Running Tests:**

To execute all tests, navigate to the project's root directory and run:

```bash
python -m unittest discover tests
```

This command will discover and run all test files in the `tests/` directory.


## Licensing and Dependencies

- **License**: This project is licensed under the BSD-3-Clause License.
- **Dependencies**:
  - The project aims to maintain minimal dependencies, currently limited to `numpy` and `scipy`.
  - If additional dependencies are necessary, please discuss them by opening an issue before implementation.