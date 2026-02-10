# Changelog

All notable changes to rbamlib (Radiation Belts Analysis and Modeling Library) will be documented in this file.

## [26.02] - 2026-02-09

### Added

#### Conversion Functions
- `en2gamma`: Energy to relativistic gamma conversion
- `mlt2phi`: Magnetic local time to azimuthal angle conversion
- `phi2mlt`: Azimuthal angle to magnetic local time conversion

#### Motion Functions
- Gyro motion calculations: `f_gyro`, `omega_gyro`, `T_gyro`
- Bounce motion calculations: `f_bounce`, `omega_bounce`, `T_bounce`
- Drift motion calculations: `f_drift`, `omega_drift`, `T_drift`

#### Dipole Field Functions
- `al_lc`: Loss cone pitch angle calculation
- `tau_lc`: Lifetime based on quarter bounce period in loss cone

#### Empirical Models - Electron Density
- `CA1992`: Carpenter and Anderson (1992) model
- `S2001`: Sheeley et al. (2001) model
- `D2002`: Denton et al. (2002) model
- `D2004`: Denton et al. (2004) model
- `D2006`: Denton et al. (2006) model

#### Empirical Models - Electric Field
- `VS1975`: Volland (1973) and Stern (1975) convection electric field model

#### Empirical Models - Lifetime
- `G2012`: Gu et al. (2012) chorus wave lifetime model
- `W2024`: Wang et al. (2024) chorus wave lifetime lookup table model
- `O2016`: Orlova et al. (2016) hiss wave lifetime model

#### Web Data Access
- `download_unzip`: Utility for downloading and extracting model data files
- Enhanced `omni`: Improvements to OMNIWeb data retrieval interface

#### Utility Functions
- `idx`: Index search utility
- `storm_idx`: Storm index search tool
- `parse_datetime`: Date/time parsing utility
- `fixfill`: Data fill value correction utility

#### Other
- Additional physical constants for radiation belt studies
- Support for Jupiter and Saturn in dipole field calculations
- Comprehensive test coverage for all new functions

### Changed
- Complete documentation overhaul with BibTeX-based citation management
- All references now centrally managed in `docs/bibliography.bib`
- Enhanced symbol definitions in documentation

## [25.04] - 2025-04-01

### Added

#### Radial Diffusion Coefficient Models
- `BA2000`: Brautigam and Albert (2000) model
- `O2014`: Ozeke et al. (2014) model
- `A2016`: Ali et al. (2016) model
- `L2016`: Liu et al. (2016) model

#### Plasmapause Location Models
- `CA1992`: Carpenter and Anderson (1992) model
- `M2002`: Moldwin et al. (2002) model
- `OBM2003`: O'Brien and Moldwin (2003) model

#### Magnetopause Models
- `S1998`: Shue et al. (1998) model

#### Magnetic Field Models
- `TS2005_S`: Tsyganenko (2005) S-coefficients
- `TS2005_W`: Tsyganenko (2005) W-coefficients

#### Web Data Access
- `omni`: OMNIWeb data retrieval interface

#### Utility Functions
- `storm_idx`: Storm index search tool
- `parse_datetime`: Date/time parsing utility
- `fixfill`: Data fill value correction utility

#### Other
- Additional physical constants
- Extended test coverage

## [25.02] - 2025-02-01

### Added

#### Conversion Functions
- `en2pc`: Energy to momentum conversion
- `pc2en`: Momentum to energy conversion
- `Jcmu2K`: Second adiabatic invariant from bounce integral and first adiabatic invariant (also aliased as `Jc2K`)
- `Kmu2Jc`: Bounce integral from second adiabatic invariant and first adiabatic invariant (also aliased as `K2Jc`)
- `mural2pc`: Momentum from first adiabatic invariant, radius, and pitch angle (also aliased as `mu2pc`)
- `pcral2mu`: First adiabatic invariant from momentum, radius, and pitch angle (also aliased as `pc2mu`)
- `mural2en`: Energy from first adiabatic invariant, radius, and pitch angle (also aliased as `mu2en`)
- `enral2mu`: First adiabatic invariant from energy, radius, and pitch angle (also aliased as `en2mu`)
- `Lal2K`: Second adiabatic invariant from L-shell and pitch angle
- `LK2al`: Pitch angle from L-shell and second adiabatic invariant

#### Dipole Field Model Functions
- `B`: Magnetic field magnitude in dipole approximation
- `B0`: Equatorial magnetic field at given L-shell
- `T`: Auxiliary function T(α) for bounce integral calculations
- `Y`: Auxiliary function Y(α) for bounce integral calculations

#### Other
- Initial test framework
- Package structure for models (dll, lpp, mp, ne, pad, tau, etc.)
- Basic simulation support infrastructure

## [24.08] - 2024-08-01

### Added

#### Initial Release
- Core library structure and architecture
- Package organization: `conv`, `models`, `sim`, `utils`, `web`
- Basic documentation framework with Sphinx
- ReadTheDocs integration
- Testing infrastructure with unittest
- PyPI packaging configuration
- BSD-3-Clause license

---

*The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).*
