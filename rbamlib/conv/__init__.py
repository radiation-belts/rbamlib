"""
The 'conv' (conversion) provides tools for the conversion between various physical quantities pertinent to particle physics.

This module includes handling  the system properties calculation and conversion such as
adiabatic invariant calculations and facilitating unit transformations in radiation belt studies.

Main Features:
    - Calculation of adiabatic invariants.
    - Energy and momentum conversion (en2pc, pc2en).
"""
from .en2pc import en2pc
from .pc2en import pc2en
from .Jcmu2K import Jcmu2K
from .Kmu2Jc import Kmu2Jc

