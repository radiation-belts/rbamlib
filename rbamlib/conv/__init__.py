"""
The 'conv' (conversion) provides tools for the conversion between various physical quantities pertinent to particle physics.

This module includes handling  the system properties calculation and conversion such as
adiabatic invariant calculations and facilitating unit transformations in radiation belt studies.

Main Features:
    - Calculation of adiabatic invariants.
    - Energy and momentum conversion (en2pc, pc2en).
    - Conversion between Jc and K (Jcmu2K, Kmu2Jc).
    - Conversion between mu and pc (mural2pc/mu2pc, pcral2mu/pc2mu).
"""
from .en2pc import en2pc
from .pc2en import pc2en
from .Jcmu2K import Jcmu2K
from .Kmu2Jc import Kmu2Jc

from .mural2pc import mural2pc
from .mural2pc import mu2pc

from .pcral2mu import pcral2mu
from .pcral2mu import pc2mu

from .mural2en import mural2en
from .mural2en import mu2en

from .enral2mu import enral2mu
from .enral2mu import en2mu
