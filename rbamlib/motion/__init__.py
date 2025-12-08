"""
The 'motion' provides functions for representation of charged particle motion in electric and magnetic field.

Main Features:
    - Frequencies (gyro, bounce, and drift) [TBD]
    - Motion period (gyro, bounce and drift-times) [TBD]
    - Drift velocities calculation [TBD]
"""

from .gyro import f_gyro
from .gyro import omega_gyro
from .gyro import T_gyro

from .bounce import f_bounce
from .bounce import omega_bounce
from .bounce import T_bounce

from .drift import f_drift
from .drift import omega_drift
from .drift import T_drift