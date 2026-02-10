"""
The `motion` provides functions for representation of charged particle motion in electric and magnetic field.

Main Features:
    - Oscilation frequencies (gyro, bounce, and drift)
    - Angular Frequencies (gyro, bounce, and drift)
    - Motion period (gyro, bounce and drift)
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