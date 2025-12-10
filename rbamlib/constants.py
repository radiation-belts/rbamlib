"""
Physical constants for use in `rbamlib` calculations. Most of the constants are defined in CGS system with a few exceptions.

Constants:
    - MC2 ≈ 0.511 : rest mass energy of an electron in MeV.
    - B0_Earth ≈ 0.31 : mean value of the magnetic field at the magnetic equator on the Earth's surface in G.
    - B0_Saturn ≈ 0.21 : mean value of the magnetic field at the magnetic equator on the Saturn's surface in G.
    - B0_Jupiter ≈ 4.28 : mean value of the magnetic field at the magnetic equator on the Jupiter's surface in G.
    - T0 ≈ 1.3802 : approximate value of T(0)
    - T1 ≈ 0.7405 : approximate value of T(1)
    - R_Earth ≈ 637,800,000 : radius of Earth in cm
    - R_Saturn ≈ 6,026,800,000 : radius of Earth in cm
    - R_Jupiter ≈ 7,149,200,000 : radius of Earth in cm
    - c ≈ 2.9e10 : speed of light in cm/s
    - q ≈ 4.8e-10 : electron charge in CGS
    - me ≈ 0.91e-21 electrons mass in g in CGS

CGS:
    - Class of constants in CGS

SI:
    - Class of constants in SI
"""

from scipy import constants as scic

MC2 = 0.51099895  # rest mass energy of an electron in MeV
B0_Earth = 0.312  # mean value of the Earth's surface magnetic field in Gauss
B0_Saturn = 0.215  # mean value of the Saturn's surface magnetic field in Gauss
B0_Jupiter = 4.28  # mean value of the Jupiter's surface magnetic field in Gauss
T0 = 1.3802  # Approximate value of T(0)
T1 = 0.7405  # Approximate value of T(1)

Omega_Earth = 7.43e-5 # Earth rotation frequency, 1/s 
Omega_Saturn = 1.62e-4 # Saturn rotation frequency, 1/s 

E0_Earth = 4.5 / 6.378e6 # V/m
E0_Saturn = 0.3E-3 # V/m

# Based on NASA Planetary Fact Sheet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/
R_Earth = 6.378e8  # Earth radius in cm
R_Saturn = 60.268e8  # Saturn radius in cm
R_Jupiter = 71.492e8  # Jupiter radius in cm
c = scic.c * 100  # Speed of light in cm/s
q = 4.803e-10  # charge of electron in CGS
me = 0.91e-27 # Electrons mass in g in CGS


class CGS:
    MC2 = MC2 * 1e6 * scic.eV / scic.erg  # erg (electron rest energy)
    B0_Earth = B0_Earth
    B0_Saturn = B0_Saturn
    B0_Jupiter = B0_Jupiter
    R_Earth = R_Earth
    R_Saturn = R_Saturn
    R_Jupiter = R_Jupiter
    c = c
    q = q
    me = me


class SI:
    MC2 = CGS.MC2 * scic.erg  # Joules (1 erg = 1e-7 J)
    B0_Earth = CGS.B0_Earth * 1e-4  # Tesla
    B0_Saturn = CGS.B0_Saturn * 1e-4  # Tesla
    B0_Jupiter = CGS.B0_Jupiter * 1e-4  # Tesla
    R_Earth = CGS.R_Earth * 1e-2  # meters
    R_Saturn = CGS.R_Saturn * 1e-2  # meters
    R_Jupiter = CGS.R_Jupiter * 1e-2  # meters
    c = scic.c  # m/s
    q = scic.e  # Coulombs
    me = scic.electron_mass # kg
