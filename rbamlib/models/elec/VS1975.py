import rbamlib.constants as const
import numpy as np

def VS1975_parameters_corr(planet='Earth', Omega=None, B0=None, R0=None):
    r"""
    Retrieve corotational parameters (Omega, B0, R0) for the specified planet.

    Parameters
    ----------
    planet : str, optional, default='Earth'
        Name of the planet.
    Omega : float, optional
        Angular rotation rate of the planet (in s⁻¹).
    B0 : float, optional
        Reference magnetic field (in Tesla).
    R0 : float, optional
        Planet radius (in meters).

    Returns
    -------
    tuple
        A tuple containing the following planetary parameters:
        - Omega (float): Angular rotation rate of the planet (in s⁻¹).
        - B0 (float): Reference magnetic field (in Tesla).
        - R0 (float): Planet radius (in meters).

    Notes
    -----
    Supported options are 'Earth' and 'Saturn'. Other planets will be added in future versions.
    """

    if planet == 'Earth':
        Omega1=const.Omega_Earth
        B01=const.SI.B0_Earth
        R01=const.SI.R_Earth
    elif planet == 'Saturn':
        Omega1=const.Omega_Saturn
        B01=const.SI.B0_Saturn
        R01=const.SI.R_Saturn
    else:
        raise ValueError("Unsupported planet.")

    parameters = zip((Omega, B0, R0), (Omega1, B01, R01))
    
    return tuple(val if val is not None else val1 for val, val1 in parameters)
    

def VS1975_Phi_corr(r, theta, Omega, B0, R0):
    r"""
    Compute the corotational electric potential for the Volland‐Stern :cite:t:`volland:1973,stern:1975` model.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    theta : float or ndarray
        Polar (colatitude) angle in radians.
    Omega : float
        Angular rotation rate of the planet (in s⁻¹).
    B0 : float
        Reference magnetic field (in Tesla).
    R0 : float
        Planet radius (in meters).

    Returns
    -------
    float or ndarray
        Corotational electric potential (in Volts)

    Notes
    -----

    .. math::
        \Phi_{corr} = -\frac{\Omega B_0 R_0^2 \sin^2(\theta)}{r}
    """

    return -Omega * B0 * R0**2 * np.sin(theta)**2 / r


def VS1975_E_corr(r, theta, Omega, B0, R0):
    r"""
    Compute the corotational electric field for the Volland‐Stern :cite:p:`volland:1973,stern:1975` model.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    theta : float
        Polar (colatitude) angle in radians.
    Omega : float
        Angular rotation rate of the planet (in s⁻¹).
    B0 : float
        Reference magnetic field (in Tesla).
    R0 : float
        Planet radius (in meters).

    Returns
    -------
    dict
        Dictionary representing the electric field components (in V/m) with keys:
           
           - 'r': Radial component, computed as :math:`E_{corr}(r) = -\frac{\Phi_{corr}}{r}`
           - 'phi': Azimuthal component, set to zero.
    
    Notes
    -----
    The corotational electric field is obtained by taking spatial derivative of the
    corotational potential computed by :func:`VS1975_Phi_corr`.
    """

    E_corr = dict(r=None, phi=None)
    E_corr['r'] = -VS1975_Phi_corr(r, theta, Omega, B0, R0) / r / R0    
    E_corr['r'] = np.array(E_corr['r'])
    E_corr['phi'] = np.zeros(E_corr['r'].shape)

    return E_corr


def VS1975_parameters_conv(planet='Earth', gamma=None, C=None, r0=None):
    r"""
    Retrieve convective parameters (gamma, C) for the specified planet.

    Parameters
    ----------
    planet : str, optional, default='Earth'
        Name of the planet.
    gamma : int or float, optional
        Exponent in the scaling law for the convective potential.
    C : float, optional
        Scaling constant for the convective potential.
    r0 : float, optional
        Radial distance from the center of the planet (in planet's radii) where the scaling constant `C` is defined.

    Returns
    -------
    tuple
        A tuple containing the following parameters:
        - gamma (float): Exponent in the scaling law for the convective potential.
        - C (float): Scaling constant for the convective potential (in V/m).
        - r0 (float): Radial distance where `C` is defined.

    Notes
    -----
    Supported options are 'Earth' and 'Saturn'. Other planets will be added in future versions.
    """


    if planet == 'Earth':
        gamma1=2
        C1=const.E0_Earth
        r01 = 1
    elif planet == 'Saturn':
        gamma1=0.5
        C1=const.E0_Saturn
        r01 = 5
    else:
        raise ValueError("Unsupported planet.")

    parameters = zip((gamma, C, r0), (gamma1, C1, r01))

    return tuple(val if val is not None else val1 for val, val1 in parameters)


def VS1975_Phi_conv(r, phi, theta, gamma, C, r0):
    r"""
    Compute the convective electric potential for the Volland‐Stern :cite:p:`volland:1973,stern:1975` model.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float or ndarray
        Azimuthal (longitude) angle in radians, (phi=0 is noon).
    theta : float or ndarray
        Polar (colatitude) angle in radians.
    gamma : int or float
        Exponent in the scaling law for the convective potential.
    C : float
        Scaling constant for the convective potential in V/m.
    r0 : float
        Radial distance from the center of the planet (in planet's radii) where the scaling constant `C` is defined. 


    Returns
    -------
    float or ndarray
        Convective potential (in Volts).

    Notes
    -----
    .. math::
        \Phi_{conv} = C \left(\frac{r}{r_0 \sin^2(\theta)}\right)^\gamma \sin(\phi),
    """

    return C * (r / (r0 * np.sin(theta)**2))**gamma * np.sin(phi)


def VS1975_E_conv(r, phi, theta, gamma, C, r0):
    r"""
    Compute the convective electric field for the Volland‐Stern :cite:p:`volland:1973,stern:1975` model.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float or ndarray
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float or ndarray
        Polar (colatitude) angle in radians.
    gamma : int or float
        Exponent in the scaling law for the convective potential.
    C : float
        Scaling constant for the convective potential in V/m.
    r0 : float
        Radial distance from the center of the planet (in planet's radii) where the scaling constant `C` is defined. 


    Returns
    -------
    dict
        Dictionary with the convective electric field components (in V/m) having keys:

           - 'r': Radial component, approximated as :math:`E_{conv}(r) = \frac{\gamma \Phi_{conv}}{r}`
           - 'phi': Azimuthal component, approximated as :math:`E_{conv}(\phi) = C \left(\frac{r}{r_0 \sin^2(\theta)}\right)^\gamma \frac{\cos(\phi)}{r}`

    Notes
    -----
    The convective electric field is obtained by taking spatial derivatives of the
    convective potential computed by :func:`VS1975_Phi_conv`.
    """

    E_conv = dict(r=None, phi=None)

    E_conv['r'] = VS1975_Phi_conv(r, phi, theta, gamma, C, r0) * gamma / r

    # Replace phi with pi/2 to make sin(phi) === 1 in the potental. In this case the formula for gradient is correct.    
    E_conv['phi'] = VS1975_Phi_conv(r, np.pi/2, theta, gamma, C, r0) * np.cos(phi) / r
    
    return E_conv


def VS1975(r, phi=0, theta=np.pi/2, planet='Earth', gamma=None, C=None, r0=None, Omega=None, B0=None, R0=None):
    r"""Compute the total electric field for the Volland‐Stern :cite:p:`volland:1973,stern:1975` model.

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float or ndarray, optional, default=0
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float or ndarray, optional, default=pi/2
        Polar (colatitude) angle in radians. Default is π/2 (equatorial plane).
    planet : str, optional, default='Earth'
        Name of the planet. If set to None, gamma, C, Omega, B0, R0 must be set.
    gamma : int or float, optional
        Exponent in the scaling law for the convective potential.
    C : float, optional
        Scaling constant for the convective potential in V/m.
    r0 : float, optional
        Radial distance from the center of the planet (in planet's radii) where the scaling constant `C` is defined.         
    Omega : float, optional
        Angular rotation rate of the planet (in s⁻¹).
    B0 : float, optional
        Reference magnetic field (in Tesla).
    R0 : float, optional
        Planet radius (in meters).

    Notes
    -----
    The total electric field is obtained as the sum of a convective component and a
    corotational component. The individual parts are defined by the following equations.

    **Convective potential:**

    .. math::
        \Phi_{conv} = C \left(\frac{r}{r_0 \sin^2(\theta)}\right)^\gamma \sin(\phi),

    - :math:`r` is the radial distance,
    - :math:`\theta` is the polar (colatitude) angle,
    - :math:`\phi` is the azimuthal angle,
    - :math:`\gamma` is an exponent parameter (default is 2),
    - :math:`C` is a scaling constant (default is ``const.E0_Earth``).
    - :math:`r_0` is the radial distance where :math:`C` is defined

    The convective electric field is then approximated by taking spatial derivatives of
    the convective potential, yielding:

    .. math::
       E_{conv}(r) = \frac{\gamma \Phi_{conv}}{r}, \quad
       E_{conv}(\phi) = C \left(\frac{r}{r_0 \sin^2(\theta)}\right)^\gamma \frac{\cos(\phi)}{r}.

    **Corotational potential:**

    .. math::
       \Phi_{corr} = -\frac{\Omega B_0 R_0^2 \sin^2(\theta)}{r},

    - :math:`\Omega` is the Earth's angular rotation rate (default is ``const.Omega_Earth``),
    - :math:`B_0` is the reference magnetic field (default is ``const.SI.B0_Earth``),
    - :math:`R_0` is the reference Earth radius (default is ``const.SI.R_Earth``).

    The corresponding corotational electric field (which affects only the radial
    component) is computed as

    .. math::
       E_{corr}(r) = -\frac{\Phi_{corr}}{r}, \left( E_{corr}(\phi) = 0 \right)

    **Total Electric Field:**

    .. math::
       E(r) = E_{conv}(r) + E_{corr}(r), \quad 
       E(\phi) = E_{conv}(\phi).

    .. warning::
       Only basic tests were impmeneted for this function.
       
    Returns
    -------
    dict
        Dictionary with the total electric field components (in V/m) having keys:
          
            - 'r': Total radial component.
            - 'phi': Azimuthal component.
    
    Examples
    --------
    **Example 1:** Default behavior (Earth).

    >>> E = VS1975(r=6.6, phi=np.pi/4, theta=np.pi/2)
    >>> print(E['r'], E['phi'])

    **Example 2:** Specify planet (Saturn).

    >>> E = VS1975(r=10, phi=0, theta=np.pi/2, planet='Saturn')

    **Example 3:** Provide explicit parameters.

    >>> E = VS1975(r=6.6, phi=np.pi/4, theta=np.pi/2, Omega=7.43e-5, B0=0.312e-4, R0=6.378e6, gamma=2, C=4.5/6.378e6)
    """

    gamma, C, r0 = VS1975_parameters_conv(planet, gamma, C, r0)
    Omega, B0, R0 = VS1975_parameters_corr(planet, Omega, B0, R0)

    E_conv = VS1975_E_conv(r, phi, theta, gamma, C, r0)
    E_corr = VS1975_E_corr(r, theta, Omega, B0, R0)

    E = E_conv
    E['r'] = E['r'] + E_corr['r']

    return E