import rbamlib.constants as const
import numpy as np


def VS1975_Phi_corr(r, phi=0, theta=np.pi/2, Omega=const.Omega_Earth, B0=const.SI.B0_Earth, R0=const.SI.R_Earth):
    r"""
    Compute the corotational electric potential for the Volland‐Stern model (Volland (1973) [#]_ and Stern (1975) [#]_)

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float, optional, default=0
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float, optional, default=pi/2
        Polar (colatitude) angle in radians. Default is π/2 (equatorial plane).
    Omega : float, optional
        Angular rotation rate of the planet (in s⁻¹). Default is Earth
    B0 : float, optional
        Reference magnetic field (in Tesla). Default is Earth
    R0 : float, optional
        Planet radius (in meters). Default is Earth

    Returns
    -------
    float or ndarray
        Corotational electric potential (in Volts)

    Notes
    -----

    .. math::   
        \Phi_{corr} = -\frac{\Omega B_0 R_0^2 \sin^2(\theta)}{r}
            
    References
    ----------
    .. [#] Volland, H. (1973). A statically defined global electric field model. Planetary and Space Science, 21(6), 741–750.
    .. [#] Stern, D. P. (1975). Electric field models in the magnetosphere. Reviews of Geophysics and Space Physics, 13(3), 547–558.
    """

    return -Omega * B0 * R0**2 * np.sin(theta)**2 / r


def VS1975_E_corr(r, phi=0, theta=np.pi/2, Omega=const.Omega_Earth, B0=const.SI.B0_Earth, R0=const.SI.R_Earth):
    r"""
    Compute the corotational electric field for the Volland‐Stern model (Volland (1973) [#]_ and Stern (1975) [#]_).

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float, optional, default=0
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float, optional, default=pi/2
        Polar (colatitude) angle in radians. Default is π/2 (equatorial plane).
    Omega : float, optional
        Angular rotation rate of the planet (in s⁻¹). Default is Earth
    B0 : float, optional
        Reference magnetic field (in Tesla). Default is Earth
    R0 : float, optional
        Planet radius (in meters). Default is Earth

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
        
    References
    ----------
    .. [#] Volland, H. (1973). A statically defined global electric field model. Planetary and Space Science, 21(6), 741–750.
    .. [#] Stern, D. P. (1975). Electric field models in the magnetosphere. Reviews of Geophysics and Space Physics, 13(3), 547–558.
    """

    E_corr = dict(r=None, phi=None)
    E_corr['r'] = -VS1975_Phi_corr(r, phi, theta, Omega, B0, R0) / r / R0    
    E_corr['r'] = np.array(E_corr['r'])
    E_corr['phi'] = np.zeros(E_corr['r'].shape)

    return E_corr


def VS1975_Phi_conv(r, phi=0, theta=np.pi/2, gamma=2, C=const.E0_Earth):
    r"""
    Compute the convective electric potential for the Volland‐Stern model (Volland (1973) [#]_ and Stern (1975) [#]_).

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float, optional, default=0
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float, optional, default=pi/2
        Polar (colatitude) angle in radians. Default is π/2 (equatorial plane).
    gamma : int or float, optional, default=2
        Exponent in the scaling law for the convective potential.
    C : float, optional
        Scaling constant for the convective potential. Units depend on the value
        of :math:`\gamma`. For example, if :math:`\gamma = 2`, the units are V/m² if r is in m.
        Default is Earth.

    Returns
    -------
    float or ndarray
        Convective potential (in Volts).

    Notes
    -----

    .. math::
        \Phi_{conv} = C \left(\frac{r}{\sin^2(\theta)}\right)^\gamma \sin(\phi),

    References
    ----------
    .. [#] Volland, H. (1973). A statically defined global electric field model. Planetary and Space Science, 21(6), 741–750.
    .. [#] Stern, D. P. (1975). Electric field models in the magnetosphere. Reviews of Geophysics and Space Physics, 13(3), 547–558.
    """

    return C * (r / np.sin(theta)**2)**gamma * np.sin(phi)


def VS1975_E_conv(r, phi=0, theta=np.pi/2, gamma=2, C=const.E0_Earth):
    r"""
    Compute the convective electric field for the Volland‐Stern model (Volland (1973) [#]_ and Stern (1975) [#]_).

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float, optional, default=0
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float, optional, default=pi/2
        Polar (colatitude) angle in radians. Default is π/2 (equatorial plane).
    gamma : int or float, optional, default=2
        Exponent in the scaling law for the convective potential.
    C : float, optional
        Scaling constant for the convective potential. Units depend on the value
        of :math:`\gamma`. For example, if :math:`\gamma = 2`, the units are V/m² if r is in m.
        Default is Earth.

    Returns
    -------
    dict
        Dictionary with the convective electric field components (in V/m) having keys:

           - 'r': Radial component, approximated as :math:`E_{conv}(r) = \frac{\gamma \Phi_{conv}}{r}`
           - 'phi': Azimuthal component, approximated as :math:`E_{conv}(\phi) = \Phi_{conv}\frac{\cos(\phi)}{\sin(\phi)}`

    Notes
    -----
    The convective electric field is obtained by taking spatial derivatives of the 
    convective potential computed by :func:`VS1975_Phi_conv`.
    
    References
    ----------
    .. [#] Volland, H. (1973). A statically defined global electric field model. Planetary and Space Science, 21(6), 741–750.
    .. [#] Stern, D. P. (1975). Electric field models in the magnetosphere. Reviews of Geophysics and Space Physics, 13(3), 547–558.
    """

    E_conv = dict(r=None, phi=None)

    E_conv['r'] = VS1975_Phi_conv(r, phi, theta, gamma, C) * gamma / r
    
    # phi = np.array(phi) # Ensure that input is np.array for the purpuse zero case
    with np.errstate(divide='ignore', invalid='ignore'):
        E_conv['phi'] = VS1975_Phi_conv(r, phi, theta, gamma, C) * np.cos(phi) / np.sin(phi)
    
    if np.isscalar(E_conv['phi']) or np.size(E_conv['phi']) == 1:
        if phi == 0:
            E_conv['phi'] = 0
    else:
        mask = (phi == 0)
        E_conv['phi'][mask] = 0

    return E_conv


def VS1975(r, phi=0, theta=np.pi/2, gamma=2, C=const.E0_Earth, Omega=const.Omega_Earth, B0=const.SI.B0_Earth, R0=const.SI.R_Earth):
    r"""Compute the total electric field for the Volland-Stern model (Volland (1973) [#]_ and Stern (1975) [#]_)

    Parameters
    ----------
    r : float or ndarray
        Radial distance from the center of the planet (in planet's radii).
    phi : float, optional, default=0
        Azimuthal (longitude) angle in radians, (phi=0 is noon)
    theta : float, optional, default=pi/2
        Polar (colatitude) angle in radians. Default is π/2 (equatorial plane).
    gamma : int or float, optional, default=2
        Exponent in the scaling law for the convective potential.
    C : float, optional
        Scaling constant for the convective potential. Units depend on the value
        of :math:`\gamma`. For example, if :math:`\gamma = 2`, the units are V/m² if r is in m.
        Default is Earth.
    Omega : float, optional
        Angular rotation rate of the planet (in s⁻¹). Default is Earth
    B0 : float, optional
        Reference magnetic field (in Tesla). Default is Earth
    R0 : float, optional
        Planet radius (in meters). Default is Earth

    Notes
    -----
    The total electric field is obtained as the sum of a convective component and a
    corotational component. The individual parts are defined by the following equations.

    **Convective potential:**

    .. math::
        \Phi_{conv} = C \left(\frac{r}{\sin^2(\theta)}\right)^\gamma \sin(\phi),

    - :math:`r` is the radial distance,
    - :math:`\theta` is the polar (colatitude) angle,
    - :math:`\phi` is the azimuthal angle,
    - :math:`\gamma` is an exponent parameter (default is 2),
    - :math:`C` is a scaling constant (default is ``const.E0_Earth``).

    The convective electric field is then approximated by taking spatial derivatives of
    the convective potential, yielding:

    .. math::
       E_{conv}(r) = \frac{\gamma \Phi_{conv}}{r}, \quad
       E_{conv}(\phi) = \Phi_{conv} \frac{\cos(\phi)}{\sin(\phi)} \quad (\phi \neq 0).

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
       E(r) = E_{conv}(r) + E_{corr}(r), \quad E(\phi) = E_{conv}(\phi).

    Returns
    -------
    dict
        Dictionary with the total electric field components (in V/m) having keys:
          
            - 'r': Total radial component.
            - 'phi': Azimuthal component.
    
    References
    ----------
    .. [#] Volland, H. (1973). A statically defined global electric field model. Planetary and Space Science, 21(6), 741–750.
    .. [#] Stern, D. P. (1975). Electric field models in the magnetosphere. Reviews of Geophysics and Space Physics, 13(3), 547–558.
    """

    E_conv = VS1975_E_conv(r, phi, theta, gamma, C)
    E_corr = VS1975_E_corr(r, phi, theta, Omega, B0, R0)

    E = E_conv
    E['r'] = E['r'] + E_corr['r']

    return E