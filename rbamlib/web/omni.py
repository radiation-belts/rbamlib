import requests
import numpy as np
import datetime as dt
import re

data_mapping = {
    "BRN": 3,               # Bartels Rotation Number
    "IMF_ID": 4,            # IMF Spacecraft ID
    "Plasma_ID": 5,         # Plasma Spacecraft ID
    "IMF_Fine": 6,          # Fine Scale Points in IMF Averages
    "Plasma_Fine": 7,       # Fine Scale Points in Plasma Averages
    "IMF_Mag": 8,           # IMF Magnitude Average, nT
    "IMF_Vr": 9,            # Magnitude, Avg IMF Vr, nT
    "IMF_Lat": 10,          # Latitude of Avg. IMF, degrees
    "IMF_Lon": 11,          # Longitude of Avg. IMF, degrees
    "Bx_GSE": 12,           # Bx component, GSE/GSM, nT
    "By_GSE": 13,           # By component, GSE, nT
    "Bz_GSE": 14,           # Bz component, GSE, nT
    "By_GSM": 15,           # By component, GSM, nT
    "Bz_GSM": 16,           # Bz component, GSM, nT
    "Sigma_IMF_Mag": 17,    # Sigma in IMF Magnitude Average
    "Sigma_IMF_Vec": 18,    # Sigma in IMF Vector Average
    "Sigma_Bx": 19,         # Sigma Bx, nT
    "Sigma_By": 20,         # Sigma By, nT
    "Sigma_Bz": 21,         # Sigma Bz, nT
    "T_p": 22,              # Proton Temperature, K
    "N_p": 23,              # Proton Density, n/cc
    "V_flow": 24,           # Flow Speed, km/sec
    "Flow_Lon": 25,         # Flow Longitude, degrees
    "Flow_Lat": 26,         # Flow Latitude, degrees
    "Alpha_Proton": 27,     # Alpha/Proton Density Ratio
    "Flow_P": 28,           # Flow Pressure, nPa
    "Sigma_T": 29,          # Sigma-T
    "Sigma_Np": 30,         # Sigma-Np
    "Sigma_V": 31,          # Sigma-V
    "Sigma_Flow_Lon": 32,   # Sigma-Flow-Longitude
    "Sigma_Flow_Lat": 33,   # Sigma-Flow-Latitude
    "Sigma_Alpha": 34,      # Sigma-Alpha/Proton Ratio
    "Ey": 35,               # Electric Field, mV/m
    "Plasma_Beta": 36,      # Plasma Beta
    "Alfven": 37,           # Alfven Mach Number
    "Kp": 38,               # Kp*10 Index
    "R_Sunspot": 39,        # R Sunspot Number
    "Dst": 40,              # Dst Index, nT
    "AE": 41,               # AE Index, nT
    "PC": 51,               # Polar Cap (PCN) index from Thule
    "AL": 52,               # AL Index, nT
    "AU": 53,               # AU Index, nT
    "AP": 49,               # AP index, nT
    "F10.7": 50,            # Solar index F10.7
    "Lyman_Alpha": 55,      # Lyman Alpha Solar Index, W/m²
    "PFlux_1MeV": 42,       # Proton Flux > 1 MeV
    "PFlux_2MeV": 43,       # Proton Flux > 2 MeV
    "PFlux_4MeV": 44,       # Proton Flux > 4 MeV
    "PFlux_10MeV": 45,      # Proton Flux > 10 MeV
    "PFlux_30MeV": 46,      # Proton Flux > 30 MeV
    "PFlux_60MeV": 47,      # Proton Flux > 60 MeV
    "Mag_Flux_Flag": 48,    # Magnetospheric Flux Flag
    "Magnetosonic": 54,     # Magnetosonic Mach Number
    "Proton_QI": 56         # Proton Quasi-Invariant (QI)
}


def omni(start_date, end_date, params=None):
    r"""
    Retrieve hourly-averaged space weather data from the Low Resolution OMNIWEB (LRO) database.

    Parameters
    ----------
    start_date : datetime or str
        Start date in 'YYYY-MM-DD' format or as a datetime object.
    end_date : datetime or str
        End date in 'YYYY-MM-DD' format or as a datetime object.
    params : set
        Set of parameter names (e.g., {'Kp', 'Dst'}) or numbers (e.g., {38, 'Dst'}).
        If a string is provided, it is mapped to the corresponding OMNIWEB variable number.
        If a number is provided, it is directly used in the request.
        Raises an error if an unknown parameter is included.

    Returns
    -------
    tuple
        If multiple parameters are requested, returns (time, *data_values), where data_values
        are ordered according to the requested parameter keys.

    Notes
    -----
    - The data is retrieved from the NASA OMNIWEB database and includes various space weather
      indices, such as Kp, Dst, AE, and solar indices.
    - The function automatically converts the retrieved data into numpy arrays and filters them
      by the specified date range.
    - Data is provided in hourly resolution.
    - The OMNIWEB service may return missing or fill values for certain periods.
    - Parameter names are case-sensitive.

    Available Parameters
    --------------------

    .. list-table::
       :header-rows: 1

       * - Short Name
         - OMNIWeb Index
         - Description
       * - "BRN"
         - 3
         - Bartels Rotation Number
       * - "IMF_ID"
         - 4
         - IMF Spacecraft ID
       * - "Plasma_ID"
         - 5
         - Plasma Spacecraft ID
       * - "IMF_Fine"
         - 6
         - Fine Scale Points in IMF Averages
       * - "Plasma_Fine"
         - 7
         - Fine Scale Points in Plasma Averages
       * - "IMF_Mag"
         - 8
         - IMF Magnitude Average, nT
       * - "IMF_Vr"
         - 9
         - Magnitude, Avg IMF Vr, nT
       * - "IMF_Lat"
         - 10
         - Latitude of Avg. IMF, degrees
       * - "IMF_Lon"
         - 11
         - Longitude of Avg. IMF, degrees
       * - "Bx_GSE"
         - 12
         - Bx component, GSE/GSM, nT
       * - "By_GSE"
         - 13
         - By component, GSE, nT
       * - "Bz_GSE"
         - 14
         - Bz component, GSE, nT
       * - "By_GSM"
         - 15
         - By component, GSM, nT
       * - "Bz_GSM"
         - 16
         - Bz component, GSM, nT
       * - "Sigma_IMF_Mag"
         - 17
         - Sigma in IMF Magnitude Average
       * - "Sigma_IMF_Vec"
         - 18
         - Sigma in IMF Vector Average
       * - "Sigma_Bx"
         - 19
         - Sigma Bx, nT
       * - "Sigma_By"
         - 20
         - Sigma By, nT
       * - "Sigma_Bz"
         - 21
         - Sigma Bz, nT
       * - "T_p"
         - 22
         - Proton Temperature, K
       * - "N_p"
         - 23
         - Proton Density, n/cc
       * - "V_flow"
         - 24
         - Flow Speed, km/sec
       * - "Flow_Lon"
         - 25
         - Flow Longitude, degrees
       * - "Flow_Lat"
         - 26
         - Flow Latitude, degrees
       * - "Alpha_Proton"
         - 27
         - Alpha/Proton Density Ratio
       * - "Flow_P"
         - 28
         - Flow Pressure, nPa
       * - "Sigma_T"
         - 29
         - Sigma-T
       * - "Sigma_Np"
         - 30
         - Sigma-Np
       * - "Sigma_V"
         - 31
         - Sigma-V
       * - "Sigma_Flow_Lon"
         - 32
         - Sigma-Flow-Longitude
       * - "Sigma_Flow_Lat"
         - 33
         - Sigma-Flow-Latitude
       * - "Sigma_Alpha"
         - 34
         - Sigma-Alpha/Proton Ratio
       * - "Ey"
         - 35
         - Electric Field, mV/m
       * - "Plasma_Beta"
         - 36
         - Plasma Beta
       * - "Alfven"
         - 37
         - Alfven Mach Number
       * - "Kp"
         - 38
         - Kp*10 Index (Geomagnetic Activity)
       * - "R_Sunspot"
         - 39
         - R Sunspot Number
       * - "Dst"
         - 40
         - Dst Index, nT (Geomagnetic Storm Index)
       * - "AE"
         - 41
         - AE Index, nT (Auroral Electrojet)
       * - "PFlux_1MeV"
         - 42
         - Proton Flux > 1 MeV
       * - "PFlux_2MeV"
         - 43
         - Proton Flux > 2 MeV
       * - "PFlux_4MeV"
         - 44
         - Proton Flux > 4 MeV
       * - "PFlux_10MeV"
         - 45
         - Proton Flux > 10 MeV
       * - "PFlux_30MeV"
         - 46
         - Proton Flux > 30 MeV
       * - "PFlux_60MeV"
         - 47
         - Proton Flux > 60 MeV
       * - "Mag_Flux_Flag"
         - 48
         - Magnetospheric Flux Flag
       * - "AP"
         - 49
         - AP index, nT
       * - "F10.7"
         - 50
         - Solar Flux Index (F10.7)
       * - "PC"
         - 51
         - Polar Cap Index
       * - "AL"
         - 52
         - AL Index, nT
       * - "AU"
         - 53
         - AU Index, nT
       * - "Magnetosonic"
         - 54
         - Magnetosonic Mach Number
       * - "Lyman_Alpha"
         - 55
         - Lyman Alpha Solar Index, W/m²
       * - "Proton_QI"
         - 56
         - Proton Quasi-Invariant (QI)


    Example
    -------
    Retrieve the Kp and Dst indices for a given date range:

    .. code-block:: python

        time, kp, dst = omni('2023-01-01', '2023-01-10', {'Kp', 'Dst'})

    References
    ----------
    NASA LRO OMNIWEB: https://omniweb.gsfc.nasa.gov/form/dx1.html
    """
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")

    if not params or not isinstance(params, set):
        raise ValueError("No OMNIWEB data requested. Provide a set of parameter names or numbers.")

    resolved_params = {}
    for param in params:
        if isinstance(param, int):
            resolved_params[f'Var{param}'] = param
        elif isinstance(param, str) and param in data_mapping:
            resolved_params[param] = data_mapping[param]
        else:
            raise ValueError(f"Unknown parameter: {param}")

    start_date_url = start_date.strftime('%Y%m%d')
    end_date_url = end_date.strftime('%Y%m%d')

    vars_query = "".join([f'vars={var}&' for var in resolved_params.values()])
    url = (f'https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve&res=hour'
           f'&spacecraft=omni2&start_date={start_date_url}&end_date={end_date_url}'
           f'&maxdays=31&{vars_query}')

    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.split('\n')

    # Remove HTML headers
    data_start = next((i for i, line in enumerate(lines) if re.match(r'YEAR\s+DOY\s+HR', line)), None)
    if data_start is None:
        raise ValueError(f"Failed to parse data table from OMNIWEB response:\n{url}.")

    # Extract numerical data
    data_lines = lines[data_start + 1:]
    data_lines = [line for line in data_lines if re.match(r'\d+', line.strip())]  # Keep only numeric rows

    if not data_lines:
        raise ValueError("No valid data found in the response.")

    # Read data into numpy array
    data_array = np.loadtxt(data_lines, dtype=float)

    # Extract time and requested parameters
    years, doy, hours = data_array[:, 0], data_array[:, 1], data_array[:, 2]
    time = np.array(
        [dt.datetime(int(y), 1, 1) + dt.timedelta(days=int(d) - 1, hours=int(h)) for y, d, h in zip(years, doy, hours)])

    mask = (time >= start_date) & (time <= end_date)
    time = time[mask]
    filtered_data = [data_array[:, i][mask] for i in range(3, data_array.shape[1])]

    return (time, *filtered_data)
