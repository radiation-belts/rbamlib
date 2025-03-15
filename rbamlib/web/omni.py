import requests
import numpy as np
import datetime as dt
import re
from rbamlib.utils import parse_datetime

data_mapping_low = {
    "BRN": 3,               # Bartels Rotation Number
    "IMF_ID": 4,            # IMF Spacecraft ID
    "Plasma_ID": 5,         # Plasma Spacecraft ID
    "IMF_Fine": 6,          # Fine Scale Points in IMF Averages
    "Plasma_Fine": 7,       # Fine Scale Points in Plasma Averages
    "IMF_Mag": 8,           # IMF Magnitude Average, nT
    "IMF": 8,               # IMF Magnitude Average, nT
    "IMF_Vr": 9,            # Magnitude, Avg IMF Vr, nT
    "IMF_Lat": 10,          # Latitude of Avg. IMF, degrees
    "IMF_Lon": 11,          # Longitude of Avg. IMF, degrees
    "Bx_GSE": 12,           # Bx component, GSE/GSM, nT
    "Bx_GSM": 12,           # Bx component, GSE/GSM, nT
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
    "E": 35,                # Electric Field, mV/m
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

data_mapping_high = {
    "IMF_ID": 4,            # IMF Spacecraft ID
    "Plasma_ID": 5,         # Plasma Spacecraft ID
    "IMF_Fine": 6,          # Fine Scale Points in IMF Averages
    "Plasma_Fine": 7,       # Fine Scale Points in Plasma Averages
    "Percent_Interpolated": 8,  # Percent interpolated
    "Timeshift": 9,         # Timeshift, sec
    "Sigma_Timeshift": 10,  # Sigma Timeshift
    "Sigma_Min_Var_Vector": 11,  # Sigma Min_var_vector
    "Time_Between_Obs": 12, # Time between observations, sec
    "IMF_Mag": 13,          # IMF Magnitude Avg (Scalar), nT
    "IMF": 13,              # IMF Magnitude Avg (Scalar), nT
    "Bx_GSE": 14,           # Bx, GSE/GSM, nT
    "Bx_GSM": 14,           # Bx, GSE/GSM, nT
    "By_GSE": 15,           # By, GSE, nT
    "Bz_GSE": 16,           # Bz, GSE, nT
    "By_GSM": 17,           # By, GSM, nT
    "Bz_GSM": 18,           # Bz, GSM, nT
    "Sigma_IMF_Mag": 19,    # Sigma in IMF Magnitude Avg.
    "Sigma_IMF_Vec": 20,    # Sigma in IMF Vector Avg
    "Flow_Speed": 21,       # Flow Speed, km/sec
    "Vsw": 21,              # Flow Speed, km/sec
    "Vx_GSE": 22,           # Vx Velocity, GSE, km/s
    "Vy_GSE": 23,           # Vy Velocity, GSE, km/s
    "Vz_GSE": 24,           # Vz Velocity, GSE, km/s
    "Nsw": 25,              # Proton Density, n/cc
    "N_p": 25,              # Proton Density, n/cc
    "Tsw": 26,              # Proton Temperature, K
    "T_p": 26,              # Proton Temperature, K
    "Flow_P": 27,           # Flow Pressure, nPa
    "Pdyn": 27,             # Flow Pressure, nPa
    "Ey": 28,               # Electric Field, mV/m
    "E": 28,                # Electric Field, mV/m
    "Plasma_Beta": 29,      # Plasma Beta
    "Alfven": 30,           # Alfven Mach Number
    "Magnetosonic": 45,     # Magnetosonic Mach Number
    "Spacecraft_X": 31,     # Spacecraft X, GSE, Re
    "Spacecraft_Y": 32,     # Spacecraft Y, GSE, Re
    "Spacecraft_Z": 33,     # Spacecraft Z, GSE, Re
    "BSN_X": 34,           # Bow Shock Nose (BSN) location X, GSE, Re
    "BSN_Y": 35,           # BSN location Y, GSE, Re
    "BSN_Z": 36,           # BSN location Z, GSE, Re
    "AE": 37,               # AE Index, nT
    "AL": 38,               # AL Index, nT
    "AU": 39,               # AU Index, nT
    "SYM_D": 40,            # SYM/D, nT
    "SYM_H": 41,            # SYM/H, nT
    "SYMD": 40,             # SYM/D, nT
    "SYMH": 41,             # SYM/H, nT
    "SYM/D": 40,            # SYM/D, nT
    "SYM/H": 41,            # SYM/H, nT
    "ASY_D": 42,            # ASY/D, nT
    "ASY_H": 43,            # ASY/H, nT
    "PC": 44,               # Polar Cap (PC) index from Thule
    "PFlux_10MeV": 46,      # Proton Flux > 10 MeV, 1/(cm**2-sec-ster)
    "PFlux_30MeV": 47,      # Proton Flux > 30 MeV, 1/(cm**2-sec-ster)
    "PFlux_60MeV": 48       # Proton Flux > 60 MeV, 1/(cm**2-sec-ster)
}

resolution_low = {
    "source": "LRO",
    "data_mapping": data_mapping_low,
    "date_format": "%Y%m%d",
    "pattern": r'YEAR\s+DOY\s+HR',
    "data_start_idx": 3
}

resolution_high = {
    "source": "HRO",
    "data_mapping": data_mapping_high,
    "date_format": "%Y%m%d",
    "pattern": r'YYYY\s+DOY\s+HR',
    "data_start_idx": 4
}

resolution_settings = {
    "hour": {"spacecraft": "omni2", **resolution_low},
    "daily": {"spacecraft": "omni2_daily", **resolution_low},
    "27day": {"spacecraft": "omni2_27day", **resolution_low},
    "yearly": {"spacecraft": "omni2_yearly", **resolution_low},
    "min": {"spacecraft": "omni_min", **resolution_high},
    "5min": {"spacecraft": "omni_5min", **resolution_high},
}

def omniweb_request(start_date, end_date, params, resolution):
    r"""
    Fetch space weather data from OMNIWEB.

    This function performs the API request to OMNIWEB, processes the response,
    and returns time-series data for the specified parameters and resolution.

    Parameters
    ----------
    start_date : str or datetime
        Start date in multiple formats ('YYYY-MM-DD', 'YYYYMMDDHH').
    end_date : str or datetime
        End date matching the format of start_date.
    params : set
        Set of parameter names (e.g., {'Kp', 'Dst'}) or corresponding OMNIWEB indices.
    resolution : str
        Specifies the data resolution. Must be one of:

        - 'hour', 'daily', '27day', 'yearly' (Low Resolution OMNI - LRO)
        - 'min', '5min' (High Resolution OMNI - HRO)

    Returns
    -------
    tuple
        A tuple containing (time, *data_values), where: `time` is a NumPy array of datetime objects.
        `data_values` are NumPy arrays corresponding to the requested parameters.
    """
    if resolution not in resolution_settings:
        raise ValueError(f"Invalid resolution: {resolution}. Must be one of {list(resolution_settings.keys())}.")

    settings = resolution_settings[resolution]
    spacecraft = settings["spacecraft"]
    data_mapping = settings["data_mapping"]
    date_format = settings["date_format"]
    pattern = settings["pattern"]
    data_start_idx = settings["data_start_idx"]


    start_date = parse_datetime(start_date).strftime(date_format)
    end_date = parse_datetime(end_date).strftime(date_format)

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

    vars_query = "".join([f'vars={var}&' for var in resolved_params.values()])
    url = (f'https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve&res={resolution}'
           f'&spacecraft={spacecraft}&start_date={start_date}&end_date={end_date}'
           f'&{vars_query}')

    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.split('\n')

    data_start = next((i for i, line in enumerate(lines) if re.match(pattern, line)), None)
    if data_start is None:
        if "Error" in response.text:
            error_message = re.search(r'<H2><TT>(.*?)</TT></H2>', response.text)
            if error_message:
                raise ValueError(f"OMNIWEB Error: {error_message.group(1)}")
            else:
                raise ValueError("OMNIWEB Error: Unknown error occurred.")
        raise ValueError("Failed to parse data table from OMNIWEB response.")

    data_lines = lines[data_start + 1:]
    data_lines = [line for line in data_lines if re.match(r'\d+', line.strip())]

    if not data_lines:
        raise ValueError("No valid data found in the response.")

    data_array = np.loadtxt(data_lines, dtype=float)

    years, doy, hours = data_array[:, 0], data_array[:, 1], data_array[:, 2]
    # Is HRO dataset, parse minutes
    minutes = data_array[:, 3] if data_start_idx == 4 else np.zeros_like(years)
    time = np.array(
        [dt.datetime(int(y), 1, 1) + dt.timedelta(days=int(d) - 1, hours=int(h), minutes=int(m)) for y, d, h, m in zip(years, doy, hours, minutes)])

    mask = (time >= parse_datetime(start_date)) & (time <= parse_datetime(end_date))
    time = time[mask]
    filtered_data = [data_array[:, i][mask] for i in range(data_start_idx, data_array.shape[1])]

    return (time, *filtered_data)


def omni(start_date, end_date, params=None, resolution='hour'):
    r"""
    Retrieve space weather data from the OMNIWEB database at various resolutions.

    Parameters
    ----------
    start_date : str or datetime
        Start date in multiple supported formats, such as 'YYYY-MM-DD', or 'YYYYMMDDHH'.
    end_date : str or datetime
        End date in multiple supported formats, matching the start_date format.
    params : set
        Set of parameter names (e.g., {'Kp', 'Dst'}) or numbers (e.g., {38, 'Dst'}).

        - If a string is provided, it is mapped to the corresponding OMNIWEB variable number.
        - If a number is provided, it is directly used in the request.
        - Raises an error if an unknown parameter is included.
    resolution : str, default= `hour`
        The resolution of the data, corresponding to either the Low Resolution OMNI (LRO) or
        High Resolution OMNI (HRO) datasets:

        **Low Resolution OMNI (LRO) Data Set:**
            - 'hour'   (Hourly resolution, source 'omni2')
            - 'daily'  (Daily resolution, source 'omni2_daily')
            - '27day'  (27-day resolution, source 'omni2_27day')
            - 'yearly' (Yearly resolution, source 'omni2_yearly')

        **High Resolution OMNI (HRO) Data Set:**
            - 'min' (1-minute resolution, source 'omni_min')
            - '5min' (5-minute resolution, source 'omni_5min')

    Returns
    -------
    tuple
        If multiple parameters are requested, returns (time, *data_values), where:

        - `time` is a NumPy array of datetime objects.
        - `data_values` are NumPy arrays ordered according to the requested parameter keys.

    Notes
    -----
        - The data is retrieved from the NASA OMNIWEB database and includes various space weather
          indices, such as Kp, Dst, AE, and solar indices.
        - The function automatically converts the retrieved data into numpy arrays and filters them
          by the specified date range.
        - The OMNIWEB service may return missing or fill values for certain periods.
        - Parameter names are case-sensitive.

    Available Parameters
    --------------------
    **Low Resolution OMNI (LRO) Data Set:**

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


    **High Resolution OMNI (HRO) Data Set:**

    .. list-table::
       :header-rows: 1

       * - Short Name
         - OMNIWeb Index
         - Description
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
       * - "Percent_Interpolated"
         - 8
         - Percent interpolated
       * - "Timeshift"
         - 9
         - Timeshift, sec
       * - "Sigma_Timeshift"
         - 10
         - Sigma Timeshift
       * - "Sigma_Min_Var_Vector"
         - 11
         - Sigma Min_var_vector
       * - "Time_Between_Obs"
         - 12
         - Time between observations, sec
       * - "IMF_Mag"
         - 13
         - IMF Magnitude Avg (Scalar), nT
       * - "Bx_GSE"
         - 14
         - Bx component, GSE/GSM, nT
       * - "By_GSE"
         - 15
         - By component, GSE, nT
       * - "Bz_GSE"
         - 16
         - Bz component, GSE, nT
       * - "By_GSM"
         - 17
         - By component, GSM, nT
       * - "Bz_GSM"
         - 18
         - Bz component, GSM, nT
       * - "Sigma_IMF_Mag"
         - 19
         - Sigma in IMF Magnitude Avg.
       * - "Sigma_IMF_Vec"
         - 20
         - Sigma in IMF Vector Avg
       * - "Flow_Speed"
         - 21
         - Flow Speed, km/sec
       * - "Vx_GSE"
         - 22
         - Vx Velocity, GSE, km/s
       * - "Vy_GSE"
         - 23
         - Vy Velocity, GSE, km/s
       * - "Vz_GSE"
         - 24
         - Vz Velocity, GSE, km/s
       * - "N_p"
         - 25
         - Proton Density, n/cc
       * - "T_p"
         - 26
         - Proton Temperature, K
       * - "Flow_P"
         - 27
         - Flow Pressure, nPa
       * - "Ey"
         - 28
         - Electric Field, mV/m
       * - "Plasma_Beta"
         - 29
         - Plasma Beta
       * - "Alfven"
         - 30
         - Alfven Mach Number
       * - "Magnetosonic"
         - 45
         - Magnetosonic Mach Number
       * - "Spacecraft_X"
         - 31
         - Spacecraft X, GSE, Re
       * - "Spacecraft_Y"
         - 32
         - Spacecraft Y, GSE, Re
       * - "Spacecraft_Z"
         - 33
         - Spacecraft Z, GSE, Re
       * - "BSN_X"
         - 34
         - Bow Shock Nose (BSN) location X, GSE, Re
       * - "BSN_Y"
         - 35
         - BSN location Y, GSE, Re
       * - "BSN_Z"
         - 36
         - BSN location Z, GSE, Re
       * - "AE"
         - 37
         - AE Index, nT
       * - "AL"
         - 38
         - AL Index, nT
       * - "AU"
         - 39
         - AU Index, nT
       * - "SYM_D"
         - 40
         - SYM/D, nT
       * - "SYM_H"
         - 41
         - SYM/H, nT
       * - "ASY_D"
         - 42
         - ASY/D, nT
       * - "ASY_H"
         - 43
         - ASY/H, nT
       * - "PC"
         - 44
         - Polar Cap (PC) index from Thule
       * - "PFlux_10MeV"
         - 46
         - Proton Flux > 10 MeV, 1/(cm²-sec-ster)
       * - "PFlux_30MeV"
         - 47
         - Proton Flux > 30 MeV, 1/(cm²-sec-ster)
       * - "PFlux_60MeV"
         - 48
         - Proton Flux > 60 MeV, 1/(cm²-sec-ster)


    Example
    -------
    Retrieve the Kp and Dst indices for a given date range:

    .. code-block:: python

        time, kp, dst = omni('2023-01-01', '2023-01-10', {'Kp', 'Dst'})

        time, AL = omni('2023-01-01', '2023-01-10', {'AL'}, resolution='min')

    References
    ----------
    NASA LRO OMNIWEB: https://omniweb.gsfc.nasa.gov/form/dx1.html

    NASA HRO OMNIWEB: https://omniweb.gsfc.nasa.gov/form/omni_min.html
    """

    return omniweb_request(start_date, end_date, params, resolution)
