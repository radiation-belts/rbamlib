import numpy as np
import scipy.io
from scipy.spatial import cKDTree
import warnings
import os

# Constants
#DEFAULT_FILENAME = '2022_002_Wang-et-al_life_time_all_Kp_MLT.mat'
DEFAULT_FILENAME = 'life_time_all_Kp_MLT_calculated_from_matrix.mat'
#DOWNLOAD_URL = 'https://datapub.gfz.de/download/10.5880.GFZ.2.7.2022.002fwdfs/2022_002_Wang-et-al_Data.zip'
DOWNLOAD_URL = 'https://nextcloud.gfz.de/public.php/dav/files/yQaKw8CnK7cd5B5/life_time_all_Kp_MLT_calculated_from_matrix_mat.zip'

# Module-level cache for data and KDTree (simple performance optimization)
_data_cache = {}

def W2024(L, en, kp, mlt, method='albert', data_file=None, data_folder=None, auto_download=False):
    r"""
    Calculates electron lifetime due to chorus waves following Wang et al. :cite:yearpar:`wang:2024` model.

    This model uses nearest neighbor lookup from a precomputed lifetime database as a function
    of L-shell, energy, Kp, and MLT.

    Parameters
    ----------
    L : float or ndarray
        L-shell (McIlwain L), dimensionless. Valid range: 3 ≤ L ≤ 7.
    en : float or ndarray
        Kinetic energy (MeV). Valid range: 0.001 ≤ en ≤ 2.0.
    kp : float or ndarray
        Kp index (dimensionless). Valid range: 1 ≤ Kp ≤ 6.
            *Warning*: Model is valid only for Kp ≤ 6. Use caution for Kp > 6.
    mlt : float or ndarray
        Magnetic local time in hours (0-24).
    method : str, optional
        Lifetime calculation method:
            - `albert` (default): Albert & Shprits (2009)
            - `lc`: Loss cone method (Shprits et al., 2006)
    data_file : str, optional
        Path to data file (absolute, relative, or filename).
            Default: `life_time_all_Kp_MLT_calculated_from_matrix.mat`
    data_folder : str, optional
        Folder to search for data file. Default: current working directory.
    auto_download : bool, default=False
        If True, automatically download the data file from GFZ Data Services if data file not found.

    Returns
    -------
    float or ndarray
        Electron lifetime in seconds.

    Notes
    -----
    **Data File Requirements:**

    This model requires: ``life_time_all_Kp_MLT_calculated_from_matrix.mat`` (~35 MB)

    Download from:
        - Direct (auto_download): https://nextcloud.gfz.de/public.php/dav/files/yQaKw8CnK7cd5B5/life_time_all_Kp_MLT_calculated_from_matrix_mat.zip
        - Original DOI: https://doi.org/10.5880/GFZ.2.7.2022.002

    .. note::
       The original paper references data files at DOI 10.5880/GFZ.2.7.2022.002, but the
       ``auto_download`` feature uses an updated version of the dataset hosted on GFZ Nextcloud
       with filename ``life_time_all_Kp_MLT_calculated_from_matrix.mat``.

    File search order:
        1. If ``data_file`` provided → try opening
        2. If not found → search in ``data_folder`` (default: current directory)
        3. If ``auto_download=True`` → download and extract to data_folder

    Examples
    --------
    >>> from rbamlib.models.tau.chorus import W2024
    >>> tau = W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0, auto_download=True)
    >>> print(f"Lifetime: {tau:.2e} seconds")

    >>> # Using manually downloaded file
    >>> tau = W2024(L=5.0, en=1.0, kp=3.0, mlt=12.0, data_file='path/to/file.mat')

    >>> # Array inputs
    >>> import numpy as np
    >>> L = np.array([4.0, 5.0, 6.0])
    >>> tau = W2024(L, en=1.0, kp=3.0, mlt=12.0)
    """

    # Validate method
    method = method.lower()
    if method not in ['albert', 'lc', 'loss_cone']:
        raise ValueError(f"method must be 'albert', 'lc', or 'loss_cone', got '{method}'")
    if method == 'loss_cone':
        method = 'lc'  # Normalize

    # Handle default filename and folder
    if data_file is None:
        data_file = DEFAULT_FILENAME
    if data_folder is None:
        data_folder = os.getcwd()  # Current working directory

    # Locate data file using simple fallback logic
    file_path = _locate_data_file(data_file, data_folder, auto_download)

    # Get or load data and KDTree (cached for performance)
    cache_key = (file_path, method)
    if cache_key not in _data_cache:
        _data_cache[cache_key] = _load_data(file_path, method)
    kdtree, log10_tau_data = _data_cache[cache_key]

    # Broadcast inputs
    L, en, kp, mlt = np.broadcast_arrays(
        np.asarray(L, dtype=float),
        np.asarray(en, dtype=float),
        np.asarray(kp, dtype=float),
        np.asarray(mlt, dtype=float)
    )

    # Warn for Kp > 6
    if np.any(kp > 6):
        warnings.warn(
            "Kp > 6 is outside the valid range of the chorus wave model. Use with caution.",
            UserWarning
        )

    # Transform inputs
    log10_en = np.log10(en)
    mlt_wrapped = mlt % 24.0

    # Prepare query points (kp, L, log10_en, mlt)
    query_points = np.column_stack([kp.ravel(), L.ravel(), log10_en.ravel(), mlt_wrapped.ravel()])

    # Find nearest neighbors
    distances, indices = kdtree.query(query_points)

    # Get log10(tau in days) from nearest neighbors
    log10_tau_days = log10_tau_data[indices]

    # Convert to seconds: 10^(log10_tau) * 86400
    tau_seconds = np.power(10.0, log10_tau_days) * 86400.0

    # Reshape to match input shape
    tau_seconds = tau_seconds.reshape(L.shape)

    # Return scalar if input was scalar (0-d array)
    if tau_seconds.ndim == 0:
        return tau_seconds.item()

    return tau_seconds


def _locate_data_file(filename, data_folder, auto_download):
    """File location logic with optional download."""

    # Try in data_folder
    path_in_folder = os.path.join(data_folder, filename)
    if os.path.isfile(path_in_folder):
        return path_in_folder
    
    # Try as-is (absolute or relative path)
    if os.path.isfile(filename):
        return filename

    # Auto-download if enabled
    if auto_download:
        from rbamlib.web import download_unzip
        print(f"Downloading {filename} from {DOWNLOAD_URL.split('/')[2]}...")
        return download_unzip(DOWNLOAD_URL, data_folder, filename)

    # Not found
    raise FileNotFoundError(
        f"Data file '{filename}' not found in {data_folder}. "
        f"Download from {DOWNLOAD_URL} or set auto_download=True"
    )


def _load_data(file_path, method):
    """Load MATLAB data and create KDTree for nearest neighbor search."""

    # Load MATLAB file
    mat_data = scipy.io.loadmat(file_path)

    # Get the data array from the known variable name
    if 'life_time_all_MLT' not in mat_data:
        raise ValueError(f"Variable 'life_time_all_MLT' not found in {file_path}")

    data_array = mat_data['life_time_all_MLT']

    if data_array.ndim != 2 or data_array.shape[1] != 8:
        raise ValueError(f"Expected 8-column data array, got shape {data_array.shape}")

    # Extract columns
    # col 0: Kp, col 1: MLT, col 4: L, col 5: log10(E), col 6: log10(tau_albert), col 7: log10(tau_lc)
    kp_data = data_array[:, 0]
    mlt_data = data_array[:, 1]
    L_data = data_array[:, 4]
    log10_E_data = data_array[:, 5]

    if method == 'albert':
        log10_tau_data = data_array[:, 6]
    else:  # 'lc'
        log10_tau_data = data_array[:, 7]

    # Build KDTree for nearest neighbor search (4D: kp, L, log10_E, mlt)
    points = np.column_stack([kp_data, L_data, log10_E_data, mlt_data])
    kdtree = cKDTree(points)

    return kdtree, log10_tau_data
