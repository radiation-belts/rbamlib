import os
import urllib.request
import zipfile
import tempfile


def download_unzip(url, target_folder=None, filename_in_zip=None):
    r"""
    Download a zip file and extract files from it.

    Utility for downloading model data files.
    Used by models that require external data.

    Parameters
    ----------
    url : str
        URL to download zip from
    target_folder : str, optional
        Folder where to extract the file(s). If not provided, extracts to
        current working directory.
    filename_in_zip : str, optional
        Specific file to extract from the zip. If not provided, extracts
        all files from the archive.

    Returns
    -------
    str or list of str
        If `filename_in_zip` is provided: path to the extracted file (str)
        If `filename_in_zip` is None: list of paths to all extracted files

    Examples
    --------
    Extract a specific file to a target folder:

    >>> from rbamlib.web import download_unzip
    >>> file_path = download_unzip(
    ...     url='https://example.com/data.zip',
    ...     target_folder='./data',
    ...     filename_in_zip='model_data.mat')

    Extract all files to current directory:

    >>> file_paths = download_unzip('https://example.com/data.zip')

    Extract all files to a specific folder:

    >>> file_paths = download_unzip(
    ...     'https://example.com/data.zip',
    ...     target_folder='./output')
    """
    # Use current directory if target_folder not provided
    if target_folder is None:
        target_folder = os.getcwd()

    # Create target folder if needed
    os.makedirs(target_folder, exist_ok=True)

    # Download zip to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        tmp_path = tmp_file.name

    # Extract file(s)
    try:
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            if filename_in_zip is None:
                # Extract all files
                zip_ref.extractall(target_folder)
                # Collect all extracted paths (exclude directories)
                extracted_paths = [
                    os.path.join(target_folder, name)
                    for name in zip_ref.namelist()
                    if not name.endswith('/')
                ]
                return extracted_paths
            else:
                # Extract specific file
                # Find the file in the zip (may be in subdirectory)
                for name in zip_ref.namelist():
                    if name.endswith(filename_in_zip):
                        # Extract to target folder with just the filename
                        extracted = zip_ref.extract(name, target_folder)
                        # Move to target location if in subdirectory
                        final_path = os.path.join(target_folder, filename_in_zip)
                        if extracted != final_path:
                            os.rename(extracted, final_path)
                        return final_path
                raise FileNotFoundError(f"File {filename_in_zip} not found in zip")
    finally:
        os.unlink(tmp_path)  # Clean up temp zip
