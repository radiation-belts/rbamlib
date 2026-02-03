import os
import urllib.request
import zipfile
import tempfile


def download_unzip(url, target_folder, filename_in_zip):
    """
    Download a zip file and extract a specific file.

    Utility for downloading model data files.
    Used by models that require external data (exception to the rule).

    Parameters
    ----------
    url : str
        URL to download zip from
    target_folder : str
        Folder where to extract the file
    filename_in_zip : str
        Specific file to extract from the zip

    Returns
    -------
    str
        Path to extracted file
    """
    # Create target folder if needed
    os.makedirs(target_folder, exist_ok=True)

    # Download zip to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        tmp_path = tmp_file.name

    # Extract specific file
    try:
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
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
