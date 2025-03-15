import datetime as dt

def parse_datetime(date_input):
    """
    Parses a datetime input from various formats into a Python datetime object.

    Parameters
    ----------
    date_input : str or datetime
        Input date in various formats, such as:

        - '2025010112'         → YYYYMMDDHH
        - '2025-01-01'         → YYYY-MM-DD
        - '20250101'           → YYYYMMDD
        - '20250101T12:00'     → ISO-like format
        - '2025-01-01T12:00'   → ISO-like format
        - '2025-01-01 12:30'   → Standard format
        - '01-01-2025'         → European format
        - 'Jan 01, 2025'       → Human-readable

    Returns
    -------
    datetime.datetime
        Parsed datetime object.

    Raises
    ------
    ValueError
        If the input format is invalid.
    """
    if isinstance(date_input, dt.datetime):
        return date_input  # Already a datetime object

    formats = [
        "%Y%m%d%H", "%Y-%m-%d", "%Y%m%d", "%Y%m%dT%H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%Y %H:%M", "%b %d, %Y", "%B %d, %Y",
        "%Y-%m-%dT%H:%M:%S"
    ]

    for fmt in formats:
        try:
            return dt.datetime.strptime(date_input, fmt)
        except ValueError:
            continue

    raise ValueError(f"Invalid date format: {date_input}")
