"""Make pyproj use a PROJ database compatible with the active interpreter."""

from pathlib import Path
import sys
import warnings


def configure_proj():
    """Return a working PROJ data directory, repairing mixed conda installs.

    Some Windows conda environments contain both pyproj's bundled database and
    ``Library/share/proj/proj.db``.  The bundled database can be incompatible
    with the PROJ DLL loaded by conda.  Validate an EPSG lookup and, if needed,
    prefer the database installed alongside that DLL.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pyproj unable to set PROJ database path.*",
            category=UserWarning,
        )
        import pyproj

    try:
        pyproj.CRS.from_epsg(4326)
        return Path(pyproj.datadir.get_data_dir())
    except pyproj.exceptions.CRSError:
        pass

    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "share" / "proj",
    ]
    for candidate in candidates:
        if not (candidate / "proj.db").is_file():
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pyproj unable to set PROJ database path.*",
                category=UserWarning,
            )
            pyproj.datadir.set_data_dir(candidate)
        try:
            pyproj.CRS.from_epsg(4326)
            return candidate
        except pyproj.exceptions.CRSError:
            continue

    raise RuntimeError(
        "No PROJ database compatible with the active pyproj/PROJ binaries. "
        "Reinstall pyproj and proj from the same conda-forge environment."
    )
