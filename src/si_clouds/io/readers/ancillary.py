"""
Read ancillary data.
"""

import os

import xarray as xr
from dotenv import load_dotenv

load_dotenv()


def read_ancillary_data():
    """
    Read ancillary data.
    """

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_clouds/detect/ancillary_data",
            "ancillary.nc",
        )
    )

    return ds
