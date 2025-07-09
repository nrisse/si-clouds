"""
Read VELOX classification data.
"""

import os

import xarray as xr
from dotenv import load_dotenv

load_dotenv()


def read_velox_classification():

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_clouds/velox",
            "HALO-AC3_VELOX_surface_type_HAMP_clear_sky.nc",
        )
    )

    return ds
