"""
Read VELOX data
"""

import xarray as xr
import os

from lizard.ac3airlib import day_of_flight


def read_velox(flight_id):
    """
    Read VELOX data stripe
    """

    date = day_of_flight(flight_id)

    mission, platform, name = flight_id.split("_")

    ds = xr.open_mfdataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/halo-ac3/velox",
            f"HALO-AC3_HALO_VELOX_BT3_10740nm_{date.strftime('%Y%m%d')}_{name}_v3.0.nc",
        ),
        preprocess=preprocess,
    ).load()

    ds["BT_2D"] += 273.15

    return ds


def preprocess(ds):
    """Select the stripe that goes through nadir at y=253"""
    ds = ds.isel(y=253)
    return ds
