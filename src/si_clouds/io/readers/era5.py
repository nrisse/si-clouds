"""
Reads ERA5 data for si-clouds project.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from lizard.ac3airlib import day_of_flight

load_dotenv()


def read_era5_model_levels_inst(flight_id):
    """
    Reads ERA5 on model levels data. The specific data imported here is
    tailored to the si_clouds project in the HALO-AC3 region.

    This is instantaneous data, i.e., at the time of the flight.

    The data also includes z and lnsp.
    """

    date = day_of_flight(flight_id)

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/era5/si_clouds_flights",
            pd.Timestamp(date).strftime("%Y/%m"),
            f'era5-model-levels_siclouds_{pd.Timestamp(date).strftime("%Y%m%d")}.nc',
        )
    )

    ds["z"] = ds["z"].sel(level=1).reset_coords(drop=True)
    ds["lnsp"] = ds["lnsp"].sel(level=1).reset_coords(drop=True)

    return ds


def read_era5_model_levels_clim():
    """
    Reads ERA5 on model levels data. The specific data imported here is
    tailored to the si_clouds project in the HALO-AC3 region.

    This is climatological data, i.e., over several years in the same region.

    All the climatological files will be returned as dask array to allow
    further computations

    The data also includes z and lnsp.
    """

    ds = xr.open_mfdataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/era5/si_clouds/*/*/",
            f"era5-model-levels_siclouds_*.nc",
        )
    )

    ds["z"] = ds["z"].sel(level=1).reset_coords(drop=True)
    ds["lnsp"] = ds["lnsp"].sel(level=1).reset_coords(drop=True)

    return ds


def read_era5_coef():
    """
    Reads ERA5 model level coefficients. The half level 137 corresponds to
    the surface, i.e., below full level 137. The half level 0 corresponds to
    the top of the atmosphere, i.e., above full level 1.

    Returns
    -------
    ds_coef : xr.Dataset
        ERA5 model level coefficients at half levels.
    """

    file = os.path.join(os.environ["PATH_SEC"], "data/era5/l137def.csv")

    with open(file, "r") as f:
        df_coef = pd.read_csv(f, comment="#", index_col=0)
    ds_coef = df_coef.to_xarray()
    ds_coef = ds_coef.rename({"n": "half_level"})

    # create an indices that give upper/lower half level of a given full level
    # upper means here higher in the atmosphere
    ix_upper_level = xr.DataArray(
        np.arange(0, 137), dims="level", coords={"level": np.arange(1, 138)}
    )
    ix_lower_level = ix_upper_level + 1

    ds_coef["ix_lower_level"] = ix_lower_level
    ds_coef["ix_upper_level"] = ix_upper_level

    return ds_coef


def read_era5_single_levels_inst(flight_id):
    """
    Reads ERA5 on single levels data. The specific data imported here is
    tailored to the si_clouds project in the HALO-AC3 region.

    This is instantaneous data, i.e., at the time of the flight.

    Reads instant data and accumulated data (snow, precip)
    """

    date = day_of_flight(flight_id)

    ds = (
        xr.open_mfdataset(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/era5/si_clouds_flights",
                pd.Timestamp(date).strftime("%Y/%m"),
                f'era5-single-levels_siclouds_{pd.Timestamp(date).strftime("%Y%m%d")}_*.nc',
            )
        )
        .load()
        .rename({"valid_time": "time"})
    )

    return ds
