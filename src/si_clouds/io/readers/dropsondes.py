"""
Reader for HALO-(AC)3 Level 3 dropsonde data.
"""

import os
from glob import glob

import pandas as pd
import xarray as xr


def read_dropsondes_l3():
    """
    Reads dropsonde data from the HALO-(AC)3 Level 3 data set.
    """

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_DAT"],
            "obs/campaigns/halo-ac3/halo/dropsondes/l3/merged_HALO_P5_beta_v2.nc",
        )
    )

    return ds


def read_dropsondes_l2(flight_id, time):
    """
    Reads dropsonde data from the HALO-(AC)3 Level 3 data set.
    """

    mission, platform, name = flight_id.split("_")
    time = pd.Timestamp(time)
    files = glob(
        os.path.join(
            os.environ["PATH_DAT"],
            "obs/campaigns/halo-ac3/halo/dropsondes/l2/",
            f"HALO-(AC)3_{platform}_{time.strftime('%Y-%m-%dT%H%M%S')}_000000_{time.strftime('%Y%m%d')}_{name}_*_Level_2.nc",
        )
    )
    # two sondes have the same launch time, this is the one with a wrong launch time
    if "20220314_RF04_210330184_Level_2.nc" in files[0]:
        files = files[1:]
    assert len(files) == 1
    file = files[0]
    ds = xr.open_dataset(file)

    return ds
