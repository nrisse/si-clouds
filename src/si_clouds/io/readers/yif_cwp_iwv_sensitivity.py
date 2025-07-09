"""
Read result of CWP and young ice fraction sensitivity experiment.
"""

import os

import xarray as xr
from dotenv import load_dotenv

load_dotenv()


def read_yif_cwp_sensitivity():

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_clouds/model_uncertainty",
            "yif_cwp_sensitivity.nc",
        )
    )

    return ds


def read_iwv_sensitivity():

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_clouds/model_uncertainty",
            "iwv_jacobian.nc",
        )
    )
    
    return ds
