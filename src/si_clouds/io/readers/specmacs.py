"""
Read SpecMACS 1024 nm radiance data.
"""

import os

import xarray as xr
from lizard.ac3airlib import day_of_flight


def read_specmacs(flight_id):
    """
    Read SpecMACS 1024.4 nm radiance data for a given flight.
    """

    date = day_of_flight(flight_id)

    mission, platform, name = flight_id.split("_")

    ds = xr.open_dataset(
        os.path.join(
            "/data/obs/campaigns/halo-ac3/halo/specmacs/swir/",
            f"HALO-AC3_HALO_specMACS_SWIR_{date.strftime('%Y%m%d')}_{name}_1024.40nm_1144.73nm.nc",
        ),
    ).isel(wavelength=0)

    return ds
