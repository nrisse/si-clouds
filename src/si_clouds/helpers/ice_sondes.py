"""
Finds dropsondes launched over sea ice.
"""

import numpy as np
import pandas as pd
from lizard import ac3airlib
from lizard.readers.amsr2_sic_track import read_amsr2_sic_track_all

from si_clouds.io.readers.dropsondes import read_dropsondes_l3


def sea_ice_sondes(sic_threshold=90):
    """
    Get dropsondes that were launched over sea ice.
    """

    # get halo dropsondes
    ds = read_dropsondes_l3()
    ds = ds.sel(sonde_id=ds.platform_id == "HALO")

    # get sea ice concentration below aircraft during sonde launches
    ds_sic = read_amsr2_sic_track_all()
    ds_sic = ds_sic.sel(time=ds.launch_time, method="nearest")
    ds_sic = ds_sic.load()
    assert ((ds_sic.time - ds.launch_time) / np.timedelta64(1, "s") == 0).all()
    ds["sic"] = ds_sic.sic

    # filter out sonde launches over sea ice, keep northpole
    ds = ds.sel(sonde_id=(ds.sic >= sic_threshold) | (ds.flight_lat > 88))

    # filter out sondes with incomplete profiles (at least 75% available)
    ix_rh = ds.profile_fullness_fraction_rh > 0.75
    ix_p = ds.profile_fullness_fraction_pres > 0.75
    ix_t = ds.profile_fullness_fraction_tdry > 0.75
    ix = ix_rh & ix_p & ix_t
    ds = ds.sel(sonde_id=ix)

    # this sonde has a wrong launch time
    ds = ds.sel(sonde_id=ds.sonde_id != "210330184")

    # sort by time and add flight_id
    ds = ds.sortby(ds.launch_time)
    flight_ids = []
    for time in ds.launch_time.values:
        flight_ids.extend(
            list(
                ac3airlib.flights_of_day(
                    pd.Timestamp(time).date(),
                    missions=["HALO-AC3"],
                    platforms=["HALO"],
                )
            )
        )
    ds["flight_id"] = ("sonde_id", flight_ids)

    return ds
