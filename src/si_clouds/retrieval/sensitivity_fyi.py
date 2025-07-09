"""
Test sensitivity to young ice fraction and cloud water path (CWP).

- Young ice fraction is increased by 50%
- CWP is increased by 150 g/m^2
"""

import os

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from si_clouds.io.readers.retrieval_times import read_retrieval_times
from si_clouds.retrieval.oem import OEMRetrieval


def main():
    """
    Runs forward model sensitivity test with varying young ice fraction and
    CWP.
    """

    setting_file = "setting_retrieval_r2.yaml"

    df = read_retrieval_times("clear_sky_kt19")
    obs_times = pd.to_datetime(df.iloc[:, 0].values)
    flight_ids = df.iloc[:, 1].values

    n_channels = 6
    y0_arr = np.full(
        (len(obs_times), n_channels),
        fill_value=np.nan,
    )
    y1_arr = np.full(
        (len(obs_times), n_channels),
        fill_value=np.nan,
    )
    y2_arr = np.full(
        (len(obs_times), n_channels),
        fill_value=np.nan,
    )

    for i in tqdm.tqdm(range(len(obs_times))):
        obs_time = obs_times[i]
        flight_id = flight_ids[i]

        try:
            r = OEMRetrieval(setting_file=setting_file, flight_id=flight_id)

            r.read_position_of_observation(obs_time=obs_time)
            r.create_a_priori_x(obs_time=obs_time)

            x_vec = pd.Series(data=r.x_a, index=r.x_vars)
            b_vec = pd.Series(data=r.b, index=r.b_vars)
            xb_vec = pd.concat([x_vec, b_vec])

            y0 = r.forward_model(
                x=xb_vec,
                **r.forwardKwArgs,
            )

            # change cloud water path
            xb_vec["cwp"] = 0.15
            y1 = r.forward_model(
                x=xb_vec,
                **r.forwardKwArgs,
            )
            xb_vec["cwp"] = 0  # reset CWP

            # change young ice fraction
            r.setting.surface.yi_fraction = 0.5
            r.create_a_priori_x(obs_time=obs_time)
            y2 = r.forward_model(
                x=xb_vec,
                **r.forwardKwArgs,
            )

            y0_arr[i, :] = y0
            y1_arr[i, :] = y1
            y2_arr[i, :] = y2

        except Exception as e:
            print(
                f"Error processing observation at {obs_time} for flight {flight_id}: {e}"
            )

    ds = xr.Dataset()
    ds.coords["time"] = obs_times
    ds.coords["channel"] = r.setting.observation.channels
    ds["flight_id"] = ("time", flight_ids)
    ds["y0"] = (("time", "channel"), y0_arr)
    ds["y1"] = (("time", "channel"), y1_arr)
    ds["y2"] = (("time", "channel"), y2_arr)

    file = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_clouds/model_uncertainty",
        "yif_cwp_sensitivity.nc",
    )
    ds.to_netcdf(file)
    print(f"Written result to {file}")


if __name__ == "__main__":
    main()
