"""
Test sensitivity to integrated water vapor.
"""

import os

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from lizard.readers.band_pass import read_band_pass

from si_clouds.io.readers.retrieval_times import read_retrieval_times
from si_clouds.retrieval.oem import OEMRetrieval


def main(n):
    """
    Runs forward model sensitivity test with varying integrated water vapor.
    """

    ds_bp = read_band_pass("HAMP")
    setting_file = "setting_retrieval_r2.yaml"

    df = read_retrieval_times("all_sky")
    obs_times = pd.to_datetime(df.iloc[:, 0].values)
    flight_ids = df.iloc[:, 1].values

    j_lst = []
    iwv0_lst = []
    iwv1_lst = []
    y0_lst = []
    y1_lst = []

    # adjust number of channels in forward simulations
    channels = np.arange(1, 26)
    ds_bp = ds_bp.sel(channel=channels)

    for i in tqdm.tqdm(range(n)):
        obs_time = obs_times[i]
        flight_id = flight_ids[i]

        r = OEMRetrieval(setting_file=setting_file, flight_id=flight_id)

        r.setting.general.n_processes_pamtra = 26

        # update the forward model arguments to include more channels
        r.forwardKwArgs["frequency"] = np.unique(ds_bp["avg_freq"])
        r.forwardKwArgs["polarization"] = ds_bp["polarization"]
        r.forwardKwArgs["da_avg_freq"] = ds_bp["avg_freq"]

        r.read_position_of_observation(obs_time=obs_time)
        r.create_a_priori_x(obs_time=obs_time)

        x_vec = pd.Series(data=r.x_a, index=r.x_vars)
        b_vec = pd.Series(data=r.b, index=r.b_vars)
        xb_vec = pd.concat([x_vec, b_vec])

        y0 = r.forward_model(
            x=xb_vec,
            **r.forwardKwArgs,
        )

        # scale the specific humidity profile to get an increase in iwv by 1 kg/m^2
        iwv0 = r.calculate_iwv(
            pressure=r.forwardKwArgs["pressure"],
            temperature=r.forwardKwArgs["t_ut"],
            specific_humidity=(10 ** r.forwardKwArgs["qv_ut"] * 1e-3),
        )
        iwv_high = iwv0 + 1
        iwv_ratio = iwv_high / iwv0
        r.forwardKwArgs["qv_ut"] = np.log10(
            (10 ** r.forwardKwArgs["qv_ut"] * 1e-3) * iwv_ratio * 1e3
        )
        iwv1 = r.calculate_iwv(
            pressure=r.forwardKwArgs["pressure"],
            temperature=r.forwardKwArgs["t_ut"],
            specific_humidity=(10 ** r.forwardKwArgs["qv_ut"] * 1e-3),
        )
        assert np.isclose(iwv1, iwv_high, rtol=1e-3), "iwv not increased by 1 kg/m^2"

        y1 = r.forward_model(
            x=xb_vec,
            **r.forwardKwArgs,
        )

        # compute jacobian
        j = (y1 - y0) / (iwv1 - iwv0)

        j_lst.append(j)
        iwv0_lst.append(iwv0)
        iwv1_lst.append(iwv1)
        y0_lst.append(y0)
        y1_lst.append(y1)

    ds = xr.Dataset()
    ds.coords["time"] = obs_times[:n]
    ds.coords["channel"] = channels
    ds["flight_id"] = ("time", flight_ids[:n])
    ds["j"] = (("time", "channel"), np.array(j_lst))
    ds["iwv0"] = ("time", np.array(iwv0_lst))
    ds["iwv1"] = ("time", np.array(iwv1_lst))
    ds["y0"] = (("time", "channel"), np.array(y0_lst))
    ds["y1"] = (("time", "channel"), np.array(y1_lst))

    file = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_clouds/model_uncertainty",
        "iwv_jacobian.nc",
    )
    ds.to_netcdf(file)
    print(f"Written result to {file}")


if __name__ == "__main__":
    main(n=200)
