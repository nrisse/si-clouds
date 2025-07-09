"""
Run synthetic retrievals.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from si_clouds.retrieval.oem import OEMRetrieval, config_logger, write_to_netcdf


def main(
    setting_file,
    flight_id,
    obs_time,
    version,
    test_id,
    syn_type,
    cwp_distribution,
    random_seed,
):
    """
    Synthetic retrieval.

    Note on where simulation, observation and state are stored:
    The actual HAMP observation is saved in ds_syn and the forward simulation
    in ds_a. The synthetic state is in ds_syn and the a priori state in ds_a.
    This means that the true state of the synthetic retrieval is not in the
    same file as its forward simulation! This is to not keep the forward
    simulation twice and have a place to keep the real observation.
    """

    np.random.seed(random_seed)

    obs_time = pd.Timestamp(obs_time)

    config_logger(
        log_file=os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_clouds/retrieved/log",
            f"log_syn_{flight_id}_{obs_time.strftime('%Y%m%d_%H%M%S')}_{test_id}_{version}.log",
        )
    )
    r = OEMRetrieval(setting_file=setting_file, flight_id=flight_id)
    r.initialize(obs_time=obs_time)
    if syn_type == "random":
        xb_vec_syn = modify_state_random(r, cwp_distribution)
    else:
        raise ValueError("Unknown synthetic retrieval type")

    # store the actual hamp observation
    y_obs_hamp = pd.Series(r.y_obs.copy(), r.y_vars)

    # overwrite the observation with the forward simulation
    r.y_obs = r.forward_model(
        x=xb_vec_syn,
        mode="mean",
        **r.forwardKwArgs,
    )
    r.run()

    write_to_netcdf(r, flight_id, obs_time, version, test_id)

    logging.info("Writing synthetic retrieval to netcdf")
    path = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_clouds/retrieved",
    )
    ds_syn = r.to_xarray(
        x=xb_vec_syn[r.x_vars],
        x_err=r.oe.x_a_err,
        y=y_obs_hamp,
        b=xb_vec_syn[r.b_vars],
        obs_time=obs_time,
    )
    # y_obs = from hamp, y_sim = simulation of synthetic state
    # note: y_sim in ds_syn = y_obs in ds_a/ds_i/ds_op
    # note: y_obs in ds_syn != y_obs in ds_a/ds_i/ds_op
    ds_syn = ds_syn.rename({"y_sim": "y_obs", "y_obs": "y_sim"})
    ds_syn["y_obs"].attrs["long_name"] = "observed brightness temperature"
    ds_syn["y_obs"].attrs[
        "description"
    ] = "Brightness temperature of the radiometer (HAMP)"
    ds_syn["y_sim"].attrs["long_name"] = "simulated brightness temperature"
    ds_syn["y_sim"].attrs[
        "description"
    ] = "Brightness temperature of the forward model (perturbed state)"
    ds_syn.to_netcdf(
        os.path.join(
            path,
            f"x_syn_{flight_id}_{obs_time.strftime('%Y%m%d_%H%M%S')}_{version}{test_id}.nc",
        )
    )


def modify_state_random(r, cwp_distribution, cwp_min=0, cwp_max=0.5):
    """
    Samples a random state from state and model parameter covariance.

    The cwp variable can be perturbed unfirormly or normally. Uniform gives a
    better coverage of the state space, but normal is more realistic when
    identifying ambiguities in the retrieval.
    """

    x_syn = np.random.multivariate_normal(r.x_a, r.S_a)
    b_syn = np.random.multivariate_normal(r.b, r.S_b)
    xb_vec_syn = pd.Series(
        data=np.concatenate([x_syn, b_syn]),
        index=np.concatenate([r.x_vars, r.b_vars]),
    )
    if cwp_distribution == "uniform":
        # overwrite cwp with random value from uniform distribution
        xb_vec_syn.loc["cwp"] = np.random.uniform(low=cwp_min, high=cwp_max)
    elif cwp_distribution == "normal":
        pass
    else:
        raise ValueError("Unknown cwp distribution")

    # ensure that specularity is not below 0
    xb_vec_syn["specularity"] = np.max([0, xb_vec_syn["specularity"]])
    xb_vec_syn["specularity"] = np.min([1, xb_vec_syn["specularity"]])

    return xb_vec_syn


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting_file",
        type=str,
        help="Path to the setting file",
    )
    parser.add_argument(
        "--flight_id",
        type=str,
        help="Flight ID",
    )
    parser.add_argument(
        "--obs_time",
        type=str,
        help="Observation time",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Version of retrieval output",
    )
    parser.add_argument(
        "--syn_type",
        type=str,
        help="Type of synthetic retrieval",
        choices=["deterministic", "random"],
    )
    parser.add_argument(
        "--test_id",
        type=str,
        help="Test ID",
    )
    parser.add_argument(
        "--cwp_distribution",
        type=str,
        help="Distribution of cloud liquid water path",
        choices=["uniform", "normal"],
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed",
    )

    args = parser.parse_args()

    main(
        setting_file=args.setting_file,
        flight_id=args.flight_id,
        obs_time=args.obs_time,
        version=args.version,
        syn_type=args.syn_type,
        test_id=args.test_id,
        cwp_distribution=args.cwp_distribution,
        random_seed=args.random_seed,
    )
