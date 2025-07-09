"""
Optimal estimation retrieval for passive microwave observations over sea ice.
"""

import argparse
import inspect
import logging
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyOptimalEstimation as pyOE
import pyPamtra
import xarray as xr
from lizard.readers.band_pass import read_band_pass
from lizard.readers.hamp import read_hamp
from lizard.rt.polarization import vh2qh, vh2qv
from metpy.calc import (
    density,
    dewpoint_from_specific_humidity,
    mixing_ratio_from_relative_humidity,
    potential_temperature,
    precipitable_water,
    relative_humidity_from_specific_humidity,
)
from metpy.units import units

from si_clouds.helpers.smrt_medium import make_medium
from si_clouds.io.readers.setting import read_setting
from si_clouds.retrieval.apatm import (
    apriori_covariance_era5,
    apriori_mean_era5,
    profile_dropsonde_era5,
)
from si_clouds.retrieval.apsur import apriori_surface
from si_clouds.retrieval.surface import SMRTSurface

warnings.filterwarnings("ignore", category=FutureWarning, module="smrt.core.snowpack")

# this maps the name of the integrated hydrometeor content to the name of the
# specific hydrometeor content
CONTENT_VARS = {
    "cwp": "cwc",
    "rwp": "rwc",
    "swp": "swc",
}


def main(
    setting_file,
    flight_id,
    obs_time,
    version,
):
    """
    Run retrieval for a specific flight and time
    """

    obs_time = pd.Timestamp(obs_time)

    config_logger(
        log_file=os.path.join(
            os.environ["PATH_SEC"],
            f"data/sea_ice_clouds/retrieved/log_{version}",
            f"log_{flight_id}_{obs_time.strftime('%Y%m%d_%H%M%S')}_{version}.log",
        )
    )

    r = OEMRetrieval(
        setting_file=setting_file,
        flight_id=flight_id,
    )
    r.initialize(obs_time=obs_time)
    r.run()

    write_to_netcdf(
        r=r,
        flight_id=flight_id,
        obs_time=obs_time,
        version=version,
        test_id="",
    )


def write_to_netcdf(r, flight_id, obs_time, version, test_id=""):
    """
    Writes OEM retrieval to standardized netcdf file.
    """

    obs_time = pd.Timestamp(obs_time)

    ds_a, ds_op, ds_i = r.all_to_xarray(obs_time)

    path = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_clouds/retrieved",
    )
    logging.info(f"Writing retrieval results to: {path}")

    ds_a.to_netcdf(
        os.path.join(
            path,
            f"x_a_{flight_id}_{obs_time.strftime('%Y%m%d_%H%M%S')}_{version}{test_id}.nc",
        )
    )
    if ds_op is not None:
        ds_op.to_netcdf(
            os.path.join(
                path,
                f"x_op_{flight_id}_{obs_time.strftime('%Y%m%d_%H%M%S')}_{version}{test_id}.nc",
            )
        )
    ds_i.to_netcdf(
        os.path.join(
            path,
            f"x_i_{flight_id}_{obs_time.strftime('%Y%m%d_%H%M%S')}_{version}{test_id}.nc",
        )
    )

    logging.info("Done writing results.")


def config_logger(log_file):

    # create folder if it does not exists
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)


def add_variable_attributes(ds):
    """
    Add attributes to the variables of the dataset.
    """

    # coordinates
    ds["time"].encoding = {"units": "seconds since 1970-01-01"}
    ds["time"].attrs = {
        "long_name": "time",
        "description": "Observation time",
    }
    ds["channel"].attrs = {
        "long_name": "channel",
        "description": "Channel of the radiometer",
    }
    ds["channel1"].attrs = {
        "long_name": "channel",
        "description": "Channel of the radiometer",
    }
    ds["channel2"].attrs = {
        "long_name": "channel",
        "description": "Channel of the radiometer",
    }
    ds["model_level"].attrs = {
        "long_name": "ERA5 model level.",
        "description": "Model level of the ERA5 atmospheric profile",
    }
    ds["window_frequency"].attrs = {
        "long_name": "window frequency",
        "units": "GHz",
        "description": "Frequency of the SMRT sea ice RT simulation",
    }
    ds["hydro"].attrs = {
        "long_name": "hydrometeor type",
        "description": "Hydrometeor type included in the PAMTRA simulation",
    }
    ds["x_vars"].attrs = {
        "long_name": "state vector variables",
        "description": "State vector variables of the retrieval",
    }
    ds["x_vars1"].attrs = {
        "long_name": "state vector variables",
        "description": "State vector variables of the retrieval",
    }
    ds["x_vars2"].attrs = {
        "long_name": "state vector variables",
        "description": "State vector variables of the retrieval",
    }
    ds["b_vars"].attrs = {
        "long_name": "model variables",
        "description": "variables of the forward model",
    }
    ds["xb_vars"].attrs = {
        "long_name": "state vector and model variables",
        "description": "State vector and forward model variables",
    }
    ds["test"].attrs = {
        "long_name": "Chi2 test meaning",
        "description": (
            "0: optimal solution agrees with observation in Y space, "
            "1: observation agrees with prior in Y space, "
            "2: optimal solution agrees with prior in Y space, "
            "3: optimal solution agrees with prior in X space"
        ),
    }
    ds["update"].attrs = {
        "long_name": "update of state with the covariance matrices",
        "description": "Update of the state vector. Last update gives the "
        "final state vector.",
    }

    # variables
    ds["conv"].attrs = {
        "long_name": "convergence",
        "description": "Convergence of the retrieval.",
    }
    ds["conv_i"].attrs = {
        "long_name": "convergence iterations",
        "description": "Number of iterations until convergence.",
    }
    ds["conv_factor"].attrs = {
        "long_name": "convergence factor",
        "description": "Convergence factor of the retrieval.",
    }
    ds["conv_test"].attrs = {
        "long_name": "convergence test",
        "description": "Convergence test of the retrieval.",
    }
    ds["perturbation"].attrs = {
        "long_name": "perturbation",
        "description": "Perturbation factors of the state and model parameters.",
    }
    ds["dgf"].attrs = {
        "long_name": "degrees of freedom",
        "description": "Degrees of freedom of the retrieval.",
    }
    ds["dgf_x_vars"].attrs = {
        "long_name": "degrees of freedom per state variable",
        "description": "Degrees of freedom of the retrieval for each state variable.",
    }
    if ds["conv"].item():
        ds["chi2test_result"].attrs = {
            "long_name": "chi square test result",
            "description": "Chi square test result for the retrieval.",
        }
        ds["chi2test_chi2"].attrs = {
            "long_name": "chi square value",
            "description": "Chi square value.",
        }
        ds["chi2test_chi2crit"].attrs = {
            "long_name": "critical chi square value",
            "description": "Critical chi square value.",
        }
    ds["jacobian"].attrs = {
        "long_name": "jacobian matrix",
        "description": "Jacobian matrix of the state parameters.",
    }
    ds["jacobian_b"].attrs = {
        "long_name": "jacobian matrix",
        "description": "Jacobian matrix of forward model parameters.",
    }
    ds["dgf_i"].attrs = {
        "long_name": "degrees of freedom per update",
        "description": "Degrees of freedom of the retrieval for each update.",
    }
    ds["shannon_i"].attrs = {
        "long_name": "Shannon information",
        "description": "Shannon information of the retrieval for each update.",
    }
    ds["averaging_kernel"].attrs = {
        "long_name": "averaging kernel",
        "description": "Averaging kernel for each update.",
    }
    ds["conv_criterion"].attrs = {
        "long_name": "convergence criterion",
        "description": "Convergence criterion for each update.",
    }
    ds["unc_meas_eff"].attrs = {
        "long_name": "effective measurement uncertainty",
        "description": "Effective measurement uncertainty for each update.",
    }
    ds["unc_aposteriori"].attrs = {
        "long_name": "a posteriori uncertainty",
        "description": "A posteriori uncertainty for each update.",
    }
    ds["unc_apriori"].attrs = {
        "long_name": "a priori uncertainty",
        "description": "A priori uncertainty.",
    }
    ds["unc_b"].attrs = {
        "long_name": "model parameter uncertainty",
        "description": "Model parameter uncertainty.",
    }
    ds["unc_y"].attrs = {
        "long_name": "observation uncertainty",
        "description": "Observation uncertainty.",
    }
    ds["y_obs"].attrs = {
        "long_name": "observed brightness temperature",
        "units": "K",
        "description": "Brightness temperature of the radiometer.",
    }
    ds["y_sim"].attrs = {
        "long_name": "simulated brightness temperature",
        "units": "K",
        "description": "Brightness temperature of the forward model.",
    }
    ds["pressure"].attrs = {
        "long_name": "air pressure",
        "units": "hPa",
    }
    ds["height"].attrs = {
        "long_name": "height of atmospheric layer above mean sea level",
        "units": "m",
    }
    ds["altitude"].attrs = {
        "long_name": "flight altitude",
        "units": "m",
        "description": "Altitude of the aircraft",
    }
    ds["angle"].attrs = {
        "long_name": "viewing angle",
        "units": "deg",
        "description": "Viewing angle of the radiometer and PAMTRA simulation",
    }
    ds["t"].attrs = {
        "long_name": "air temperature",
        "units": "K",
        "description": "Temperature of the atmosphere",
    }
    ds["t_std"].attrs = {
        "long_name": "air temperature uncertainty",
        "units": "K",
        "description": "Uncertainty of the temperature of the atmosphere",
    }
    ds["qv"].attrs = {
        "long_name": "specific humidity",
        "units": "kg/kg",
        "description": "Specific humidity of the atmosphere",
    }
    ds["qv_std"].attrs = {
        "long_name": "specific humidity uncertainty",
        "units": "kg/kg",
        "description": "Uncertainty of the specific humidity of the atmosphere",
    }
    if "cwp" in ds:
        ds["cwp"].attrs = {
            "long_name": "cloud water path",
            "units": "kg/m^2",
            "description": "Integrated cloud water content of the atmosphere.",
        }
        ds["cwp_std"].attrs = {
            "long_name": "cloud water path uncertainty",
            "units": "kg/m^2",
            "description": "Integrated cloud water content uncertainty.",
        }
        ds["cwp_from_cwc"].attrs = {
            "long_name": "cloud water path",
            "units": "kg/m^2",
            "description": "Integrated cloud water content of the atmosphere (computed from cloud water content, it is not the state parameter).",
        }
    if "rwp" in ds:
        ds["rwp"].attrs = {
            "long_name": "rain water path",
            "units": "kg/m^2",
            "description": "Integrated rain water content of the atmosphere.",
        }
        ds["rwp_std"].attrs = {
            "long_name": "rain water path uncertainty",
            "units": "kg/m^2",
            "description": "Integrated rain water content uncertainty.",
        }
        ds["rwp_from_rwc"].attrs = {
            "long_name": "rain water path",
            "units": "kg/m^2",
            "description": "Integrated rain water content of the atmosphere (computed from rain water content, it is not the state parameter).",
        }
    if "swp" in ds:
        ds["swp"].attrs = {
            "long_name": "snow water path",
            "units": "kg/m^2",
            "description": "Integrated snow water content of the atmosphere.",
        }
        ds["swp_std"].attrs = {
            "long_name": "snow water path uncertainty",
            "units": "kg/m^2",
            "description": "Integrated snow water content uncertainty.",
        }
        ds["swp_from_swc"].attrs = {
            "long_name": "snow water path",
            "units": "kg/m^2",
            "description": "Integrated snow water content of the atmosphere (computed from snow water content, it is not the state parameter).",
        }
    ds["hydro_q"].attrs = {
        "long_name": "hydrometeor content",
        "units": "kg/kg",
        "description": "Hydrometeor content of the atmosphere",
    }
    ds["rh"].attrs = {
        "long_name": "relative humidity",
        "units": "%",
        "description": "Relative humidity of the atmosphere wrt. liquid water.",
    }
    ds["tpot"].attrs = {
        "long_name": "potential temperature",
        "units": "K",
        "description": "Potential temperature of the atmosphere.",
    }
    ds["iwv"].attrs = {
        "long_name": "integrated water vapor",
        "units": "kg/m^2",
        "description": "Integrated water vapor content of the atmosphere.",
    }
    ds["wind_slab_corr_length"].attrs = {
        "long_name": "wind slab correlation length",
        "units": "mm",
        "description": "Wind slab correlation length.",
    }
    ds["wind_slab_corr_length_std"].attrs = {
        "long_name": "wind slab correlation length uncertainty",
        "units": "mm",
        "description": "Uncertainty of the wind slab correlation length.",
    }
    if "wind_slab_density" in ds:
        ds["wind_slab_density"].attrs = {
            "long_name": "wind slab density",
            "units": "kg/m^3",
            "description": "Wind slab density.",
        }
    if "wind_slab_density_std" in ds:
        ds["wind_slab_density_std"].attrs = {
            "long_name": "wind slab density uncertainty",
            "units": "kg/m^3",
            "description": "Uncertainty of the wind slab density.",
        }
    if "wind_slab_thickness" in ds:
        ds["wind_slab_thickness"].attrs = {
            "long_name": "wind slab thickness",
            "units": "m",
            "description": "Wind slab thickness.",
        }
    if "wind_slab_thickness_std" in ds:
        ds["wind_slab_thickness_std"].attrs = {
            "long_name": "wind slab thickness uncertainty",
            "units": "m",
            "description": "Uncertainty of the wind slab thickness.",
        }
    ds["depth_hoar_corr_length"].attrs = {
        "long_name": "depth hoar correlation length",
        "units": "mm",
        "description": "Depth hoar correlation length.",
    }
    ds["depth_hoar_corr_length_std"].attrs = {
        "long_name": "depth hoar correlation length uncertainty",
        "units": "mm",
        "description": "Uncertainty of the depth hoar correlation length.",
    }
    if "depth_hoar_density" in ds:
        ds["depth_hoar_density"].attrs = {
            "long_name": "depth hoar density",
            "units": "kg/m^3",
            "description": "Depth hoar density.",
        }
    if "depth_hoar_density_std" in ds:
        ds["depth_hoar_density_std"].attrs = {
            "long_name": "Depth hoar density uncertainty",
            "units": "kg/m^3",
            "description": "Uncertainty of the depth hoar density.",
        }
    if "depth_hoar_thickness" in ds:
        ds["depth_hoar_thickness"].attrs = {
            "long_name": "depth hoar thickness",
            "units": "m",
            "description": "Depth hoar thickness.",
        }
    if "depth_hoar_thickness_std" in ds:
        ds["depth_hoar_thickness_std"].attrs = {
            "long_name": "depth hoar thickness uncertainty",
            "units": "m",
            "description": "Uncertainty of the depth hoar thickness.",
        }
    ds["specularity"].attrs = {
        "long_name": "specularity parameter",
        "description": "Specularity parameter for PAMTRA simulation (1=fully specular surface)",
    }
    ds["t_as"].attrs = {
        "long_name": "temperature at the snow-atmosphere interface",
        "units": "K",
        "description": "Temperature at the snow-atmosphere interface.",
    }
    ds["t_as_std"].attrs = {
        "long_name": "temperature at the snow-atmosphere interface uncertainty",
        "units": "K",
        "description": "Uncertainty of the temperature at the snow-atmosphere interface.",
    }
    ds["t_si"].attrs = {
        "long_name": "temperature at the snow-ice interface",
        "units": "K",
        "description": "Temperature at the snow-ice interface.",
    }
    ds["t_si_std"].attrs = {
        "long_name": "temperature at the snow-ice interface uncertainty",
        "units": "K",
        "description": "Uncertainty of the temperature at the snow-ice interface.",
    }
    ds["yi_fraction"].attrs = {
        "long_name": "fraction of young ice",
        "description": "Fraction of young ice.",
    }
    ds["yi_fraction_std"].attrs = {
        "long_name": "fraction of young ice uncertainty",
        "description": "Uncertainty of the fraction of young ice.",
    }
    ds["emissivity"].attrs = {
        "long_name": "surface emissivity",
        "description": "Microwave emissivity of the sea ice mixture derived "
        "from SMRT based on the retrieved physical parameters.",
    }
    ds["emissivity_lamb"].attrs = {
        "long_name": "lambertian surface emissivity",
        "description": "Microwave emissivity of the sea ice mixture derived "
        "from SMRT based on the retrieved physical parameters and modified for "
        "the PAMTRA simulation with Lambertian surface reflection.",
    }
    ds["surf_temp_eff"].attrs = {
        "long_name": "effective surface temperature",
        "units": "K",
        "description": "Microwave effective emitting layer temperature of "
        "the sea ice mixture derived from SMRT based on the retrieved "
        "physical parameters.",
    }

    # make sure that every variable has a standard name in their attributes
    for v in ds.data_vars:
        if "long_name" not in ds[v].attrs:
            logging.warning(f"Variable {v} has no standard_name attribute.")

    return ds


def add_global_attributes(
    ds,
    mission,
    platform,
    flight_id,
    flight_number,
    instrument,
    vars_therm,
    vars_hydro,
    vars_surf,
    creator,
    contact,
):
    """
    Add global attributes to the dataset.
    """

    ds.attrs = {
        "title": (
            f"Sea ice-atmosphere optimal estimation retrieval "
            f"output during {mission} {flight_number} from "
            f"{instrument} onboard {platform}"
        ),
        "description": (
            "The forward operators are SMRT for sea ice and PAMTRA for the "
            "atmosphere. "
            "The following atmospheric thermodynamic parameters are retrieved:"
            f"{', '.join(vars_therm)}. "
            "The following hydrometeor contents are retrieved: "
            f"{', '.join(vars_hydro)}. "
            "The following sea parameters are retrieved: "
            f"{', '.join(vars_surf)}."
        ),
        "source": "Derived from airborne observations and Optimal Estimation "
        "retrieval with the Passive and Active Microwave Radiative Transfer "
        "(PAMTRA) model (Mech et al., 2020) and Snow Microwave Radiative "
        "Transfer (SMRT) model (Picard et al., 2018).",
        "mission": mission,
        "platform": platform,
        "flight_id": flight_id,
        "instrument": instrument,
        "author": creator,
        "contact": contact,
        "created": str(np.datetime64("now")),
        "convention": "CF-1.8",
        "featureType": "trajectory",
    }

    return ds


class LoggerWriter:
    """Redirect stdout to logger."""

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != "\n":  # avoid logging extra empty lines
            self.logger.log(self.level, message)

    def flush(self):
        pass


class OEMRetrieval:
    """
    Optimal estimation retrieval from passive microwave observations
    over sea ice.
    """

    def __init__(self, setting_file, flight_id):
        """
        Initialize retrieval.
        """

        # flight information
        self.flight_id = flight_id
        self.mission, self.platform, self.name = self.flight_id.split("_")

        # retrieval setting
        self.setting = read_setting(file=setting_file)
        logging.info(f"Retrieval settings: {self.setting}")

        # create variable names
        self.y_vars = self.create_y_names()
        self.x_vars, self.x_ix = self.create_x_names()
        self.b_vars = np.array(self.setting.forward_model.model_variables)

        # covariance matrices
        self.S_y = self.create_covariance_matrix_y()
        self.S_a = np.zeros((len(self.x_vars), len(self.x_vars)))
        self.S_b = np.diag(self.setting.forward_model.model_variables_std) ** 2

        # read observation
        self.ds_obs = read_hamp(self.flight_id)

        # read instrument band pass information
        self.ds_bp = read_band_pass(
            instrument=self.setting.observation.instrument,
            satellite=self.setting.observation.satellite,
        )
        self.ds_bp = self.ds_bp.sel(channel=self.setting.observation.channels)

        # perturbation magnitudes
        self.perturbations = self.define_perturbations()

        # initialize a priori mean and observation vectors
        self.x_a = np.zeros(len(self.x_vars))
        self.y_obs = np.zeros(len(self.y_vars))

        # initialize model parameters
        self.b = np.array(self.setting.forward_model.model_variables_mean)

        # initialize latitude and longitude coordinates of observation
        self.lon = np.nan
        self.lat = np.nan

        # initialize forward model arguments
        self.forwardKwArgs = {
            "frequency": np.unique(self.ds_bp["avg_freq"]),
            "polarization": self.ds_bp["polarization"],
            "window_frequency": self.ds_bp["center_freq"]
            .sel(channel=self.setting.surface.channels)
            .values,
            "da_avg_freq": self.ds_bp["avg_freq"],
            "angle": self.setting.observation.angle,
            "altitude": self.setting.observation.altitude,
            "pressure": np.full(self.setting.atmosphere.n, np.nan),
            "height": np.full(self.setting.atmosphere.n, np.nan),
            "t_ut": np.full(
                self.setting.atmosphere.n - self.setting.atmosphere.nr,
                np.nan,
            ),
            "qv_ut": np.full(
                self.setting.atmosphere.n - self.setting.atmosphere.nr,
                np.nan,
            ),
            "surface": {},
        }

        # initialize optimal estimation object
        self.oe = np.nan

        # initialize the extended state dict, which contains iwv etc.
        self.full_x_dct = {}

        # manages smrt function call
        self.smrt_runs = 0
        self.previous_input_smrt = np.nan
        self.previous_output_smrt = (np.nan, np.nan)

    @staticmethod
    def add_to_dict(d, k):
        """
        One higher value than the current value is added to dict d under key k.
        If k already exists, the value is appended to the list of values.
        """

        try:
            i = max([i for j in d.values() for i in j]) + 1
        except ValueError:
            i = 0

        if k in d:
            d[k].append(i)
        else:
            d[k] = [i]

        return d

    def create_y_names(self):
        """
        Creates names of y variables for the retrieval. This is the HAMP
        channel order.

        Returns
        -------
        y_vars : np.array
            Array with variable names.
        """

        y_vars = []
        for channel in self.setting.observation.channels:
            y_vars.append(f"tb_{channel}")

        return y_vars

    def create_x_names(self):
        """
        Creates x variable names for the retrieval.

        Returns
        -------
        x_vars : np.array
            Array with variable names.
        x_ix : dict
            Dictionary with variable names as keys and the corresponding
            indices in the state vector as values.
        """

        x_vars = []
        x_ix = {}

        for var_type in self.setting.atmosphere.variables:
            for var in self.setting.atmosphere.variables[var_type]:
                if var_type == "thermodynamic":
                    for level in range(self.setting.atmosphere.nr):
                        x_vars.append(f"{var}_{level}")
                        x_ix = self.add_to_dict(x_ix, var)
                elif var_type == "hydro":
                    x_vars.append(f"{var}")
                    x_ix = self.add_to_dict(x_ix, var)

        for var in self.setting.surface.variables_r:
            x_vars.append(f"{var}")
            x_ix = self.add_to_dict(x_ix, var)

        x_vars = np.array(x_vars)

        return x_vars, x_ix

    def decode_state(self, x):
        """
        Decode the state vector variables as required for the forward operator.
        This requires indices from create_x_names.
        """

        x_dct = {}
        for var in self.x_ix:
            x_dct[var] = x[min(self.x_ix[var]) : max(self.x_ix[var]) + 1]

        return x_dct

    def water_path_to_content(self, full_x_dct, height, pressure):
        """
        This distributes the integrated water path to an equivalent amount
        of water content. If the specified amount cannot be reached due to
        physical constraints, then it is limited to the maximum possible
        amount. This is done for each hydrometeor type separately.

        Unit of the integrated water path is kg m-2.
        Unit of the water content is kg kg-1.
        """

        logging.info("Distributing water path to content")

        # constraints
        conditions = {
            "rwp": full_x_dct["t"] > 273.15,
            "cwp": full_x_dct["t"] > 235.15,
            "iwp": full_x_dct["t"] < 273.15,
            "swp": full_x_dct["t"] < 273.15,
        }
        max_content = 0.05  # maximum allowed hydrometeor content in kg/kg

        for hyd in self.setting.atmosphere.variables["hydro"]:
            ix_content = (
                (height >= self.setting.atmosphere.hydro_zmin)
                & (height < self.setting.atmosphere.hydro_zmax)
                & conditions[hyd]
            )
            logging.info(
                f"Number of levels with {CONTENT_VARS[hyd]}: {np.sum(ix_content)}"
            )

            if ix_content.sum() == 0:
                logging.info(
                    f"The {CONTENT_VARS[hyd]} is set to 0 kg/kg on every level."
                )
                continue

            # use gauss newton iteration to find the content that matches the
            # integrated water path
            # note that this could be also done in a simpler way from air density
            target = full_x_dct[hyd]
            max_iter = 100
            water_content = np.zeros_like(height)
            d_water_content = 0.001  # to compute jacobian
            tol = 1e-8
            for i in range(max_iter):
                r = (
                    self.calculate_integrated_hyd(
                        pressure=pressure,
                        temperature=full_x_dct["t"],
                        relhum=full_x_dct["relhum"],
                        mass_spec_content=water_content,
                        height=height,
                    )
                    - target
                )

                d_water_content_full = water_content + d_water_content
                jacobian = (
                    self.calculate_integrated_hyd(
                        pressure=pressure,
                        temperature=full_x_dct["t"],
                        relhum=full_x_dct["relhum"],
                        mass_spec_content=d_water_content_full,
                        height=height,
                    )
                    / d_water_content
                )

                delta = np.linalg.lstsq(
                    np.array([[jacobian]]), np.array([[-r]]), rcond=None
                )[0]
                water_content[ix_content] += delta[0, 0]

                # check if maximum water content is reached
                ix_max = water_content > max_content
                if np.any(ix_max):
                    logging.info(
                        f"Warning: Maximum allowed {CONTENT_VARS[hyd]} "
                        f"of {max_content} kg/kg reached. Cannot achieve the "
                        f"specified {hyd} of {target} kg/m$^2$."
                    )
                    water_content[ix_max] = max_content
                    break

                # convergence check
                if np.linalg.norm(delta) < tol:
                    logging.info(f"Convergence reached after {i} iterations.")
                    break

                if i == max_iter - 1:
                    logging.info(
                        f"Warning: Convergence not reached after {max_iter} iterations."
                    )

            # if the content is negative, set it to zero
            is_zero = water_content < 0
            if is_zero.sum() > 0:
                logging.info(
                    f"Warning: Negative {CONTENT_VARS[hyd]} values set to zero."
                )
                water_content[is_zero] = 0

            full_x_dct[CONTENT_VARS[hyd]] = water_content.copy()

            # recompute the integrated water path from the content
            full_x_dct[hyd + "_from_" + CONTENT_VARS[hyd]] = (
                self.calculate_integrated_hyd(
                    pressure=pressure,
                    temperature=full_x_dct["t"],
                    relhum=full_x_dct["relhum"],
                    mass_spec_content=full_x_dct[CONTENT_VARS[hyd]],
                    height=height,
                )
            )

        return full_x_dct

    def create_full_state_parameters(self, x, t_ut, qv_ut, pressure, height, surface):
        """
        Creates the full state parameters from the retrieval parameters.

        This function also converts all units to the correct format for the
        forward simulation. See simulate_tb_pamtra for the expected units.

        The dict is used for the forward simulation and for saving the final
        retrieved profiles.
        """

        logging.info("Creating full state parameters")

        full_x_dct = {}  # full state vector for forward simulation

        # split state vector by variables
        x_dct = self.decode_state(x)

        # add upper troposphere values
        # this considers case that t or qv are not included as state parameters
        if "t" in x_dct:
            full_x_dct["t"] = np.concatenate([x_dct["t"], t_ut])
        else:
            full_x_dct["t"] = t_ut
        if "qv" in x_dct:
            full_x_dct["qv"] = np.concatenate([x_dct["qv"], qv_ut])
        else:
            full_x_dct["qv"] = qv_ut

        # convert qv in kg kg-1
        full_x_dct["qv"] = (10 ** full_x_dct["qv"]) * 1e-3

        # calculate relative humidity from specific humidity
        full_x_dct["relhum"] = (
            relative_humidity_from_specific_humidity(
                specific_humidity=full_x_dct["qv"],
                temperature=full_x_dct["t"] * units.K,
                pressure=pressure * units.Pa,
            )
            * 100
        )
        full_x_dct["relhum"] = full_x_dct["relhum"].magnitude
        full_x_dct["relhum"][full_x_dct["relhum"] < 0] = 0
        full_x_dct["relhum"][full_x_dct["relhum"] > 200] = 200

        # make sure that no nan values in relhum exist
        assert not np.any(np.isnan(full_x_dct["relhum"]))

        # hydrometeros
        for hyd in self.setting.atmosphere.variables["hydro"]:
            full_x_dct[hyd] = x_dct[hyd].item()

        # surface parameters
        for v in self.setting.surface.variables_r:
            full_x_dct[v] = x_dct[v].item()

        # add surface variables that are not retrieved but fixed
        for v in surface:
            full_x_dct[v] = surface[v]

        # revert the scaling of the state parameters back to the nominal unit
        for v in self.setting.general.scales:
            if v in self.setting.surface.variables_r:
                print(print(f"Multiplying {v} with {self.setting.general.scales[v]}"))
                full_x_dct[v] = full_x_dct[v] * self.setting.general.scales[v]

        # apply parameter limits
        full_x_dct = self.apply_parameter_limits(full_x_dct)

        # compute water content for PAMTRA (after cwp was checked for bounds)
        full_x_dct = self.water_path_to_content(
            full_x_dct=full_x_dct,
            height=height,
            pressure=pressure,
        )

        # set hydrometeor content for PAMTRA
        # 0 or as specified from the water path state parameters
        full_x_dct["hydro_q"] = np.zeros(
            (len(height), len(self.setting.pamtra.descriptor_file_order)),
        )
        for i, hyd in enumerate(self.setting.pamtra.descriptor_file_order):
            if hyd in self.setting.atmosphere.variables["hydro"]:
                full_x_dct["hydro_q"][:, i] = full_x_dct[CONTENT_VARS[hyd]]

        # compute integrated quantities
        full_x_dct["iwv"] = self.calculate_iwv(
            pressure=pressure,
            temperature=full_x_dct["t"],
            specific_humidity=full_x_dct["qv"],
        )

        return full_x_dct

    def create_full_state_parameters_uncertainty(
        self,
        x_err,
        full_x_dct,
        height,
        surface,
    ):
        """
        Expands the uncertainty of state parameters to all heights to be
        equivalent with the state parameters themselves.

        Similar to create_full_state_parameters().

        This function also converts all units of the uncertainty to those
        of the state vector.

        This is only used for saving the simulated profile.
        """

        logging.info("Creating full uncertainty parameters")

        full_x_err_dct = {}  # full state vector for forward simulation

        # split state vector by variables
        x_err_dct = self.decode_state(x_err)

        # add upper troposphere values
        if "t" in x_err_dct:
            full_x_err_dct["t"] = np.concatenate(
                [
                    x_err_dct["t"],
                    np.full_like(self.forwardKwArgs["t_ut"], np.nan),
                ]
            )
        else:
            full_x_err_dct["t"] = np.full_like(self.forwardKwArgs["t_ut"], np.nan)
        if "qv" in x_err_dct:
            full_x_err_dct["qv"] = np.concatenate(
                [
                    x_err_dct["qv"],
                    np.full_like(self.forwardKwArgs["qv_ut"], np.nan),
                ]
            )
        else:
            full_x_err_dct["qv"] = np.full_like(self.forwardKwArgs["qv_ut"], np.nan)

        # uncertainty of spec. humidity to linear
        full_x_err_dct["qv"] = self.log_std2lin_std(
            sigma_f=full_x_err_dct["qv"], A=full_x_dct["qv"]
        )

        # hydrometeors
        for hyd in self.setting.atmosphere.variables["hydro"]:
            full_x_err_dct[hyd] = x_err_dct[hyd].values

        # surface parameter uncertainties
        for v in self.setting.surface.variables_r:
            full_x_err_dct[v] = x_err_dct[v].item()

        # surface parameters that are not retrieved but fixed
        for v in surface:
            full_x_err_dct[v] = 0

        # revert the scaling of the state parameters back to their nominal unit
        for v in self.setting.general.scales:
            if v in self.setting.surface.variables_r:
                print(f"Multiplying {v}_std with {self.setting.general.scales[v]}")
                full_x_err_dct[v] = full_x_err_dct[v] * self.setting.general.scales[v]

        return full_x_err_dct

    def apply_parameter_limits(self, full_x_dct):
        """
        Apply parameter limits.
        """

        logging.info("Applying parameter limits given in the yaml file.")

        for v in self.setting.general.x_lower_limit:
            if v in full_x_dct:
                if full_x_dct[v] < self.setting.general.x_lower_limit[v]:
                    logging.info(
                        f"Parameter '{v}' exceeds lower limit: "
                        f"{full_x_dct[v]} < {self.setting.general.x_lower_limit[v]}. "
                        "Resetting to lower limit."
                    )
                    full_x_dct[v] = self.setting.general.x_lower_limit[v]

        for v in self.setting.general.x_upper_limit:
            if v in full_x_dct:
                if full_x_dct[v] > self.setting.general.x_upper_limit[v]:
                    logging.info(
                        f"Parameter '{v}' exceeds upper limit: "
                        f"{full_x_dct[v]} > {self.setting.general.x_upper_limit[v]}. "
                        "Resetting to upper limit."
                    )
                    full_x_dct[v] = self.setting.general.x_upper_limit[v]

        return full_x_dct

    @staticmethod
    def calculate_iwv(pressure, temperature, specific_humidity):
        """
        Calculates integrated water vapor for a given profile.

        Parameters
        ----------
        pressure : float
            Pressure in Pa.
        temperature : float
            Temperature in K.
        specific_humidity : float
            Specific humidity in kg kg-1.

        Returns
        -------
        iwv : float
            Integrated water vapor in kg m-2.
        """

        dewpoint = dewpoint_from_specific_humidity(
            pressure=pressure * units.Pa,
            temperature=temperature * units.K,
            specific_humidity=specific_humidity,
        )

        iwv = precipitable_water(
            pressure=pressure * units.Pa,
            dewpoint=dewpoint,
        ).magnitude

        return iwv

    @staticmethod
    def calculate_integrated_hyd(
        pressure, temperature, relhum, mass_spec_content, height
    ):
        """
        Calculate integrated hydrometeor contents

        Parameters
        ----------
        pressure : float
            Pressure in Pa.
        temperature : float
            Temperature in K.
        relhum : float
            Relative humidity in %.
        mass_spec_content : float
            Mass specific hydrometeor content in kg kg-1.
        height : float
            Height in m.

        Returns
        -------
        water_path : float
            Integrated water path in kg m-2.
        """

        mixing_ratio = mixing_ratio_from_relative_humidity(
            pressure=pressure * units.Pa,
            temperature=temperature * units.K,
            relative_humidity=relhum * 1e-2,
        )

        rho = density(
            pressure=pressure * units.Pa,
            temperature=temperature * units.K,
            mixing_ratio=mixing_ratio,
        )

        water_path = np.trapezoid(y=(mass_spec_content * rho).magnitude, x=height)

        return water_path

    def create_covariance_matrix_y(self):
        """
        Creates covariance matrix of TB observations. The standard deviation
        is taken from the user settings.
        """

        S_y = np.diag(
            [
                self.setting.observation.uncertainty[channel] ** 2
                for channel in self.setting.observation.channels
            ]
        )

        return S_y

    def create_covariance_matrix_x(self, obs_time):
        """
        Get a priori covariance matrix.
        """

        # temperature and specific humidity
        # only if atmospheric variables are in state vector
        if self.setting.atmosphere.nr > 0:
            atm_cov = apriori_covariance_era5(self.setting.atmosphere.nr)
            logging.info(
                f"ERA5 covariance scaling: {self.setting.atmosphere.era5_cov_scaling}"
            )
            atm_cov = atm_cov * self.setting.atmosphere.era5_cov_scaling
            ix0 = min(self.x_ix[self.setting.atmosphere.variables["thermodynamic"][0]])
            ix1 = (
                max(self.x_ix[self.setting.atmosphere.variables["thermodynamic"][-1]])
                + 1
            )
            self.S_a[ix0:ix1, ix0:ix1] = atm_cov

        # hydrometeors
        # only if hydrometeor variables are in state vector
        if len(self.setting.atmosphere.variables["hydro"]) > 0:
            hyd_cov = np.diag(
                np.array(self.setting.atmosphere.hydro_std) ** 2,
            )
            ix0 = min(self.x_ix[self.setting.atmosphere.variables["hydro"][0]])
            ix1 = max(self.x_ix[self.setting.atmosphere.variables["hydro"][-1]]) + 1
            self.S_a[ix0:ix1, ix0:ix1] = hyd_cov

        # smrt
        # make sure that lon or lat are not nan
        if np.isnan(self.lon) or np.isnan(self.lat):
            raise ValueError("Longitude or latitude is not set")

        dct_apsurf = apriori_surface(
            flight_id=self.flight_id,
            lat=self.lat,
            lon=self.lon,
            time=obs_time,
            source_t_as=self.setting.surface.source_t_as,
            ir_emissivity=self.setting.surface.ir_emissivity,
            smrt_layers_filename=self.setting.surface.smrt_layers_filename,
            tas2tsi_factor=self.setting.surface.tas2tsi_factor,
        )
        dct_apsurf["t_as_std"] = self.setting.surface.t_as_std
        dct_apsurf["t_si_std"] = self.setting.surface.t_si_std
        dct_apsurf["yi_fraction_std"] = self.setting.surface.yi_fraction_std

        # apply scaling to the surface parameters
        for v in self.setting.general.scales:
            if v in self.setting.surface.variables_r:
                print(f"Dividing {v}_std by {self.setting.general.scales[v]}")
                dct_apsurf[v + "_std"] = (
                    dct_apsurf[v + "_std"] / self.setting.general.scales[v]
                )

        ix0 = min(self.x_ix[self.setting.surface.variables_r[0]])
        ix1 = max(self.x_ix[self.setting.surface.variables_r[-1]]) + 1
        self.S_a[ix0:ix1, ix0:ix1] = np.diag(
            [dct_apsurf[v + "_std"] ** 2 for v in self.setting.surface.variables_r]
        )

    def create_a_priori_x(self, obs_time):
        """
        Get a priori state vector. Similar to the covariance matrix approach.
        Additionally, get pressure and height from ERA5 as retrieval grid and
        temperature and humidity of the upper troposphere that is not
        retrieved.

        State vector order:

        - t at specific time/lat/lon from ERA5
        - qv at specific time/lat/lon from ERA5
        - hydrometeor contents
        - surface parameters
        """

        # temperature and specific humidity
        if self.setting.atmosphere.source == "era5":
            t, qv, p, h = apriori_mean_era5(
                flight_id=self.flight_id,
                time=obs_time,
                lat=self.lat,
                lon=self.lon,
            )
        elif self.setting.atmosphere.source == "dropsonde_era5":
            t, qv, p, h = profile_dropsonde_era5(
                flight_id=self.flight_id,
                time=obs_time,
                lat=self.lat,
                lon=self.lon,
            )

        # add lower troposphere layers to state vector
        # only if atmospheric variables are in state vector
        if self.setting.atmosphere.nr > 0:
            ix0 = min(self.x_ix[self.setting.atmosphere.variables["thermodynamic"][0]])
            ix1 = (
                max(self.x_ix[self.setting.atmosphere.variables["thermodynamic"][-1]])
                + 1
            )
            self.x_a[ix0:ix1] = np.concatenate(
                [
                    t[: self.setting.atmosphere.nr],
                    qv[: self.setting.atmosphere.nr],
                ]
            )

        # hydrometeors
        # only if hydrometeor variables are in state vector
        if len(self.setting.atmosphere.variables["hydro"]) > 0:
            hyd_mean = np.array(self.setting.atmosphere.hydro_mean)
            ix0 = min(self.x_ix[self.setting.atmosphere.variables["hydro"][0]])
            ix1 = max(self.x_ix[self.setting.atmosphere.variables["hydro"][-1]]) + 1
            self.x_a[ix0:ix1] = hyd_mean

        # smrt
        dct_apsurf = apriori_surface(
            flight_id=self.flight_id,
            lat=self.lat,
            lon=self.lon,
            time=obs_time,
            source_t_as=self.setting.surface.source_t_as,
            ir_emissivity=self.setting.surface.ir_emissivity,
            smrt_layers_filename=self.setting.surface.smrt_layers_filename,
            tas2tsi_factor=self.setting.surface.tas2tsi_factor,
        )
        if self.setting.surface.source_t_as == "yaml":
            dct_apsurf["t_as"] = self.setting.surface.t_as
        if self.setting.surface.source_t_si == "yaml":
            dct_apsurf["t_si"] = self.setting.surface.t_si
        dct_apsurf["yi_fraction"] = self.setting.surface.yi_fraction
        dct_apsurf["wind_slab_volumetric_liquid_water"] = (
            self.setting.surface.wind_slab_volumetric_liquid_water
        )
        ix0 = min(self.x_ix[self.setting.surface.variables_r[0]])
        ix1 = max(self.x_ix[self.setting.surface.variables_r[-1]]) + 1
        self.x_a[ix0:ix1] = np.array(
            [dct_apsurf[v] for v in self.setting.surface.variables_r]
        )

        assert np.isnan(self.x_a).sum() == 0

        # scale the state parameters to the retrieval space
        for v in self.setting.general.scales:
            if v in self.x_vars:
                print(f"Dividing {v} by {self.setting.general.scales[v]}")
                self.x_a[self.x_vars == v] = (
                    self.x_a[self.x_vars == v] / self.setting.general.scales[v]
                )

        # add upper troposphere temperature and humidity, pressure, and height
        # to forward parameter dictionary
        self.forwardKwArgs["t_ut"] = t[self.setting.atmosphere.nr :]
        self.forwardKwArgs["qv_ut"] = qv[self.setting.atmosphere.nr :]
        self.forwardKwArgs["pressure"] = p
        self.forwardKwArgs["height"] = h

        # pass surface parameters that are not retrieved to forward model
        for v in self.setting.surface.variables:
            if v not in self.setting.surface.variables_r:
                self.forwardKwArgs["surface"][v] = dct_apsurf[v]

    def create_observation_y(self, obs_time):
        """
        Get observation space vector. This contains the TB at a specific time.
        """

        self.y_obs = self.ds_obs.tb.sel(
            time=obs_time, channel=self.setting.observation.channels
        ).values

    def read_position_of_observation(self, obs_time):
        """
        Get latitude, longitude, and altitude of the observation. When
        satellites are simulated, the altitude is taken from the yaml file.
        """

        self.lat = self.ds_obs.lat.sel(time=obs_time).item()
        self.lon = self.ds_obs.lon.sel(time=obs_time).item()
        if self.forwardKwArgs["altitude"] is None:
            self.forwardKwArgs["altitude"] = self.ds_obs.alt.sel(time=obs_time).item()
        else:
            assert self.setting.observation.altitude > 100000
            assert self.setting.observation.instrument in [
                "MHS",
                "ATMS",
                "AMSR2",
                "SSMIS",
            ]

    def forward_model_smrt(
        self,
        full_x_dct,
        b_dct,
        window_frequency,
        frequency,
        angle,
        mode,
        layer_name,
        layer_parameter,
    ):
        """
        Runs SMRT if this function is called for the first time or if the
        input changed from the past simulation.

        Parameters
        ----------
        full_x_dct : dict
            Full state parameters for the forward simulation.
        b_dct : dict
            Dictionary with model parameters. These will be treated like the
            state parameters
        window_frequency : np.array
            SMRT will simulate these frequencies.
        frequency : np.array
            SMRT simulation will be interpolated to these frequencies. These
            correspond the the frequencies of the PAMTRA simulation.
        mode : str
            Select "mean" when running this during retrieval and "gaussian"
            when computing forward model uncertainty. In this case, the
            layer name and layer parameter need to be provided.
        layer_name : str
            Name of the layer for which the given layer parameter will be
            perturbed
        layer_parameter : str
            Name of the parameter that will be perturbed.

        Returns
        -------
        emissivity : np.array
            Emissivity of the surface interpolated to the frequencies.
        surf_temp_eff : np.array
            Surface temperature interpolated to the frequencies.
        """

        # this is a workaround to filter surface variables from b
        expected_variables = [
            "wind_slab_thickness",
            "depth_hoar_thickness",
            "wind_slab_density",
            "depth_hoar_density",
            "wind_slab_corr_length",
            "depth_hoar_corr_length",
            "t_as",
            "t_si",
            "yi_fraction",
            "wind_slab_volumetric_liquid_water",
        ]

        # save smrt input
        current_input_smrt = (
            *[full_x_dct[v] for v in self.setting.surface.variables],
            *[
                b_dct[v]
                for v in self.setting.forward_model.model_variables
                if v in expected_variables
            ],
        )

        print(current_input_smrt)

        run_smrt = False
        if (self.smrt_runs == 0) or (mode == "gaussian"):
            run_smrt = True
        else:
            for v0, v1 in zip(self.previous_input_smrt, current_input_smrt):
                diff = np.abs(v0 - v1)
                if isinstance(diff, np.ndarray):
                    changed = (diff > 1e-9).any()
                else:
                    changed = diff > 1e-9
                if changed:
                    run_smrt = True
                    break

        if run_smrt:
            # create profiles for smrt simulation and update parameters
            # note that correlation length in state is mm, here converted to m
            # the selection ensures that the parameter is grabbed from either
            # the full state vector or the model parameters
            if mode == "mean":  # retrieval mode
                overwrite = {
                    "base_layers": {
                        "snow": {
                            "wind_slab": {
                                "corr_length": (
                                    full_x_dct.get("wind_slab_corr_length")
                                    if full_x_dct.get("wind_slab_corr_length")
                                    is not None
                                    else b_dct["wind_slab_corr_length"]
                                )
                                * 1e-3,
                                "density": (
                                    full_x_dct.get("wind_slab_density")
                                    if full_x_dct.get("wind_slab_density") is not None
                                    else b_dct["wind_slab_density"]
                                ),
                                "thickness": (
                                    full_x_dct.get("wind_slab_thickness")
                                    if full_x_dct.get("wind_slab_thickness") is not None
                                    else b_dct["wind_slab_thickness"]
                                ),
                                "volumetric_liquid_water": (
                                    full_x_dct.get("wind_slab_volumetric_liquid_water")
                                    if full_x_dct.get(
                                        "wind_slab_volumetric_liquid_water"
                                    )
                                    is not None
                                    else b_dct["wind_slab_volumetric_liquid_water"]
                                ),
                            },
                            "depth_hoar": {
                                "corr_length": (
                                    full_x_dct.get("depth_hoar_corr_length")
                                    if full_x_dct.get("depth_hoar_corr_length")
                                    is not None
                                    else b_dct["depth_hoar_corr_length"]
                                )
                                * 1e-3,
                                "density": (
                                    full_x_dct.get("depth_hoar_density")
                                    if full_x_dct.get("depth_hoar_density") is not None
                                    else b_dct["depth_hoar_density"]
                                ),
                                "thickness": (
                                    full_x_dct.get("depth_hoar_thickness")
                                    if full_x_dct.get("depth_hoar_thickness")
                                    is not None
                                    else b_dct["depth_hoar_thickness"]
                                ),
                            },
                        }
                    },
                    "mediums": {
                        self.setting.surface.medium_young_ice: {
                            "temperature_as": (
                                full_x_dct.get("t_as")
                                if full_x_dct.get("t_as") is not None
                                else b_dct["t_as"]
                            ),
                        },
                        self.setting.surface.medium_snow_covered_ice: {
                            "temperature_as": (
                                full_x_dct.get("t_as")
                                if full_x_dct.get("t_as") is not None
                                else b_dct["t_as"]
                            ),
                            "temperature_si": (
                                full_x_dct.get("t_si")
                                if full_x_dct.get("t_si") is not None
                                else b_dct["t_si"]
                            ),
                            "temperature_iw": (
                                full_x_dct.get("t_si")
                                if full_x_dct.get("t_si") is not None
                                else b_dct["t_si"]
                            ),
                        },
                    },
                }

            elif mode == "gaussian":  # forward model uncertainty mode
                # overwrite only the temperature, which comes from a priori
                # data. all other layer parameters are used as stated in yaml
                overwrite = {
                    "mediums": {
                        self.setting.surface.medium_young_ice: {
                            "temperature_as": full_x_dct.get("t_as") or b_dct["t_as"],
                            "temperature_iw": full_x_dct.get("t_as") or b_dct["t_as"],
                        },
                        self.setting.surface.medium_snow_covered_ice: {
                            "temperature_as": full_x_dct.get("t_as") or b_dct["t_as"],
                            "temperature_si": full_x_dct.get("t_si") or b_dct["t_si"],
                            "temperature_iw": full_x_dct.get("t_si") or b_dct["t_si"],
                        },
                    },
                }

            profile_snow_covered_ice = make_medium(
                filename=self.setting.surface.smrt_layers_filename,
                medium=self.setting.surface.medium_snow_covered_ice,
                mode=mode,
                overwrite=overwrite,
                layer_name=layer_name,
                layer_parameter=layer_parameter,
            )
            profile_young_ice = make_medium(
                filename=self.setting.surface.smrt_layers_filename,
                medium=self.setting.surface.medium_young_ice,
                mode=mode,
                overwrite=overwrite,
                layer_name=layer_name,
                layer_parameter=layer_parameter,
            )

            logging.info("Starting SMRT simulation with these profiles:")
            logging.info(profile_snow_covered_ice)
            logging.info(profile_young_ice)
            tic = time.time()
            if full_x_dct["yi_fraction"] > 0:
                sm = SMRTSurface(
                    mwr_frequency=window_frequency,
                    angle=self.setting.pamtra.angle[:-3],  # without high angles
                    profiles=[profile_snow_covered_ice, profile_young_ice],
                    n_jobs=self.setting.general.n_processes_smrt,
                )
                sm.calculate_mixture(
                    fraction=[
                        1 - full_x_dct["yi_fraction"],
                        full_x_dct["yi_fraction"],
                    ]
                )
            else:
                sm = SMRTSurface(
                    mwr_frequency=window_frequency,
                    angle=self.setting.pamtra.angle[:-3],  # without high angles
                    profiles=[profile_snow_covered_ice],
                    n_jobs=self.setting.general.n_processes_smrt,
                )
                sm.calculate_mixture(fraction=[1])
            toc = time.time()
            logging.info("SMRT simulation: {:.2f} s".format(toc - tic))
            logging.info("SMRT simulation emissivity:")
            logging.info(sm.ds.e_m)
            logging.info("SMRT simulation effective surface temperature:")
            logging.info(sm.ds.ts_m)

            # interpolate emissivity and temperature linearly to angles/channel
            ds_int = sm.ds.interp(
                coords={
                    "frequency": frequency * 1e9,
                    "theta": self.setting.pamtra.angle,
                },
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            assert ds_int.polarization.values.tolist() == ["V", "H"]
            emissivity = ds_int.e_m.transpose(
                "polarization", "frequency", "theta"
            ).values
            if np.any(emissivity < 0) or np.any(emissivity > 1):
                logging.error("Emissivity out of bounds. Setting to 0 or 1.")
            emissivity[emissivity < 0] = 0
            emissivity[emissivity > 1] = 1
            if isinstance(angle, np.ndarray) or isinstance(angle, list):
                angle = np.mean(angle)
            surf_temp_eff = (
                ds_int.ts_m.mean("polarization")
                .interp(theta=angle, method="linear")
                .values
            )

            # save input and output
            self.previous_output_smrt = (emissivity, surf_temp_eff)
            self.previous_input_smrt = current_input_smrt

            # counting up only during retrieval mode ensures that SMRT
            # gets called always during gaussian mode
            if mode == "mean":
                self.smrt_runs += 1

        else:
            logging.info("Use output from previous SMRT simulation")
            emissivity, surf_temp_eff = self.previous_output_smrt

        return emissivity, surf_temp_eff

    def forward_model_pamtra(
        self,
        full_x_dct,
        frequency,
        altitude,
        height,
        pressure,
        emissivity,
        surf_temp_eff,
        surf_refl,
        da_avg_freq,
        angle,
        polarization,
    ):
        """
        Atmospheric part of the forward simulations using the PAMTRA model.
        Frequencies are simulated in parallel. Note that this simulation is
        only for nadir
        """

        # pamtra simulation of each frequency in parallel
        tic = time.time()
        with multiprocessing.Pool(
            processes=self.setting.general.n_processes_pamtra
        ) as pool:
            lst_tb = pool.starmap(
                self.simulate_tb_pamtra,
                [
                    (
                        altitude,
                        freq,
                        height,
                        pressure,
                        full_x_dct["t"],
                        full_x_dct["relhum"],
                        full_x_dct["hydro_q"],
                        emissivity[:, i_freq : (i_freq + 1), :],  # keep shape
                        surf_temp_eff[i_freq],
                        surf_refl,
                    )
                    for i_freq, freq in enumerate(frequency)
                ],
            )
        toc = time.time()
        logging.info("RT simulation: {:.2f} s".format(toc - tic))

        da_tb = xr.DataArray(
            np.array(lst_tb)[:, :16, :],
            dims=["frequency", "angle", "polarization"],
            coords={
                "frequency": frequency,
                "angle": self.setting.pamtra.angle,
                "polarization": ["V", "H"],
            },
        )

        # interpolate tb to the given angle
        da_tb = da_tb.interp(
            angle=angle,
            method="linear",
        )

        # average the dsb channels
        da_tb = da_tb.sel(frequency=da_avg_freq).mean("n_avg_freq")

        # compute mixed polarization and add to tb data array
        da_qv = vh2qv(
            v=da_tb.sel(polarization="V"),
            h=da_tb.sel(polarization="H"),
            incidence_angle=angle,
            alt=altitude,
            angle_deg=True,
        ).expand_dims({"polarization": ["QV"]}, axis=-1)
        da_qh = vh2qh(
            v=da_tb.sel(polarization="V"),
            h=da_tb.sel(polarization="H"),
            incidence_angle=angle,
            alt=altitude,
            angle_deg=True,
        ).expand_dims({"polarization": ["QH"]}, axis=-1)
        da_q = xr.concat([da_qv, da_qh], dim="polarization")
        da_tb_q = xr.concat([da_tb, da_q], dim="polarization")

        # select the polatization for each channel
        da_pol = (
            polarization.astype("str")
            .str.replace("1", "QV")
            .str.replace("2", "QH")
            .str.replace("3", "V")
            .str.replace("4", "H")
        )
        da_tb = da_tb_q.sel(polarization=da_pol)

        # to numpy array
        y_mod = da_tb.values

        return y_mod

    def simulate_tb_pamtra(
        self,
        altitude,
        frequency,
        height,
        pressure,
        temperature,
        relhum,
        hydro_q,
        emissivity,
        surf_temp_eff,
        surf_refl,
        lon=1,
        lat=1,
        timestamp=0,
        wind10u=0,
        wind10v=0,
        sfc_type=-1,
        sfc_model=-1,
        salinity=33,
        max_height=14000,
        outlevel=0,
        i_frequency=0,
        return_pam=False,
    ):
        """
        Simulate TB at a single frequency with PAMTRA.

        Units expected for PAMTRA profiles:

        - altitude: m
        - frequency: GHz
        - height: m
        - pressure: Pa
        - temperature: K
        - relhum: %
        - hydro_q: kg kg-1
        - emissivity: 1
        - surf_temp_eff: K
        - surf_refl: -
        - lon: deg
        - lat: deg
        - timestamp: s
        - wind10u: m s-1
        - wind10v: m s-1
        - sfc_type: -
        - sfc_model: -
        - salinity: ppt

        emissivity shape: (2=pol(v, h), 1=frequency, 16=angle(0, ..., 85))
        """

        # simulate atmosphere only up to max_height
        height_ix = height < max_height
        height = height[height_ix]
        pressure = pressure[height_ix]
        temperature = temperature[height_ix]
        relhum = relhum[height_ix]
        hydro_q = hydro_q[height_ix, :]

        # dummy atmosphere variable
        arr_zero = np.zeros_like(pressure[np.newaxis, np.newaxis, :])
        arr_zero_hydro = np.zeros_like(hydro_q[np.newaxis, np.newaxis, :, :])

        # if altitude is a single value, convert to 3D array
        altitude = np.array([altitude])[np.newaxis, np.newaxis, :]
        if len(altitude.shape) == 4:
            altitude = altitude[0, ...]

        # define pamtra profile
        profile = {
            "lon": np.array([[lon]]),
            "lat": np.array([[lat]]),
            "timestamp": np.array([[timestamp]]),
            "wind10u": np.array([[wind10u]]),
            "wind10v": np.array([[wind10v]]),
            "sfc_slf": np.array([[0]]),
            "sfc_sif": np.array([[0]]),
            "sfc_salinity": np.array([[salinity]]),
            "hydro_q": hydro_q[np.newaxis, np.newaxis, :, :],
            "hydro_n": arr_zero_hydro,
            "hydro_reff": arr_zero_hydro,
            "airturb": arr_zero,
            "wind_w": arr_zero,
            "wind_uv": arr_zero,
            "turb_edr": arr_zero,
            "groundtemp": np.array([[surf_temp_eff]]),
            "press": pressure[np.newaxis, np.newaxis, :],
            "hgt": height[np.newaxis, np.newaxis, :],
            "relhum": relhum[np.newaxis, np.newaxis, :],
            "temp": temperature[np.newaxis, np.newaxis, :],
            "obs_height": altitude,
            "sfc_type": sfc_type,
            "sfc_model": sfc_model,
            "sfc_refl": surf_refl,
            "sfc_emissivity": emissivity,
        }

        # create PAMTRA object
        pam = pyPamtra.pyPamtra()

        pam.df.readFile(self.setting.pamtra.descriptor_file)
        pam.createProfile(**profile)

        dct_nml = self.setting.pamtra.nmlSet.copy()
        pam.nmlSet.update(dct_nml)
        pam.nmlSet["save_ssp"] = True  # saves the atmospheric opacity
        pam.addIntegratedValues()  # not needed, just for consistency

        # simulate
        pam.runPamtra(frequency)

        # grid_x, grid_y, outlevel, angle, frequency, polarization
        tb = pam.r["tb"][0, 0, outlevel, :, i_frequency, :]

        logging.info(
            f"Hydrometeor water paths from PAMTRA in kg/m2: {pam.p['hydro_wp']}"
        )
        logging.info(f"IWV from PAMTRA in kg/m2: {pam.p['iwv']}")

        if return_pam:
            return tb, pam

        else:
            return tb

    @staticmethod
    def get_lamb_emissivity(emissivity):
        """
        Returns Lambertian emissivity from a specular SMRT emissivity with the
        following shape: (polarization, frequency, angle).
        The first angle index is the nadir angle. The returned vector retains
        the shape of the input emissivity.
        """

        return emissivity[0:1, :, 0:1]

    def forward_model(
        self,
        x,
        frequency,
        polarization,
        window_frequency,
        da_avg_freq,
        angle,
        altitude,
        pressure,
        height,
        t_ut,
        qv_ut,
        surface,
        mode="mean",
        layer_name=None,
        layer_parameter=None,
    ):
        """
        Forward model based on SMRT-PAMTRA coupling. SMRT provides the
        emissivity and surface temperature for PAMTRA. PAMTRA simulates the
        airborne TB signal.

        Parameters
        ----------
        x : np.array
            State vector of the retrieval. In case model parameters are given,
            the input is xb = pd.concat((x,b)). These model parameters are then
            used to compute forward model uncertainty by the
            pyOptimalEstimation module.
        frequency : np.array
            Frequencies to simulate with PAMTRA. This includes the separate
            bandpasses of double sideband channels.
        da_avg_freq : xr.DataArray
            Frequencies that are averaged for each channel. Differs from center
            frequency for double sideband channels.
        window_frequency : xr.DataArray
            Window frequencies for SMRT simulation.
        angle : float
            Incidence angle of the radiometer.
        altitude : float
            Altitude of the aircraft.
        pressure : np.array
            Pressure profile in Pa.
        height : np.array
            Height profile in m.
        t_ut : np.array
            Temperature of upper troposphere in K.
        qv_ut : np.array
            Specific humidity of upper troposphere in g/kg.
        surface : dict
            Surface parameters that are not retrieved but fixed. These are
            provided by the user settings.
        mode : str
            Select "mean" when running this during retrieval and "gaussian"
            for random variable perturbations. In this case, the
            layer name and layer parameter need to be provided. Passed to
            SMRT simulator
        layer_name : str
            Name of the layer for which the given layer parameter will be
            perturbed. Passed to SMRT simulator
        layer_parameter : str
            Name of the parameter that will be perturbed. Passed to SMRT
            simulator.
        """

        # log the forward model call
        stack = inspect.stack()
        caller_frame = stack[1]
        logging.info(
            f"Called forward model: {caller_frame.function} line {caller_frame.lineno}"
        )

        state_dct = {x_var: x.loc[x_var].item() for x_var in self.x_vars}
        b_dct = {b_var: x.loc[b_var].item() for b_var in self.b_vars}
        logging.info(f"State vector of current simulation: {state_dct}")
        logging.info(f"Model parameters of current simulation: {b_dct}")

        # prepares state parameters as expected by forward operators
        self.full_x_dct = self.create_full_state_parameters(
            x=x.loc[self.x_vars],
            t_ut=t_ut,
            qv_ut=qv_ut,
            pressure=pressure,
            height=height,
            surface=surface,
        )

        logging.info(f"State variables prepared for forward model: {self.full_x_dct}")

        # smrt simulation
        emissivity, surf_temp_eff = self.forward_model_smrt(
            full_x_dct=self.full_x_dct,
            b_dct=b_dct,
            window_frequency=window_frequency,
            frequency=frequency,
            angle=angle,
            mode=mode,
            layer_name=layer_name,
            layer_parameter=layer_parameter,
        )
        # ensures that nadir is at first angle index
        assert emissivity[0, 0, 0] == emissivity[1, 0, 0]

        logging.info(
            f"Simulated emissivity and effective temperature with SMRT: {np.round(emissivity[:, :, 0], 2)}, {np.round(surf_temp_eff, 1)}"
        )

        # pamtra simulation
        if b_dct["specularity"] < 1:

            if angle != 0:
                logging.warning(
                    "Calling forward model with an off-nadir angle. Make sure that the correct lambertian emissivity is used."
                )

            # get emissivity at nadir for lambertian simulation
            emissivity_lamb = self.get_lamb_emissivity(emissivity)

            logging.info(
                f"Simulating PAMTRA with lambertian surface and lambertian emissivity: {np.round(emissivity_lamb, 2)}"
            )
            y_mod_lamb = self.forward_model_pamtra(
                full_x_dct=self.full_x_dct,
                frequency=frequency,
                altitude=altitude,
                height=height,
                pressure=pressure,
                emissivity=emissivity_lamb,
                surf_temp_eff=surf_temp_eff,
                surf_refl="L",
                da_avg_freq=da_avg_freq,
                angle=angle,
                polarization=polarization,
            )
        if b_dct["specularity"] > 0:
            logging.info("Simulating PAMTRA with specular surface")
            y_mod_spec = self.forward_model_pamtra(
                full_x_dct=self.full_x_dct,
                frequency=frequency,
                altitude=altitude,
                height=height,
                pressure=pressure,
                emissivity=emissivity,
                surf_temp_eff=surf_temp_eff,
                surf_refl="S",
                da_avg_freq=da_avg_freq,
                angle=angle,
                polarization=polarization,
            )
        if b_dct["specularity"] == 0:
            logging.info("Surface is fully lambertian")
            y_mod_spec = 0

        elif b_dct["specularity"] == 1:
            logging.info("Surface is fully specular")
            y_mod_lamb = 0

        y_mod = y_mod_spec * b_dct["specularity"] + y_mod_lamb * (
            1 - b_dct["specularity"]
        )

        # compare with observation if it is available
        if (self.y_obs != 0).all():
            logging.info(
                "\nFrequency | Simulation | Observation | Simulation - Observation:\n"
                + "\n".join(
                    [
                        f"{self.ds_bp.label.values[i]} | "
                        f"{y_mod[i]:.2f} K | {self.y_obs[i]:.2f} K | "
                        f"{y_mod[i] - self.y_obs[i]:.2f} K"
                        for i in range(len(y_mod))
                    ]
                )
            )
            logging.info(f"MAE [K]: {np.mean(np.abs(y_mod - self.y_obs))}")
            logging.info(f"RMSE [K]: {np.sqrt(np.mean((y_mod - self.y_obs)**2))}")

        return y_mod

    def define_perturbations(self):
        """
        Set perturbation magnitude for each variable individually. This is
        needed as snow temperature could might fall out of the realistic range.
        """

        perturbations = {}
        for v in np.concatenate([self.x_vars, self.b_vars]):
            if v in self.setting.general.perturbations:
                logging.info(
                    f"Set perturbation factor for {v} to {self.setting.general.perturbations[v]}"
                )
                perturbations[v] = self.setting.general.perturbations[v]
            else:
                logging.info(
                    f"Set perturbation factor for {v} to {self.setting.general.perturbations['other']}"
                )
                perturbations[v] = self.setting.general.perturbations["other"]

        return perturbations

    @staticmethod
    def log_std2lin_std(sigma_f, A, a=1):
        """
        Convert a logarithmic standard deviation to linear standard
        deviation.

        source: https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae

        Parameters
        ----------
        sigma_f : float
            Standard deviation of the logarithmic variable.
        A : float
            Value of the linear variable.
        a : float, optional
            Conversion factor within logarithm. The default is 1.

        Returns
        -------
        sigma_A : float
            Standard deviation of the linear variable.
        """

        sigma_A = np.abs(sigma_f * A * np.log(10) / a)

        return sigma_A

    def to_xarray(self, x, x_err, y, b, obs_time):
        """
        OEM variables to xarray dataset. State can be either a priori, a
        posteriori, iteration step, or a synthetic state.
        """

        full_x_dct = self.create_full_state_parameters(
            x=x,
            t_ut=self.forwardKwArgs["t_ut"],
            qv_ut=self.forwardKwArgs["qv_ut"],
            pressure=self.forwardKwArgs["pressure"],
            height=self.forwardKwArgs["height"],
            surface=self.forwardKwArgs["surface"],
        )
        full_x_err_dct = self.create_full_state_parameters_uncertainty(
            x_err=x_err,
            full_x_dct=full_x_dct,
            height=self.forwardKwArgs["height"],
            surface=self.forwardKwArgs["surface"],
        )

        # write all relevant retrieval information to xarray dataset
        ds = xr.Dataset()
        ds.coords["time"] = pd.Timestamp(obs_time)
        ds.coords["channel"] = self.setting.observation.channels
        ds.coords["channel1"] = self.setting.observation.channels  # for S ap
        ds.coords["channel2"] = self.setting.observation.channels  # for S ap
        ds.coords["model_level"] = np.arange(len(self.forwardKwArgs["height"]))
        ds.coords["window_frequency"] = self.forwardKwArgs["window_frequency"]
        ds.coords["polarization"] = np.array(["V", "H"])
        ds.coords["theta"] = self.setting.pamtra.angle
        ds.coords["hydro"] = self.setting.pamtra.descriptor_file_order
        ds.coords["x_vars"] = self.x_vars
        ds.coords["x_vars1"] = self.x_vars  # for averaging kernel
        ds.coords["x_vars2"] = self.x_vars  # for averaging kernel
        ds.coords["b_vars"] = self.b_vars
        ds.coords["b_vars1"] = self.b_vars  # for b cov matrix
        ds.coords["b_vars2"] = self.b_vars  # for b cov matrix
        ds.coords["xb_vars"] = np.append(self.x_vars, self.b_vars)
        ds.coords["test"] = np.array([0, 1, 2, 3])
        ds.coords["update"] = np.arange(len(self.oe.x_i) - 1)

        # retrieval information
        ds["conv"] = self.oe.converged
        ds["conv_i"] = self.oe.convI
        ds["conv_factor"] = self.oe.convergenceFactor
        ds["conv_test"] = self.oe.convergenceTest
        ds["perturbation"] = ("xb_vars", list(self.oe.perturbation.values()))
        ds["dgf"] = self.oe.dgf
        ds["dgf_x_vars"] = ("x_vars", self.oe.A_i[-1].diagonal())
        if self.oe.converged:
            self.smrt_runs = 0
            chi2test = self.oe.chiSquareTest(significance=0.05)
            ds["chi2test_result"] = ("test", chi2test[0].values)
            ds["chi2test_chi2"] = ("test", chi2test[1].values)
            ds["chi2test_chi2crit"] = ("test", chi2test[2].values)
        ds["jacobian"] = (
            ("update", "channel", "x_vars"),
            np.array(self.oe.K_i),
        )
        ds["jacobian_b"] = (("update", "channel", "b_vars"), self.oe.K_b_i)
        ds["dgf_i"] = ("update", self.oe.dgf_i)
        ds["shannon_i"] = ("update", self.oe.H_i)
        ds["averaging_kernel"] = (
            ("update", "x_vars1", "x_vars2"),
            self.oe.A_i,
        )
        ds["conv_criterion"] = ("update", self.oe.d_i2)
        ds["unc_meas_eff"] = (
            ("update", "channel1", "channel2"),
            self.oe.S_ep_i,
        )
        ds["unc_aposteriori"] = (
            ("update", "x_vars1", "x_vars2"),
            self.oe.S_aposteriori_i,
        )
        ds["unc_apriori"] = (("x_vars1", "x_vars2"), self.oe.S_a)
        ds["unc_b"] = (("b_vars1", "b_vars2"), self.oe.S_b)
        ds["unc_y"] = (("channel1", "channel2"), self.oe.S_y)

        # observation and forward simulation
        ds["y_obs"] = ("channel", self.y_obs)
        ds["y_sim"] = ("channel", y.values)

        # model parameters
        ds["pressure"] = ("model_level", self.forwardKwArgs["pressure"] * 1e-2)
        ds["height"] = ("model_level", self.forwardKwArgs["height"])
        ds["altitude"] = self.forwardKwArgs["altitude"]
        ds["angle"] = self.forwardKwArgs["angle"]

        # atmosphere state
        for v in self.setting.atmosphere.variables["thermodynamic"]:
            ds[v] = ("model_level", full_x_dct[v])
            ds[v + "_std"] = ("model_level", full_x_err_dct[v])

        # water paths and those recomputed from water content
        for hyd in self.setting.atmosphere.variables["hydro"]:
            ds[hyd] = full_x_dct[hyd]
            ds[hyd + "_std"] = full_x_err_dct[hyd].item()
            v = hyd + "_from_" + CONTENT_VARS[hyd]
            ds[v] = full_x_dct[v].item()

        # hydrometeor contents
        ds["hydro_q"] = (("model_level", "hydro"), full_x_dct["hydro_q"])

        # relative humidity and potential temperature
        ds["rh"] = (ds["t"].dims, full_x_dct["relhum"])
        ds["tpot"] = (
            ds["t"].dims,
            potential_temperature(
                ds["pressure"].values * units.hPa,
                ds["t"].values * units.K,
            ).magnitude,
        )

        # iwv
        ds["iwv"] = full_x_dct["iwv"]

        # surface parameters
        for v in self.setting.surface.variables:
            ds[v] = full_x_dct[v]
            ds[v + "_std"] = full_x_err_dct[v]

        # model parameters
        for v in self.oe.b_vars:
            ds[v] = b.loc[v].item()

        # calculate emissivity and surface temperature at window channels
        self.smrt_runs = 0
        emissivity, surf_temp_eff = self.forward_model_smrt(
            full_x_dct=full_x_dct,
            b_dct={b_var: b.loc[b_var].item() for b_var in self.b_vars},
            window_frequency=self.forwardKwArgs["window_frequency"],
            frequency=self.forwardKwArgs["window_frequency"],
            angle=self.forwardKwArgs["angle"],
            mode="mean",
            layer_name=None,
            layer_parameter=None,
        )
        ds["emissivity"] = (
            ("polarization", "window_frequency", "theta"),
            emissivity,
        )
        ds["emissivity_lamb"] = (
            "window_frequency",
            self.get_lamb_emissivity(emissivity).flatten(),
        )
        ds["surf_temp_eff"] = (("window_frequency"), surf_temp_eff)

        ds = add_variable_attributes(ds)
        ds = add_global_attributes(
            ds=ds,
            mission=self.mission,
            platform=self.platform,
            flight_id=self.flight_id,
            flight_number=self.name,
            instrument=self.setting.observation.instrument,
            vars_therm=self.setting.atmosphere.variables["thermodynamic"],
            vars_hydro=self.setting.atmosphere.variables["hydro"],
            vars_surf=self.setting.surface.variables,
            creator=self.setting.pamtra.nmlSet["creator"],
            contact=self.setting.general.contact,
        )

        return ds

    def all_to_xarray(self, obs_time):
        """
        Writes a priori, a posteriori, and iterations to xarray.
        """

        ds_a = self.to_xarray(
            x=self.oe.x_a,
            x_err=self.oe.x_a_err,
            y=self.oe.y_a,
            b=self.oe.b_p,
            obs_time=obs_time,
        )
        if self.oe.converged:
            ds_op = self.to_xarray(
                x=self.oe.x_op,
                x_err=self.oe.x_op_err,
                y=self.oe.y_op,
                b=self.oe.b_p,
                obs_time=obs_time,
            )
        else:
            ds_op = None

        # iteration states with prior error
        lst_ds = []
        for i in range(len(self.oe.x_i) - 1):
            ds = self.to_xarray(
                x=self.oe.x_i[i],
                x_err=np.sqrt(
                    pd.Series(
                        np.diag(self.oe.S_aposteriori_i[i]),
                        index=self.oe.x_vars,
                    )
                ),
                y=self.oe.y_i[i],
                b=self.oe.b_p,
                obs_time=obs_time,
            )
            lst_ds.append(ds)
        ds_i = xr.concat(
            lst_ds,
            dim="iteration",
        )
        ds_i.coords["iteration"] = np.arange(len(lst_ds))

        return ds_a, ds_op, ds_i

    def initialize(self, obs_time):
        """
        Initialize retrieval for a given observation time.
        """

        logging.info(f"Initialize retrieval for {obs_time}")

        obs_time = pd.Timestamp(obs_time)
        self.read_position_of_observation(obs_time=obs_time)
        self.create_covariance_matrix_x(obs_time=obs_time)
        self.create_a_priori_x(obs_time=obs_time)
        self.create_observation_y(obs_time=obs_time)

    def run(self):
        """
        Run retrieval.
        """

        logging.info(f"Run retrieval with these settings:")
        logging.info(f"Forward model arguments: {self.forwardKwArgs}")

        # apply scaling to the lower and upper parameter limits
        x_lower_limit_scaled = {}
        for v in self.setting.general.x_lower_limit:
            if v in self.setting.general.scales:
                x_lower_limit_scaled[v] = (
                    self.setting.general.x_lower_limit[v]
                    / self.setting.general.scales[v]
                )
            else:
                x_lower_limit_scaled[v] = self.setting.general.x_lower_limit[v]

        x_upper_limit_scaled = {}
        for v in self.setting.general.x_upper_limit:
            if v in self.setting.general.scales:
                x_upper_limit_scaled[v] = (
                    self.setting.general.x_upper_limit[v]
                    / self.setting.general.scales[v]
                )
            else:
                x_upper_limit_scaled[v] = self.setting.general.x_upper_limit[v]

        self.smrt_runs = 0  # ensures that SMRT is called in the beginning
        self.oe = pyOE.optimalEstimation(
            x_vars=self.x_vars,
            y_vars=self.y_vars,
            x_a=self.x_a,
            y_obs=self.y_obs,
            S_a=self.S_a,
            S_y=self.S_y,
            forward=self.forward_model,
            forwardKwArgs=self.forwardKwArgs,
            verbose=True,
            perturbation=self.perturbations,
            convergenceFactor=self.setting.general.convergence_factor,
            userJacobian=None,
            x_truth=None,
            b_vars=self.b_vars,
            b_p=self.b,
            S_b=self.S_b,
            x_lowerLimit=x_lower_limit_scaled,
            x_upperLimit=x_upper_limit_scaled,
            useFactorInJac=False,
            gammaFactor=self.setting.general.gamma_factor,
            disturbance=None,
            convergenceTest=self.setting.general.convergence_test,
        )
        self.oe.doRetrieval(maxIter=self.setting.general.max_iter, maxTime=1e20)


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
    args = parser.parse_args()

    main(
        setting_file=args.setting_file,
        flight_id=args.flight_id,
        obs_time=args.obs_time,
        version=args.version,
    )
