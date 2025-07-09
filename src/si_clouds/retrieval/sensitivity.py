"""
Perform sensitivity tests using the forward operator of the OEM retrieval.

Performs random and fixed deviation sensitivity tests for a state at a given
location. Each case an be assigned a name and the plots and results are saved
in a folder with the case name for later use. More sensitivity tests can be
added by modifying the random and astd functions.
"""

import logging
import os
from typing import Union

import cmcrameri.cm as cmc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from lizard.ac3airlib import segments_dict
from lizard.mpltools import style
from lizard.writers.figure_to_file import write_figure
from matplotlib.ticker import MultipleLocator
from xhistogram.xarray import histogram

from si_clouds.retrieval.oem import OEMRetrieval

load_dotenv()

SEED = 103

np.random.seed(SEED)

# Configure logging
logging.basicConfig(
    filename="sensitivity.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


class SensitivityTest(OEMRetrieval):
    """
    Sensitivity tests using the forward operator of the OEM retrieval.
    """

    def __init__(
        self,
        setting_file: str,
        flight_id: str,
        lon: Union[float, None] = None,
        lat: Union[float, None] = None,
        altitude: Union[float, None] = None,
        obs_time: pd.Timestamp = None,
    ):
        """
        Initialize the sensitivity test.

        Parameters
        ----------
        setting_file : str
            Path to the setting file.
        flight_id : str
            Flight ID.
        lon : float
            Longitude.
        lat : float
            Latitude.
        altitude : float
            Altitude.
        obs_time : pd.Timestamp
            Observation time.
        """

        super().__init__(
            setting_file=setting_file,
            flight_id=flight_id,
        )

        self.lon = lon
        self.lat = lat
        self.forwardKwArgs["altitude"] = altitude
        self.obs_time = obs_time

        if obs_time is not None:
            self.read_position_of_observation(obs_time=obs_time)
        self.create_covariance_matrix_x(obs_time=obs_time)
        self.create_a_priori_x(obs_time=obs_time)

        # parameter uncertainty
        self.x_std = self.get_standard_deviation()

        # save the initial mean state
        self.x_a_initial = self.x_a.copy()

        # simulation outputs
        self.x_sim = []
        self.y_sim = []

        # integrated quantities
        self.iwv = []
        self.ihyd = []

        # perturbations
        self.std_factors = []
        self.variables = []

        # dataset
        self.ds = xr.Dataset()

    def get_standard_deviation(self):
        """
        Create random states for sensitivity tests from covariance matrix.
        """

        x_std = np.sqrt(np.diag(self.S_a))

        return x_std

    def modify_state_randomly(self, variables: list):
        """
        Add Gaussian noise based on parameter standard deviation to the mean
        state.

        Parameters
        ----------
        variables : list
            List of variables to be modified.
        """

        for v in variables:
            self.modify_state_astd(std_factor=np.random.normal(0, 1), v=v)

    def modify_state_astd(self, std_factor: float, v: str):
        """
        Modify state by a multiple of the standard deviation.

        Note that all height levels are changed for t, q, and hydrometeors by
        the same random factor.

        Parameters
        ----------
        std_factor : float
            Multiple of the standard deviation.
        v : str
            Variable to be modified.
        """

        logging.info(f"Modify state for variable {v} by factor {std_factor}")

        self.x_a[self.x_ix[v]] += std_factor * self.x_std[self.x_ix[v]]

    def reset_state(self):
        """
        Reset the state to the initial state.
        """

        self.x_a = self.x_a_initial.copy()

    def simulate(self):
        """
        Run forward operator.
        """

        y = self.forward_model(x=self.x_a, **self.forwardKwArgs)

        # store the integrated quantities that are calculated
        self.iwv.append(self.full_x_dct["iwv"])
        self.ihyd.append(
            [
                self.full_x_dct[q_hyd + "_int"]
                for q_hyd in self.setting.atmosphere.variables["hydro"]
            ]
        )

        # store simulation output
        self.x_sim.append(self.x_a)
        self.y_sim.append(y)

    def run_random(self, n_random: int, variables: list):
        """
        Runs with random perturbations
        """

        for i in range(n_random):
            logging.info(f"Iteration {i+1}/{n_random}")
            self.modify_state_randomly(variables)
            self.simulate()
            self.reset_state()

    def run_astd(self, std_factors: list, variables: list):
        """
        Runs with random perturbations
        """

        n = len(std_factors) * len(variables)
        i = 1

        for variable in variables:
            for std_factor in std_factors:
                logging.info(f"Iteration {i}/{n}")
                self.modify_state_astd(std_factor, variable)
                self.simulate()
                self.reset_state()

                self.std_factors.append(std_factor)
                self.variables.append(variable)

                i += 1

    def run(
        self,
        name: str,
        variables: list,
        n_random: int = 0,
        std_factors: list = [],
    ):
        """
        Run sensitivity tests and save result as netCDF file.

        Two options of sensitivity tests are available:
        1. Random perturbations of all or a set of state parameters.
        2. Perturbations of single state parameters by multiples of the
           standard deviation.

        Parameters
        ----------
        name : str
            Name of the sensitivity test.
        variables : list
            This parameter has two meanings. If n_random is provided, each
            of these variables is perturbed randomly at the same time. If
            std_factors is provided, each of these variables is perturbed
            individually one after another by the given factor.
        n_random : int
            Number of random perturbations of all the variables.
        std_factors : list
            List of factors to multiply the standard deviation with.
        """

        # empty output variables in case of multiple runs
        self.x_sim = []
        self.y_sim = []
        self.iwv = []
        self.ihyd = []
        self.std_factors = []
        self.variables = []
        self.ds = xr.Dataset()

        self.simulate()  # initial state

        if n_random > 0:
            self.run_random(n_random, variables)
        elif len(std_factors) > 0:
            self.run_astd(std_factors, variables)

        self.to_xarray()
        self.statistics()
        self.save_output(name=name)

    def to_xarray(self):
        """
        Write result to xarray dataset
        """

        self.ds = xr.Dataset(
            {
                "x0_sim": (["x_vars"], self.x_sim[0]),
                "y0_sim": (["channel"], self.y_sim[0]),
                "x_sim": (["sensitivity", "x_vars"], np.array(self.x_sim)[1:]),
                "y_sim": (
                    ["sensitivity", "channel"],
                    np.array(self.y_sim)[1:],
                ),
                "iwv0": self.iwv[0],
                "ihyd0": ("hydro", self.ihyd[0]),
                "iwv": (["sensitivity"], np.array(self.iwv)[1:]),
                "ihyd": (["sensitivity", "hydro"], np.array(self.ihyd)[1:, :]),
            },
            coords={
                "sensitivity": np.arange(0, len(self.x_sim) - 1),
                "x_vars": self.x_vars,
                "channel": self.setting.observation.channels,
                "hydro": self.setting.atmosphere.variables["hydro"],
            },
        )

        # add information on std factors and variables for each simulation
        if self.std_factors:
            self.ds["std_factors"] = ("sensitivity", self.std_factors)
            self.ds["variables"] = ("sensitivity", self.variables)

        # add meta information
        self.ds.attrs["flight_id"] = self.flight_id
        self.ds.attrs["lon"] = self.lon
        self.ds.attrs["lat"] = self.lat
        self.ds.attrs["time"] = str(self.obs_time)
        self.ds.attrs["altitude"] = self.forwardKwArgs["altitude"]
        self.ds.attrs["created"] = str(np.datetime64("now"))

    def statistics(self):
        """
        Calculate statistics for the sensitivity tests.
        """

        # calculate bias and noise
        self.ds["y_bias"] = self.ds["y_sim"] - self.ds["y0_sim"]
        self.ds["x_noise"] = self.ds["x_sim"] - self.ds["x0_sim"]

        # calculate rmsd for each channel
        self.ds["channel_rmsd"] = np.sqrt(
            ((self.ds["y_sim"] - self.ds["y0_sim"]) ** 2).mean("sensitivity")
        )

        # calculate rmsd for each sensitivity test
        self.ds["sensitivity_rmsd"] = np.sqrt(
            ((self.ds["y_sim"] - self.ds["y0_sim"]) ** 2).mean("channel")
        )

        # calculate mae for each channel
        self.ds["channel_mae"] = np.abs(self.ds["y_sim"] - self.ds["y0_sim"]).mean(
            "sensitivity"
        )

        # calculate mae for each sensitivity test
        self.ds["sensitivity_mae"] = np.abs(self.ds["y_sim"] - self.ds["y0_sim"]).mean(
            "channel"
        )

    def save_output(self, name: str):
        """
        Save output to file.

        Parameters
        ----------
        name : str
            Name of the sensitivity test.
        """

        self.ds.to_netcdf(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_clouds",
                f"sensitivity_{name}.nc",
            )
        )


def main(run_simulation: bool = False):
    """
    Perform sensitivity tests using the forward operator of the OEM retrieval.
    """

    cases_dct = {
        "HALO-AC3_HALO_RF04_hl07-snowscat": {
            "flight_id": "HALO-AC3_HALO_RF04",
            "obs_time": pd.Timestamp("2022-03-14 13:10"),
            "lon": None,
            "lat": None,
            "altitude": None,
            "segment_name": "HALO-AC3_HALO_RF04_hl07",
        },
        # "HALO-AC3_HALO_RF04_hl07-dry": {
        #    "flight_id": "HALO-AC3_HALO_RF04",
        #    "obs_time": pd.Timestamp("2022-03-14 12:40"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF04_hl07",
        # },
        # "HALO-AC3_HALO_RF04_hl07-humid": {
        #    "flight_id": "HALO-AC3_HALO_RF04",
        #    "obs_time": pd.Timestamp("2022-03-14 12:55"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF04_hl07",
        # },
        # "HALO-AC3_HALO_RF04_wai-transect-fog": {
        #    "flight_id": "HALO-AC3_HALO_RF04",
        #    "obs_time": pd.Timestamp("2022-03-14 12:25"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF04_hl06",
        # },
        # "HALO-AC3_HALO_RF04_wai-transect-low": {
        #    "flight_id": "HALO-AC3_HALO_RF04",
        #    "obs_time": pd.Timestamp("2022-03-14 12:20"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF04_hl06",
        # },
        # "HALO-AC3_HALO_RF04_wai-transect-mid": {
        #    "flight_id": "HALO-AC3_HALO_RF04",
        #    "obs_time": pd.Timestamp("2022-03-14 12:15"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF04_hl06",
        # },
        # "HALO-AC3_HALO_RF04_wai-transect-cen": {
        #    "flight_id": "HALO-AC3_HALO_RF04",
        #    "obs_time": pd.Timestamp("2022-03-14 12:00"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF04_hl06",
        # },
        # "HALO-AC3_HALO_RF10_clear-sky-circle": {
        #    "flight_id": "HALO-AC3_HALO_RF10",
        #    "obs_time": pd.Timestamp("2022-03-29 14:00"),
        #    "lon": None,
        #    "lat": None,
        #    "altitude": None,
        #    "segment_name": "HALO-AC3_HALO_RF10_ci01",
        # },
    }

    test_names = [
        "astd_all",
        # "random_surface",
        # "random_t",
        # "random_qv",
        # "random_all",
        # "random_tqv",
        # "random_hydro",
    ]

    n_random = 100
    std_factors = np.arange(0, 3.01, 0.2)

    if run_simulation:
        logging.info(f"Run the following sensitivity tests: {test_names}")
    else:
        logging.info(f"Read the following sensitivity tests: {test_names}")

    for case_name, params in cases_dct.items():

        logging.info(
            f"Run sensitivity tests for case {case_name} with parameters: {params}"
        )

        sen = SensitivityTest(
            setting_file="setting_sensitivity.yaml",
            flight_id=params["flight_id"],
            lon=params["lon"],
            lat=params["lat"],
            altitude=params["altitude"],
            obs_time=params["obs_time"],
        )

        run_case(
            run_simulation=run_simulation,
            sen=sen,
            case_name=case_name,
            test_names=test_names,
            segment_name=params["segment_name"],
            n_random=n_random,
            std_factors=std_factors,
        )


def run_case(
    run_simulation,
    sen,
    case_name,
    test_names,
    segment_name,
    n_random,
    std_factors,
):
    """
    Run simulation or plots for a given case.

    Parameters
    ----------
    run_simulation : bool
        If True, run sensitivity tests.
    sen : SensitivityTest
        Sensitivity test object.
    case_name : str
        Name of the case.
    test_names : list
        List of sensitivity tests to be simulated or plotted. To add a new
        sensitivity test, modify either the random or astd functions.
    segment_name : str
        Name of the segment. This is used to compare sensitivity test with
        observations during this flight segment.
    """

    if run_simulation:
        ds_dct = {}
        ds_dct.update(
            random_simulations(
                sen=sen,
                n_random=n_random,
                case_name=case_name,
                test_names=test_names,
            )
        )
        ds_dct.update(
            astd_simulations(
                sen=sen,
                std_factors=std_factors,
                case_name=case_name,
                test_names=test_names,
            )
        )

    else:
        ds_dct = read_sensitivity_tests(test_names=test_names, case_name=case_name)

    # plot the observed vs. simulated TB
    if "random_surface" in test_names:
        plot_observed_vs_simulated(
            sen=sen,
            ds_obs=sen.ds_obs,
            ds_sim=ds_dct["random_surface"],
            segment_name=segment_name,
            case_name=case_name,
        )

    # plot the random state perturbations
    for test_name in test_names:
        line_plot_tb(
            sen,
            ds_dct[test_name],
            x_axis="frequency",
            test_name=test_name,
            case_name=case_name,
        )
        line_plot_tb(
            sen,
            ds_dct[test_name],
            x_axis="channel",
            test_name=test_name,
            case_name=case_name,
        )
        boxplot_tb(sen, ds_dct[test_name], test_name=test_name, case_name=case_name)
        boxplot_tb_bias(
            sen, ds_dct[test_name], test_name=test_name, case_name=case_name
        )

        if test_name == "astd_all":
            line_plot_tb_bias_hydro(sen, ds=ds_dct["astd_all"], case_name=case_name)
            line_plot_tb_bias_hydro_cwc_swc(
                sen, ds=ds_dct["astd_all"], case_name=case_name
            )


def read_sensitivity_tests(test_names: list, case_name: str):
    """
    Reads a list of sensitivity tests
    """

    ds_dct = {}

    for test_name in test_names:
        ds_dct[test_name] = xr.open_dataset(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_clouds",
                f"sensitivity_{case_name}_{test_name}.nc",
            )
        )

    return ds_dct


def random_simulations(sen, n_random, case_name, test_names):
    """
    Run forward simulator for a set of random perturbations of the state
    parameters.
    """

    variables_dct = {
        "random_all": (
            sen.setting.atmosphere.variables["thermodynamic"]
            + sen.setting.atmosphere.variables["hydro"]
            + sen.setting.surface.variables
        ),
        "random_tqv": sen.setting.atmosphere.variables["thermodynamic"],
        "random_t": ["t"],
        "random_qv": ["qv"],
        "random_hydro": sen.setting.atmosphere.variables["hydro"],
        "random_surface": sen.setting.surface.variables,
    }

    dct = {}

    for test_name, variables in variables_dct.items():
        if test_name in test_names:
            sen.run(
                name=f"{case_name}_{test_name}",
                variables=variables,
                n_random=n_random,
            )
            dct[test_name] = sen.ds

    return dct


def astd_simulations(sen, std_factors, case_name, test_names):
    """
    Run forward simulator for a set of random perturbations of the state
    parameters.
    """

    dct = {}

    if "astd_all" in test_names:

        sen.run(
            name=f"{case_name}_astd_all",
            variables=(
                sen.setting.atmosphere.variables["thermodynamic"]
                + sen.setting.atmosphere.variables["hydro"]
                + sen.setting.surface.variables
            ),
            std_factors=std_factors,
        )
        dct["astd_all"] = sen.ds

    return dct


def line_plot_tb(sen, ds: xr.Dataset, x_axis: str, test_name: str, case_name: str):
    """
    Line plot of TB for each channel. This requires a baseline simulation
    and several simulations with Gaussian perturbed states. The different
    simulations are not separated.
    """

    if x_axis == "channel":
        x = sen.setting.observation.channels

    elif x_axis == "frequency":
        x = sen.ds_bp.avg_freq.max("n_avg_freq")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for s in ds.sensitivity:
        ax.plot(x, ds["y_sim"].sel(sensitivity=s))
    ax.plot(x, ds["y0_sim"], color="black")

    if x_axis == "channel":
        ax.set_xticks(x)
        ax.set_xticklabels(sen.ds_bp.label.values, rotation=90)

    ax.set_ylabel("TB [K]")

    write_figure(
        fig,
        f"sensitivity/sensitivity_{case_name}_{test_name}_lineplot_tb_{x_axis}.png",
    )

    plt.close()


def boxplot_tb(sen, ds: xr.Dataset, test_name: str, case_name: str):
    """
    Boxplot of TB for each channel. This requires a baseline simulation
    and several simulations with Gaussian perturbed states. The different
    simulations are not separated.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.scatter(sen.setting.observation.channels, ds["y0_sim"], color="black")
    ax.boxplot(ds["y_sim"])

    ax.set_xticks(sen.setting.observation.channels)
    ax.set_xticklabels(sen.ds_bp.label.values, rotation=90)

    ax.set_ylabel("TB [K]")

    write_figure(fig, f"sensitivity/sensitivity_{case_name}_{test_name}_boxplot_tb.png")

    plt.close()


def boxplot_tb_bias(sen, ds: xr.Dataset, test_name: str, case_name: str):
    """
    Boxplot of TB bias for each channel. This requires a baseline simulation
    and several simulations with Gaussian perturbed states. The different
    simulations are not separated.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.axhline(0, color="black")
    ax.boxplot(ds["y_bias"])

    ax.set_xticks(sen.setting.observation.channels)
    ax.set_xticklabels(sen.ds_bp.label.values, rotation=90)

    ax.set_ylabel("TB bias [K]")

    ax.yaxis.set_minor_locator(MultipleLocator(10))

    write_figure(
        fig,
        f"sensitivity/sensitivity_{case_name}_{test_name}_boxplot_tb_bias.png",
    )

    plt.close()


def line_plot_tb_bias_hydro(sen, ds: xr.Dataset, case_name: str):
    """
    Compare the TB bias for different hydrometeor classes in one subplot.
    """

    # compare the four hydrometeor classes
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True, sharex=True)

    std_factors = np.sort(np.unique(ds["std_factors"].values))

    # plot logarithmic colorbar from 0 to 1000
    cmap = cmc.batlow
    norm = mcolors.BoundaryNorm(np.arange(0, 1001, 50), ncolors=cmap.N)
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label="Int. hyd. cont. [g m$^{-2}$]",
    )

    for i_hyd, hydro_name in enumerate(ds["hydro"].values):
        ax = axes.flat[i_hyd]
        ax.annotate(
            hydro_name,
            xy=(0.5, 1),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
        )
        ax.axhline(0, color="black")
        for astd in std_factors:

            ihyd = (
                ds["ihyd"]
                .sel(
                    sensitivity=(ds["variables"] == hydro_name)
                    & (ds["std_factors"] == astd),
                    hydro=hydro_name,
                )
                .item()
            )
            ihyd = int(round(ihyd * 1e3, 0))  # kg m-2 to g m-2

            ax.plot(
                ds.channel,
                ds["y_bias"]
                .sel(
                    sensitivity=(ds["variables"] == hydro_name)
                    & (ds["std_factors"] == astd)
                )
                .squeeze(),
                color=cmap(norm(ihyd)),
            )

        ax.set_xticks(ds.channel.values)

        ax.legend(frameon=False)

    for ax in axes[-1, :]:
        ax.set_xticklabels(sen.ds_bp.label.values, rotation=90, fontsize=7)

    for ax in axes[:, 0]:
        ax.set_ylabel("$T_b$ bias [K]")

    write_figure(
        fig,
        f"sensitivity/sensitivity_{case_name}_astd_hyd_lineplot_tb_bias.png",
    )

    plt.close()


def line_plot_tb_bias_hydro_cwc_swc(sen, ds: xr.Dataset, case_name: str):
    """
    Compare the TB bias for different hydrometeor classes in one subplot.
    """

    # compare the four hydrometeor classes
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    std_factors = np.sort(np.unique(ds["std_factors"].values))

    labels = {
        "cwc_q": "Cloud liquid water [g m$^{-2}$]",
        "swc_q": "Snow water [g m$^{-2}$]",
    }

    # plot logarithmic colorbar from 0 to 500
    cmap = cmc.batlow
    norm = mcolors.BoundaryNorm(np.arange(0, 501, 50), ncolors=cmap.N)

    for i_hyd, hydro_name in enumerate(labels.keys()):
        ax = axes.flat[i_hyd]

        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label=labels[hydro_name],
        )
        ax.axhline(0, color="black")
        for astd in std_factors:

            ihyd = (
                ds["ihyd"]
                .sel(
                    sensitivity=(ds["variables"] == hydro_name)
                    & (ds["std_factors"] == astd),
                    hydro=hydro_name,
                )
                .item()
            )
            ihyd = int(round(ihyd * 1e3, 0))  # kg m-2 to g m-2

            ax.plot(
                ds.channel,
                ds["y_bias"]
                .sel(
                    sensitivity=(ds["variables"] == hydro_name)
                    & (ds["std_factors"] == astd)
                )
                .squeeze(),
                color=cmap(norm(ihyd)),
            )

        ax.set_xticks(ds.channel.values)

        ax.grid()

        ax.yaxis.set_minor_locator(mticker.MultipleLocator(2))

        ax.legend(frameon=False)

    axes[-1].set_xticklabels(sen.ds_bp.label.values, rotation=90)

    for ax in axes:
        ax.set_ylabel("$T_b$ bias [K]")

    write_figure(
        fig,
        f"sensitivity/sensitivity_{case_name}_astd_hyd_cwc_swc_lineplot_tb_bias.png",
    )

    plt.close()


def observed_variability(ds):
    """
    Calculates histogram of observed TB variability

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with observed TB values.
    """

    # calculate Tb histogram for each channel
    da_hist = histogram(
        ds["tb"],
        bins=np.arange(180, 271, 1),
        dim=["time"],
    )
    da_hist = da_hist.where(da_hist > 0)

    return da_hist


def plot_observed_vs_simulated(sen, ds_obs, ds_sim, segment_name, case_name):
    """
    Boxplot of simulated TB variability on top of observed TB variability.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Dataset with observed TB values.
    ds_sim : xr.Dataset
        Dataset with simulated TB values.
    """

    segments = segments_dict(sen.flight_id)
    segment = segments[segment_name]

    ds_obs = ds_obs.sel(time=slice(segment["start"], segment["end"]))

    da_tbobs_hist = observed_variability(ds_obs)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    im = ax.pcolormesh(
        da_tbobs_hist["channel"],
        da_tbobs_hist["tb_bin"],
        da_tbobs_hist.T,
        shading="nearest",
        cmap=cmc.grayC,
        norm=mcolors.LogNorm(vmin=1, vmax=da_tbobs_hist.max()),
    )
    fig.colorbar(im, label="Count $y_{obs}$")

    # simulation as boxplot
    ax.scatter(
        ds_sim.channel,
        ds_sim["y0_sim"],
        color=cmc.batlow(0.6),
        label="$F(x_a)$",
    )
    c = cmc.batlow(0.8)
    ax.boxplot(
        ds_sim["y_sim"],
        boxprops=dict(color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color=c),
    )

    ax.legend(frameon=False)

    ax.set_ylabel("$T_b$ [K]")

    ax.set_xticks(ds_obs.channel.values)
    ax.set_xticklabels(sen.ds_bp.label.values, rotation=90)

    ax.set_ylim(180, 270)

    write_figure(
        fig,
        f"sensitivity/sensitivity_tb_sim_vs_obs_{case_name}_{segment_name}.png",
    )

    plt.close()


if __name__ == "__main__":
    main(run_simulation=False)
