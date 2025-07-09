"""
Surface part of the OEM retrieval using SMRT.
"""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from smrt import PSU, make_ice_column, make_model, make_snowpack, sensor_list
from smrt.core import lib
from smrt.inputs.make_medium import make_atmosphere
from smrt.permittivity.saline_snow import saline_snow_permittivity_geldsetzer09

T_WATER = 273.15 - 1.8  # freezing temperature of sea water in Kelvin
SALINITY_WATER = 34 * PSU  # salinity of water below sea ice


class SMRTSurface:
    """
    Simulates sea ice surface with SMRT model based on idealized sea ice
    profiles.
    """

    def __init__(self, mwr_frequency, angle, profiles, n_jobs=10):
        """
        Initializes SMRT surface and calculcates emissivity and emitting layer
        temperature at the given frequencues.

        The parameters used in PAMTRA will be e_m and ts_m

        Parameters
        ----------
        profiles : dict
            Dict of profiles that define a medium.
        frequency : np.array
            Frequency of the sensor in GHz.
        angle : float
            Incidence angle of the sensor in degrees.
        """

        self.mwr_frequency = mwr_frequency
        self.angle = angle
        self.profiles = profiles

        self.m = self.define_model()
        self.mwr_sensor = self.define_sensor()

        # create smrt mediums
        self.medium_lst = self.make_mediums()

        # simulate the three ice types
        self.ds = self.simulate_mediums(n_jobs=n_jobs)

    @staticmethod
    def calculate_emissivity(tb_100, tb_0):
        """
        Calculate emissivity from SMRT model from simulations under downwelling
        TB of 100 K and 0 K.

        e.g. Wivell et al. (2023)
        """

        e = 1 - (tb_100 - tb_0) / 100

        return e

    @staticmethod
    def calculate_emitting_temperature(tb, e):
        """
        Calculate emitting temperature from SMRT model from emissivity and TB.
        """

        ts = tb / e

        return ts

    def make_mediums(self):
        """
        Creates a list of mediums for SMRT from a list of setups. Each setup
        should contain the following parameters:

        - kind: snow or sea_ice
        - layer_defs: dictionary of layer properties
        - layer_order: list of layer names
        """

        # loop over all setups
        mediums = []
        for profile in self.profiles:
            medium = self.profile_from_dict(**profile["sea_ice"])
            if profile.get("snow") is not None:
                snow_profile = self.profile_from_dict(**profile["snow"])
                medium = snow_profile + medium
            mediums.append(medium)

        return mediums

    def profile_from_dict(
        self,
        kind,
        layer_defs,
        layer_order,
        microstructure_model="exponential",
        ice_type="",
    ):
        """
        Create a profile from a dictionary of layers. Each of the layers
        should be provided with their geometrical and microphysical properties
        expected by SMRT. For temperature, only the top and bottom temperature
        are required. The temperature at the layer centers is derived from
        linear interpolation. Note, if sea ice is provided as one thick layer,
        its temperature will be half of the water and sea ice top temperature.
        To avoid this and resolve temperature gradients, provide thinner
        layers of sea ice near the surface.

        Salinity is expected in PSU and will be multiplied with 0.001 before
        being passed to SMRT.

        Note: each layer also needs a temperature value!!! This was changed recently

        Example for snow. This creates a three-layer snowpack:
        kind = "snow"
        layer_defs = {
            "surface_snow": {
                "density": 94,
                "corr_length": 0.065e-3,
                "thickness": 0.062,
                "liquid_water": 0,
                "salinity": 0,
            },
            "wind_slab": {
                "density": 310,
                "corr_length": 0.092e-3,
                "thickness": 0.12,
                "liquid_water": 0,
                "salinity": 0,
            },
            "depth_hoar": {
                "density": 260,
                "corr_length": 0.32e-3,
                "thickness": 0.21,
                "liquid_water": 0,
                "salinity": 0,
            }
        }
        layer_order = ["surface_snow", "wind_slab", "depth_hoar"]

        Example for sea_ice. This creates thin bare ice:
        kind = "sea_ice"
        layer_defs = {
            "layer_1": {
                "density": 916,
                "corr_length": 0.15e-3,
                "thickness": 0.1,
                "liquid_water": 0,
                "salinity": 30,
            },
        }
        layer_order = ["layer_1"]
        microstructure_model="exponential"
        ice_type = "firstyear"

        Parameters
        ----------
        kind : str
            Either "snow" or "sea_ice".
        layers : dict
            Dictionary of smrt parameters for each of the layers. The keys of
            each layer should be the same
        layer_order : dict
            Order of the layers from top to bottom.
        """

        parameters = layer_defs[list(layer_defs)[0]].keys()
        parameters_profile = {v: np.array([]) for v in parameters}
        for param in parameters:
            for layer in layer_order:
                parameters_profile[param] = np.append(
                    parameters_profile[param], layer_defs[layer][param]
                )

        # salinity unit
        if "salinity" in parameters_profile:
            parameters_profile["salinity"] *= PSU

        if kind == "snow":

            # saline snow layer with Geldsetzer et al. (2009) model
            ice_permittivity_model = []
            for salinity in parameters_profile["salinity"]:
                if salinity > 0:
                    ice_permittivity_model.append(saline_snow_permittivity_geldsetzer09)
                else:
                    ice_permittivity_model.append(None)

            # set the temperature to 273.16 K where volumetric liquid water > 0
            if "volumetric_liquid_water" in parameters_profile:
                wet_snow_ix = parameters_profile["volumetric_liquid_water"] > 0
                if wet_snow_ix.any():
                    print(
                        f"Warning: Setting snow temperature of {wet_snow_ix.sum()} layer(s) to 273.15 K because volumetric_liquid_water > 0"
                    )
                    parameters_profile["temperature"][wet_snow_ix] = 273.15

            profile = make_snowpack(
                microstructure_model=microstructure_model,
                ice_permittivity_model=ice_permittivity_model,
                **parameters_profile,
            )

        else:

            profile = make_ice_column(
                brine_inclusion_shape="spheres",
                microstructure_model=microstructure_model,
                ice_type=ice_type,
                add_water_substrate="ocean",
                water_temperature=T_WATER,
                water_salinity=SALINITY_WATER,
                **parameters_profile,
            )

        return profile

    @staticmethod
    def define_model():
        """
        Define SMRT model.

        Note, the error handling is set to avoid this error message:
        The re-normalization of the phase function exceeds the predefined threshold of 30%.
        smrt.core.error.SMRTError: The re-normalization of the phase function exceeds the predefined threshold of 30%.
        This is likely because of a too large grain size or a bug in the phase function. It is recommended to check the grain size.
        You can also deactivate this check using normalization="forced" as an options of the dort solver. It is at last possible
        to disable this error raise and return NaN instead by adding the argument rtsolver_options=dict(error_handling='nan') to make_model)

        It remains to be checked, which simulations are affected and how to mitigate it later. During
        retrieval, this setting should be removed again.

        prune_deep_snowpack is set to 6 based on SMRT documentation. It drops
        the bottom layers of the snowpack if they are not seen by the sensor.
        """

        m = make_model(
            emmodel="iba",
            rtsolver="dort",
            rtsolver_options={
                "n_max_stream": 128,
                "error_handling": "nan",
                "prune_deep_snowpack": 6,
            },
            emmodel_options=dict(dense_snow_correction="auto"),
        )

        return m

    def define_sensor(self):
        """
        Define SMRT sensor based on the given frequency.
        """

        mwr_sensor = sensor_list.passive(self.mwr_frequency * 1e9, self.angle)

        return mwr_sensor

    def simulate_mediums(self, n_jobs):
        """
        Runs simulation and returns xarray data array with brightness
        temperature and emissivity.

        Note: To calculate the emissivity and effective temperature, two
        simulations are required. One with and one without an atmosphere.

        Returns
        -------
        da : xarray data array
            Data array with brightness temperature and emissivity.
        """

        # passive simulation - with and without atmosphere
        atmos_100 = make_atmosphere(
            "simple_isotropic_atmosphere",
            tb_down=100,
            tb_up=0,
            transmittance=1,
        )
        medium_atmos_lst = [atmos_100 + medium for medium in self.medium_lst]
        self.medium_lst.extend(medium_atmos_lst)

        print(self.medium_lst)

        res_mwr = self.m.run(
            sensor=self.mwr_sensor,
            snowpack=self.medium_lst,
            snowpack_dimension=(
                "ice_type",
                np.arange(len(self.medium_lst)),
            ),
            runner=JoblibParallelRunner(n_jobs=n_jobs),
        )
        da_tb = res_mwr.data

        # separate simulation with and without atmosphere
        da_tb_100 = da_tb.sel(
            ice_type=slice(int(len(self.medium_lst) / 2), len(self.medium_lst))
        )
        da_tb_0 = da_tb.sel(ice_type=slice(0, int(len(self.medium_lst) / 2) - 1))

        da_tb_0["ice_type"] = da_tb_0["ice_type"].values
        da_tb_100["ice_type"] = da_tb_0["ice_type"].values

        # calculate emissivity and effective temperature
        da_e = self.calculate_emissivity(tb_100=da_tb_100, tb_0=da_tb_0)
        da_ts = self.calculate_emitting_temperature(tb=da_tb_0, e=da_e)

        # combine variables into dataset
        ds = xr.Dataset(
            {
                "tb": da_tb_0,
                "e": da_e,
                "ts": da_ts,
            }
        )

        return ds

    def calculate_mixture(self, fraction):
        """
        Calculates TB, emitting layer temperature, and emissivity of the ice
        mixture. The area fractions are used to calculate a weighted average.

        fraction : np.array
            Fractions of the ice types. E.g. [0.1, 0.4, 0.5].
        """

        # add area fractions
        self.ds["fraction"] = ("ice_type", fraction)

        self.ds["tb_m"] = (self.ds["tb"] * self.ds["fraction"]).sum("ice_type")
        self.ds["ts_m"] = (self.ds["ts"] * self.ds["fraction"]).sum("ice_type")
        self.ds["e_m"] = self.ds["tb_m"] / self.ds["ts_m"]


def calculate_temperature_at_depth(t0, t1, d):
    """
    Calculates the temperature at a given depth in the ice or snow column. The
    provided temperatures are at the layer boundaries and the temperature at
    layer center is derived from linear interpolation.

    Parameters
    ----------
    t0 : float
        Temperature at the top of the column in Kelvin.
    t1 : float
        Temperature at the bottom of the column in Kelvin.
    d : float
        Thickness of each layer in meters.
    """

    if isinstance(d, (float, int)):
        d = np.array([d])

    f = interp1d([0, d.sum()], [t0, t1], kind="linear")
    t = f(d.cumsum() - d / 2)

    return t


class JoblibParallelRunner(object):
    """
    Run the simulations on the local machine on all the cores, using the
    joblib library for parallelism.

    This class is adapted from model.py in the SMRT library.
    """

    def __init__(self, n_jobs, backend="loky", max_numerical_threads=1):
        """
        Initialize joblib.

        Parameters
        ----------
        backend : str
            See joblib documentation. The default 'loky' is the recommended
            backend.
        n_jobs : int
            See joblib documentation. The default is to use all the cores.
        max_numerical_threads : int
            Maximum number of numerical threads. The default avoids mixing
            different parallelism techniques.
        """

        self.n_jobs = n_jobs
        self.backend = backend

        if max_numerical_threads > 0:
            lib.set_max_numerical_threads(max_numerical_threads)

    def __call__(self, function, argument_list):
        runner = Parallel(n_jobs=self.n_jobs, backend=self.backend)
        return runner(delayed(function)(*args) for args in argument_list)
