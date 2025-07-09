"""
Reads retrieval setting file setting.yaml into dataclass structure
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
import yaml

import si_clouds


@dataclass
class Atmosphere:
    """
    Parameters defining the atmosphere setting.

    Parameters
    ----------
    source : str
        Source of the atmospheric profile. Either "era5" or "dropsonde_era5".
    n : int
        Number of atmospheric layers.
    nr : int
        Number of retrieved atmospheric layers (from surface).
    hydro_zmin : int
        Minimum height of the hydrometeor in the atmosphere [m].
    hydro_zmax : int
        Maximum height of the hydrometeor in the atmosphere [m].
    zmaxq : int
        Maximum height of hydrometeor retrieval layers [m].
    variables: Dict[str, List[str]], optional
        Dictionary containing lists of hydrometeor and atmospheric
        thermodynamic variables.
    era5_cov_scaling : float
        Scaling factor for ERA5 covariance matrix.
    """

    source: str
    n: int
    nr: int
    hydro_zmin: int
    hydro_zmax: int
    hydro_mean: np.ndarray
    hydro_std: np.ndarray
    era5_cov_scaling: float
    variables: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Surface:
    """
    Parameters defining the surface setting.

    Parameters
    ----------
    smrt_layers_filename : str
        Name of SMRT layer definition file.
    medium_snow_covered_ice : str
        Medium for snow-covered ice. Must be in smrt_layers.yaml
    medium_young_ice : str
        Medium for young ice. Must be in smrt_layers.yaml
    source_t_as : str
        Source of the surface temperature. Either "era5" or "kt19" or "yaml".
    source_t_si : str
        Source of the surface temperature. Either "yaml" or null.
    ir_emissivity : float
        Infrared emissivity of the surface.
    t_as : float
        Air-snow interface temperature.
    t_as_std : float
        Standard deviation of the air-snow interface temperature.
    t_si : float
        Snow-ice interface temperature.
    t_si_std : float
        Standard deviation of the snow-ice interface temperature.
    yi_fraction : float
        Fraction of young ice.
    yi_fraction_std : float
        Standard deviation of the young ice fraction.
    wind_slab_volumetric_liquid_water : float
        Volumetric liquid water content of the wind slab.
    variables_r : List[str]
        Retrieved surface variables.
    variables : List[str]
        Variables for simulation with SMRT.
    channels : List[int]
        Channels for surface emissivity- or smrt-based retrieval.
    """

    smrt_layers_filename: str
    medium_snow_covered_ice: str
    medium_young_ice: str
    source_t_as: str
    source_t_si: str
    ir_emissivity: float
    tas2tsi_factor: float
    t_as: float
    t_as_std: float
    t_si: float
    t_si_std: float
    yi_fraction: float
    yi_fraction_std: float
    wind_slab_volumetric_liquid_water: float
    variables_r: List[str]
    variables: List[str]
    channels: List[int]


@dataclass
class ForwardModel:
    """
    Forward model arguments.

    Parameters
    ----------
    variables : List[str]
        Model variables considered for forward model uncertainty calculation.
    """

    model_variables: List[str]
    model_variables_mean: np.ndarray
    model_variables_std: np.ndarray


@dataclass
class Observation:
    """
    Parameters defining the observation setting.

    Parameters
    ----------
    instrument : str
        Name of the instrument.
    satellite : str
        Name of the satellite. If not a satellite, use "unspecified". Only
        important for SSMIS, because the polarization at some channels is
        different between satellites.
    altitude : float
        Altitude of the instrument. Null for aircraft, as the altitude will
        be taken from the GPS.
    channels : List[int]
        Channels to be used for retrieval.
    uncertainty : Dict[int, float]
        Dictionary containing channel-wise brightness temperature
        uncertainties.
    """

    instrument: str
    satellite: str
    altitude: float
    angle: Union[float, List[float]]
    channels: List[int]
    uncertainty: Dict[int, float]


@dataclass
class Pamtra:
    """
    Parameters related to the PAMTRA setting.

    Parameters
    ----------
    nmlSet : Dict[str, object]
        Dictionary containing settings for PAMTRA.
    descriptor_file : str
        Path to descriptor file.
    descriptor_file_order : List[str]
        Order of the hydrometeors in the descriptor file.
    """

    angle: List[float]
    nmlSet: Dict[str, object]
    descriptor_file: str
    descriptor_file_order: List[str]


@dataclass
class General:
    """
    Parameters related to the PAMTRA setting.

    Parameters
    ----------
    convergence_test : str
        Convergence test for the retrieval.
    x_lower_limit : Dict[str, float]
        Lower limits for the retrieval.
    x_upper_limit : Dict[str, float]
        Upper limits for the retrieval.
    scales : Dict[str, float]
        Scales for specific retrieval parameters. The value will be divided by
        this scale to ensure the Jacobian is well conditioned. For the output
        mean and std the value is again converted to the original scale.
    gamma_factor : List[int]
        Gamma factor for the retrieval. See the retrieval documentation for
        more information.
    convergence_factor : float
        Convergence factor for the retrieval.
    max_iter : int
        Maximum number of iterations for the retrieval.
    contact : str
        E-mail address of the contact person.
    n_processes_pamtra : int
        Number of processes for PAMTRA.
    n_processes_smrt : int
        Number of processes for SMRT.
    """

    perturbations: Dict[str, float]
    convergence_test: str
    x_lower_limit: Dict[str, float]
    x_upper_limit: Dict[str, float]
    scales: Dict[str, float]
    gamma_factor: List[int]
    convergence_factor: float
    max_iter: int
    contact: str
    n_processes_pamtra: int
    n_processes_smrt: int


@dataclass
class Setting:
    """
    Parameters defining the retrieval setting.

    Parameters
    ----------
    atmosphere : Atmosphere
        Atmosphere setting parameters.
    surface : Surface
        Surface setting parameters.
    forward_model : ForwardModel
        Forward model setting parameters.
    observation : Observation
        Observation setting parameters.
    pamtra : Pamtra
        PAMTRA setting parameters.
    general : General
        General setting parameters.
    """

    atmosphere: Atmosphere
    surface: Surface
    forward_model: ForwardModel
    observation: Observation
    pamtra: Pamtra
    general: General


def read_setting(file):
    """
    Read setting into dataclass structure.

    Parameters
    ----------
    file : str, optional
        Name of setting file.
    """

    # read yaml file
    file = os.path.join(si_clouds.__path__[0], f"retrieval/settings/{file}")
    with open(file, "r") as file:
        data = yaml.safe_load(file)

    # create instances of data classes
    setting = Setting(
        atmosphere=Atmosphere(**data["atmosphere"]),
        surface=Surface(**data["surface"]),
        forward_model=ForwardModel(**data["forward_model"]),
        observation=Observation(**data["observation"]),
        pamtra=Pamtra(**data["pamtra"]),
        general=General(**data["general"]),
    )

    return setting
