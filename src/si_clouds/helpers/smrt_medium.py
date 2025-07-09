"""
Makes SMRT medium.
"""

import random

import numpy as np
from dotenv import load_dotenv
from scipy.stats import truncnorm

from si_clouds.io.readers.smrt_layers import read_smrt_layers
from si_clouds.retrieval.surface import calculate_temperature_at_depth

load_dotenv()


def get_mean_value(value_lst):
    return value_lst[0]


def get_gaussian_value(value_lst):
    x_mean, x_std = value_lst
    return truncnorm.rvs(a=-2, b=2, loc=x_mean, scale=x_std)


def get_uniform_value(value_lst):
    x_mean, x_std = value_lst
    return random.uniform(x_mean - 2 * x_std, x_mean + 2 * x_std)


def make_medium(filename, medium, mode, overwrite=None, layer_name=None, layer_parameter=None):
    """
    Generate a random realization of a medium, which consists of sea ice and
    maybe also a snow cover for sensitivity tests.

    Parameters
    ----------
    medium : str
        The medium to generate a profile for. E.g. "yi", "fyi_two_layer_snow",
        "myi_three_layer_snow". See available mediums in smrt_layers.yaml
    mode : str
        The mode to generate the medium. Either "mean", "gaussian", or "uniform".
        In "mean", only the mean values of each parameter are used. In "gaussian",
        the mean from al but one parameters are used. For this one parameter,
        a random value is drawn using mean and standard deviation and clipped
        with the min and max ranges. For uniform, uniformly distributed values
        are drawn for all parameters.
    overwrite : dict
        Replaces the mean value by the value in the overwrite dictionary. This
        allows to set specific layer parameters for retrieval. Default is None.
    layer_name : str
        Only for "gaussian" mode. Vary this snow or ice layer only, see
        layer_parameter variable
    layer_parameter : str
        Only for "gaussian" mode. Vary this parameter of the layer specified
        in layer_name. Default is None.

    Returns
    -------
    profile : dict
        A dictionary with the profile as required by profile_from_dict().
    """

    smrt_layers = read_smrt_layers(filename)

    mode_to_function = {
        "mean": get_mean_value,
        "gaussian": get_gaussian_value,
        "uniform": get_uniform_value,
    }
    f = mode_to_function[mode]

    # replace value or value ranges with user input
    if overwrite is not None:
        if "mediums" in overwrite.keys():
            for key in overwrite["mediums"]:
                for prop, value in overwrite["mediums"][key].items():
                    smrt_layers["mediums"][key][prop] = value
        if "base_layers" in overwrite:
            for substrate in overwrite["base_layers"]:
                for layer in overwrite["base_layers"][substrate]:
                    for prop, value in overwrite["base_layers"][substrate][
                        layer
                    ].items():
                        smrt_layers["base_layers"][substrate][layer][prop] = value

    # replace all but one parameter with mean value
    if mode == "gaussian" and layer_name is not None and layer_parameter is not None:
        for substrate in ["snow", "sea_ice"]:
            for name in smrt_layers["base_layers"][substrate]:
                for param, value in smrt_layers["base_layers"][substrate][name].items():
                    if name != layer_name or param != layer_parameter:
                        if isinstance(value, list):
                            smrt_layers["base_layers"][substrate][name][param] = (
                                get_mean_value(value)
                            )

        # keep temperature at mean value always
        for key in smrt_layers["mediums"]:
            for param, value in smrt_layers["mediums"][key].items():
                if isinstance(value, list) and param in [
                    "temperature_as",
                    "temperature_si",
                ]:
                    smrt_layers["mediums"][key][param] = get_mean_value(value)

    profile = {}
    empty_profile = {
        "kind": "",
        "layer_defs": {},
        "layer_order": [],
        "microstructure_model": "exponential",
        "ice_type": "",
    }

    for substrate in ["snow", "sea_ice"]:
        substrate_type = smrt_layers["mediums"][medium][substrate]
        if substrate_type is not None:
            profile_def = smrt_layers["profiles"][substrate][substrate_type]
            profile[substrate] = empty_profile.copy()

            # fill with values that are constant for the substrate
            profile[substrate]["kind"] = substrate
            for prop, value in profile_def.items():
                if isinstance(value, list) and prop not in ["layer_order"]:
                    profile[substrate][prop] = f(value)
                else:
                    profile[substrate][prop] = value

            # fill with values for each layer of the substrate (e.g. density)
            profile[substrate]["layer_defs"] = {}
            for layer in profile_def["layer_order"]:
                profile[substrate]["layer_defs"][layer] = {}

                for prop, value in smrt_layers["base_layers"][substrate][layer].items():
                    if isinstance(value, list):
                        profile[substrate]["layer_defs"][layer][prop] = f(value)
                    else:
                        profile[substrate]["layer_defs"][layer][prop] = value

    # sample temperature for snow and sea ice interfaces
    medium_temp = {}
    for prop, value in smrt_layers["mediums"][medium].items():
        if isinstance(value, list):
            medium_temp[prop] = f(value)
        else:
            medium_temp[prop] = value

    # assign temperature to each layer
    for substrate in profile.keys():
        if substrate == "snow":
            t0, t1 = (
                medium_temp["temperature_as"],
                medium_temp["temperature_si"],
            )

        elif substrate == "sea_ice" and medium != "yi":
            t0, t1 = (
                medium_temp["temperature_si"],
                medium_temp["temperature_iw"],
            )

        elif substrate == "sea_ice" and medium == "yi":
            t0, t1 = (
                medium_temp["temperature_as"],
                medium_temp["temperature_iw"],
            )

        else:
            raise ValueError("Unknown substrate")

        # interpolate temperature and assign values to each layer
        d = [
            profile[substrate]["layer_defs"][layer]["thickness"]
            for layer in list(profile[substrate]["layer_defs"])
        ]
        t = calculate_temperature_at_depth(t0, t1, np.array(d))
        for i, layer in enumerate(profile[substrate]["layer_order"]):
            profile[substrate]["layer_defs"][layer]["temperature"] = t[i].item()

    return profile
