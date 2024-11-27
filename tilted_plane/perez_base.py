# -*- coding: utf-8 -*-
"""
Base functions and constants for Perez et al. models
"""

import sys
import warnings
from pathlib import Path
from typing import Union

import numpy as np

sys.path.append(str(Path("__file__").parent))
from .util import eccentricity_correction, calc_air_mass


CLEARNESS_THRESHOLDS = (1, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2)

# Solar constant
# Gueymard 2018 Revised composite extraterrestrial spectrum
# based on recent solar irradiance observations
# uncertainty estimated at +-0.5 W/m2
SOLAR_CONSTANT = 1361.1  # W/m2


def calc_clearness_index_array(epsilon: np.ndarray) -> np.ndarray:
    """
    Set model coefficient index using clearness thresholds

    :param epsilon: clearness parameter
    """

    ct = CLEARNESS_THRESHOLDS

    ci = np.zeros(epsilon.shape, dtype=int)

    if np.any(epsilon < ct[0]):
        raise ValueError("Clearness threshold should not be below 1")

    for i in range(len(ct) - 1):
        ci[np.bitwise_and(epsilon > ct[i], epsilon <= ct[i + 1])] = i

    ci[epsilon > ct[7]] = 7

    return ci


def calc_ci_delta_array(
    bh: Union[float, np.ndarray],
    dh: Union[float, np.ndarray],
    sza: Union[float, np.ndarray],
    doy: Union[float, np.ndarray],
) -> (np.ndarray, np.ndarray):
    """
    Clearness index and brightness parameters for Perez models

    All array inputs should be the same shape

    :return: clearness_index, delta. Same shape arrays as input
    """

    cossza = np.cos(sza)

    # Extra-terrestrial irradiance
    eti = SOLAR_CONSTANT * eccentricity_correction(doy)

    # Air mass
    air_mass = calc_air_mass(cossza)

    # Beam normal incidence
    bn = bh / cossza
    bn[cossza <= 0.08716] = 0  # SZA > 85 degrees

    # Brightness parameter
    delta = air_mass * dh / eti

    # Clearness parameter. We get some nan values in this calculation when
    # e.g. dh has zero values. Ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epsilon = 1 + (bn / dh) / (1 + 1.041 * sza**3)
    clearness_index = calc_clearness_index_array(epsilon)

    return clearness_index, delta
