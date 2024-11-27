# -*- coding: utf-8 -*-
"""
Created By: Rowan Temple
Created Date: 16/09/2022

Utitlity functions for Perez model
"""
from pathlib import Path

import numpy as np
import pandas as pd


def eccentricity_correction(doy):
    """
    Extraterrestrial irradiance eccentricity correction coefficient
    Spencer 1971 Fourier series correction
    Relative to mean sun-earth distance
    """
    day_angle = (doy - 1) * 0.0172142  # (doy - 1) * 2pi / 365
    eti_c = (
        1.00011
        + 0.034221 * np.cos(day_angle)
        + 0.000719 * np.cos(2 * day_angle)
        + 0.000077 * np.sin(2 * day_angle)
    )
    return eti_c


def calc_air_mass(cossza):
    """
    Calculate analytical approximation of the air mass

    """
    air_mass = (1.003198 * cossza + 0.101632) / (
        cossza**2 + 0.09056 * cossza + 0.003198
    )
    return air_mass


def calc_aoi(taz1, taz2, cos=False):
    """
    Return angle of incidence between two

    Inputs can be a single vector pair or arrays of vector pairs

    :param taz1: (2-tuple or ndarray of shape (2,N)) zenith, azimuth angle(s) of
        vector 1 (radians)
    :param taz2: (2-tuple or ndarray of shape (2,N)) zenith, azimuth angle(s) of
        vector 2 (radians)
    :param cos: if True return cos(aoi). Default False

    :return: ndarray of shape (N,) aoi angles in radians or cos(aoi) values
    """

    ret = np.cos(taz1[0]) * np.cos(taz2[0]) + np.sin(taz1[0]) * np.sin(
        taz2[0]
    ) * np.cos(taz1[1] - taz2[1])

    # clip to ensure no arccos errors from floating point errors
    ret = np.clip(ret, -1, 1)

    if cos:
        return ret
    else:
        return np.arccos(ret)


def spherical_integration_elements(lebedev_degree=131):
    """
    Provide a network of elements for numerical integration over the surface
    of a sphere. Lebedev quadrature is used as one of the most
    efficient/accurate ways to integrate. quadpy offers similar schemes but
    with degree only up to 47. Lebedev is extremely accurate with even a low
    degree for smooth functions but higher orders are needed for
    discontinuous functions.

    :param lebedev_degree: (int) Currently 47 (770 integration points) and
        131 (5810 integration points) supported

    :return theta: (ndarray of shape (N,)) zenith angles of all the elements
    :return phi: (ndarray of shape (N,)) azimuth angles of all the elements
    :return weights: (ndarray of shape (N,)) solid angles for all the elements
    """
    try:
        dp = Path("__file__").parent / "data"
        df = pd.read_csv(dp / f"lebedev_{lebedev_degree:0>3}.csv")
    except FileNotFoundError:
        raise ValueError(
            f"lebedev_degree {lebedev_degree} is not a valid value. Currently"
            f"supported values are 47 (770 integration points) or 131 (5810 "
            f"integration points)"
        )
    t, p = np.radians(df["theta"].to_numpy()), np.radians(df["phi"].to_numpy())
    w = df["weights"].to_numpy()
    return t, p, w
