# -*- coding: utf-8 -*-
"""
Created By: Rowan Temple
Created Date: 16/09/2022

An implementation of the model described in Perez et al. (1990) "Modeling daylight
availability and irradiance components from direct and global irradiance"

Calculate irradiance on a tilted plane given the inputs of global horizontal and beam
horizontal irradiance components
"""

import sys
from pathlib import Path
from typing import Union

import numpy as np

sys.path.append(str(Path("__file__").parent))
from tilted_plane.perez_base import calc_ci_delta_array


# F coeffs
fcoeff_tot = np.array(
    [
        [-0.0083117, 0.5877285, -0.0620636, -0.0596012, 0.0721249, -0.0220216],
        [0.1299457, 0.6825954, -0.1513725, -0.0189325, 0.0659650, -0.0288748],
        [0.3296958, 0.4868735, -0.2210958, 0.0554140, -0.0639588, -0.0260542],
        [0.5682053, 0.1874525, -0.2951290, 0.1088631, -0.1519229, -0.0139754],
        [0.8730280, -0.3920403, -0.3616149, 0.2255647, -0.4620442, 0.0012448],
        [1.1326077, -1.2367284, -0.4118494, 0.2877813, -0.8230357, 0.0558651],
        [1.0601591, -1.5999137, -0.3589221, 0.2642124, -1.1272340, 0.1310694],
        [0.6777470, -0.3272588, -0.2504286, 0.1561313, -1.3765031, 0.2506212],
    ]
)

# F coeffs for luminance model
fcoeff_luminance = np.array(
    [
        [0.011, 0.570, -0.081, -0.095, 0.158, -0.018],
        [0.429, 0.363, -0.307, 0.050, 0.008, -0.065],
        [0.809, -0.054, -0.442, 0.181, -0.169, -0.092],
        [1.014, -0.252, -0.531, 0.275, -0.350, -0.096],
        [1.282, -0.420, -0.689, 0.380, -0.559, -0.114],
        [1.426, -0.653, -0.779, 0.425, -0.785, -0.097],
        [1.485, -1.214, -0.784, 0.411, -0.629, -0.082],
        [1.170, -0.300, -0.615, 0.518, -1.892, -0.055],
    ]
)


def perez_1990(
    bh: Union[float, np.ndarray],
    dh: Union[float, np.ndarray],
    sza: Union[float, np.ndarray],
    saa: Union[float, np.ndarray],
    pza: Union[float, np.ndarray],
    paa: Union[float, np.ndarray],
    albedo: Union[float, np.ndarray],
    doy: Union[int, float, np.ndarray],
    scenario_prod: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the global tilted irradiance value for a plane using the model described
    in Perez 1990. All input arrays should be the same shape or broadcastable to the
    same shape.

    :param bh: (array of shape (N,)) beam horizontal irradiance for target
        spectral filter
    :param dh: (array of shape (N,)) diffuse horizontal irradiance for target
        spectral filter
    :param sza: (array of shape (N,)) solar zenith angle (radians)
    :param saa: (array of shape (N,)) solar azimuth angle (radians)
    :param pza: (array of shape (M,)) plane tilt angle (radians)
    :param paa: (array of shape (M,)) plane azimuth angle (radians)
    :param albedo: (array of shape (N,)) Local ground reflectivity
    :param doy: (array of shape (N,)) Day of year from 1 (Jan 1st) to 366.
    :param scenario_prod: This is an optional switch, if True then the return
        is of shape (N, M) and is a matrix of the N solar scenarios and the M
        plane angles. If False then pza and paa inputs should be of shape (N,)
        and the return is of shape (N,) (i.e. N scenarios each with its own
        solar situation and plane angles).

    Note return shapes can be affected by scenario_prod argument
    :return gti: (array of shape (N,) or (N, M)) global tilted irradiance
    :return bti: (array of shape (N,) or (N, M)) beam tilted irradiance
    :return dti: (array of shape (N,) or (N, M)) diffuse tilted irradiance
    :return rti: (array of shape (N,) or (N, M)) reflected tilted irradiance
    """

    # precalc some trig funcs we'll need a few times
    cossza = np.cos(sza)
    cospza = np.cos(pza)
    sinpza = np.sin(pza)

    if scenario_prod:
        # broadcast inputs up to shape n, m
        n = bh.shape[0]
        m = pza.shape[0]
        bh = np.broadcast_to(bh[:, np.newaxis], (n, m))
        dh = np.broadcast_to(dh[:, np.newaxis], (n, m))
        sza = np.broadcast_to(sza[:, np.newaxis], (n, m))
        cossza = np.broadcast_to(cossza[:, np.newaxis], (n, m))
        saa = np.broadcast_to(saa[:, np.newaxis], (n, m))
        albedo = np.broadcast_to(albedo[:, np.newaxis], (n, m))
        doy = np.broadcast_to(doy[:, np.newaxis], (n, m))
        pza = np.broadcast_to(pza[np.newaxis, :], (n, m))
        paa = np.broadcast_to(paa[np.newaxis, :], (n, m))
        cospza = np.broadcast_to(cospza[np.newaxis, :], (n, m))
        sinpza = np.broadcast_to(sinpza[np.newaxis, :], (n, m))

    # beam tilted irradiance
    bt, rbeam = beam_tilt(bh, cospza, sinpza, sza, cossza, saa - paa)

    # diffuse tilted irradiance
    dt_iso, dt_circ, dt_hor = diffuse_tilt(bh, dh, sza, cospza, sinpza, rbeam, doy)

    # Ground reflected irradiance
    rt = reflected_tilt(bh + dh, albedo, cospza)

    # Global tilted irradiance
    gt = bt + dt_iso + dt_circ + dt_hor + rt

    # gt is guaranteed zero when gh is zero.
    # this is because if gh=0 implies bh=0 and dh=0
    # all components of gt are proportional to either gh, bh or dh

    return gt, bt, dt_iso + dt_circ + dt_hor, rt


def diffuse_tilt(
    bh: Union[float, np.ndarray],
    dh: Union[float, np.ndarray],
    sza: Union[float, np.ndarray],
    cospza: Union[float, np.ndarray],
    sinpza: Union[float, np.ndarray],
    rbeam: Union[float, np.ndarray],
    doy: Union[int, float, np.ndarray],
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Diffuse tilted irradiance.
    All input arrays of same shape (N,) or (N, M) or broadcastable shape
    If input array shape is (N, M) then the assumption is that N solar scenarios
    have been broadcast against M plane angles
    """
    bh, dh, sza, cospza, sinpza = np.broadcast_arrays(bh, dh, sza, cospza, sinpza)

    # this calculation only needs the N solar scenarios
    if len(dh.shape) == 2:
        ci, delta = calc_ci_delta_array(bh[:, 0], dh[:, 0], sza[:, 0], doy[:, 0])
        ci = np.broadcast_to(ci[:, np.newaxis], dh.shape)
        delta = np.broadcast_to(delta[:, np.newaxis], dh.shape)
    else:
        ci, delta = calc_ci_delta_array(bh, dh, sza, doy)

    # set the F coeffs based on the clearness thresholds
    f = fcoeff_tot
    f1 = f[:, 0][ci] + f[:, 1][ci] * delta + f[:, 2][ci] * sza
    f2 = f[:, 3][ci] + f[:, 4][ci] * delta + f[:, 5][ci] * sza

    f1[f1 < 0] = 0

    # diffuse irradiance terms
    # isotropic diffuse tilted
    dt_iso = dh * (1 - f1) * (1 + cospza) / 2

    # circumsolar diffuse tilted
    dt_circ = dh * f1 * rbeam

    # horizon brightening diffuse tilted
    dt_hor = dh * f2 * sinpza

    return dt_iso, dt_circ, dt_hor


def beam_tilt(
    bh: Union[float, np.ndarray],
    cospza: Union[float, np.ndarray],
    sinpza: Union[float, np.ndarray],
    sza: Union[float, np.ndarray],
    cossza: Union[float, np.ndarray],
    raa: Union[float, np.ndarray],
) -> (np.ndarray, np.ndarray):
    """
    Beam tilted irradiance
    """
    bh, cospza, sinpza, sza, cossza, raa = np.broadcast_arrays(
        bh, cospza, sinpza, sza, cossza, raa
    )

    # angle of incidence between surface normal and sun
    cosaoi = cospza * cossza + sinpza * np.sin(sza) * np.cos(raa)

    # Beam ratio (tilted surface/horizontal surface)
    a = np.copy(cosaoi)
    a[a < 0] = 0  # zero when surface facing away from sun
    b = np.copy(cossza)
    # denominator holds steady when sun near the horizon cos(85)
    b[b < 0.08716] = 0.08716
    rbeam = a / b  # Rbeam the ratio of cos(aoi)/cos(beta).

    return bh * rbeam, rbeam


def beam_tilt_normal(
    bh: Union[float, np.ndarray], cossza: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Beam tilted irradiance for plane facing sun (normal incidence). Slightly
    optimised form of beam tilt function in the case of normal incidence.
    """
    bh, cossza = np.broadcast_arrays(bh, cossza)
    b = np.copy(cossza)

    # denominator holds steady when sun near the horizon cos(85)
    b[b < 0.08716] = 0.08716
    return bh / b


def reflected_tilt(gh, albedo, cospza):
    """
    Ground reflected irradiance
    """
    return gh * albedo * (1 - cospza) / 2
