# -*- coding: utf-8 -*-
"""
Code for sky radiance distribution model described in
Perez et al. (1993) All-weather model for sky luminance distribution - preliminary
configuration and validation
"""
import sys
import warnings
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np

sys.path.append(str(Path("__file__").parent))
from tilted_plane.perez_base import calc_ci_delta_array
from tilted_plane.util import calc_aoi, spherical_integration_elements

rdcoeff = np.array(
    [
        [
            [1.35250e00, -2.57600e-01, -2.69000e-01, -1.43660e00],
            [-1.22190e00, -7.73000e-01, 1.41480e00, 1.10160e00],
            [-1.10000e00, -2.51500e-01, 8.95200e-01, 1.56000e-02],
            [-5.48400e-01, -6.65400e-01, -2.67200e-01, 7.11700e-01],
            [-6.00000e-01, -3.56600e-01, -2.50000e00, 2.32500e00],
            [-1.01560e00, -3.67000e-01, 1.00780e00, 1.40510e00],
            [-1.00000e00, 2.11000e-02, 5.02500e-01, -5.11900e-01],
            [-1.05000e00, 2.89000e-02, 4.26000e-01, 3.59000e-01],
        ],
        [
            [-7.67000e-01, 7.00000e-04, 1.27340e00, -1.23300e-01],
            [-2.05400e-01, 3.67000e-02, -3.91280e00, 9.15600e-01],
            [2.78200e-01, -1.81200e-01, -4.50000e00, 1.17660e00],
            [7.23400e-01, -6.21900e-01, -5.68120e00, 2.62970e00],
            [2.93700e-01, 4.96000e-02, -5.68120e00, 1.84150e00],
            [2.87500e-01, -5.32800e-01, -3.85000e00, 3.37500e00],
            [-3.00000e-01, 1.92200e-01, 7.02300e-01, -1.63170e00],
            [-3.25000e-01, 1.15600e-01, 7.78100e-01, 2.50000e-03],
        ],
        [
            [2.80000e00, 6.00400e-01, 1.23750e00, 1.00000e00],
            [6.97500e00, 1.77400e-01, 6.44770e00, -1.23900e-01],
            [2.47219e01, -1.30812e01, -3.77000e01, 3.48438e01],
            [3.33389e01, -1.83000e01, -6.22500e01, 5.20781e01],
            [2.10000e01, -4.76560e00, -2.15906e01, 7.24920e00],
            [1.40000e01, -9.99900e-01, -7.14060e00, 7.54690e00],
            [1.90000e01, -5.00000e00, 1.24380e00, -1.90940e00],
            [3.10625e01, -1.45000e01, -4.61148e01, 5.53750e01],
        ],
        [
            [1.87340e00, 6.29700e-01, 9.73800e-01, 2.80900e-01],
            [-1.57980e00, -5.08100e-01, -1.78120e00, 1.08000e-01],
            [-5.00000e00, 1.52180e00, 3.92290e00, -2.62040e00],
            [-3.50000e00, 1.60000e-03, 1.14770e00, 1.06200e-01],
            [-3.50000e00, -1.55400e-01, 1.40620e00, 3.98800e-01],
            [-3.40000e00, -1.07800e-01, -1.07500e00, 1.57020e00],
            [-4.00000e00, 2.50000e-02, 3.84400e-01, 2.65600e-01],
            [-7.23120e00, 4.05000e-01, 1.33500e01, 6.23400e-01],
        ],
        [
            [3.56000e-02, -1.24600e-01, -5.71800e-01, 9.93800e-01],
            [2.62400e-01, 6.72000e-02, -2.19000e-01, -4.28500e-01],
            [-1.56000e-02, 1.59700e-01, 4.19900e-01, -5.56200e-01],
            [4.65900e-01, -3.29600e-01, -8.76000e-02, -3.29000e-02],
            [3.20000e-03, 7.66000e-02, -6.56000e-02, -1.29400e-01],
            [-6.72000e-02, 4.01600e-01, 3.01700e-01, -4.84400e-01],
            [1.04680e00, -3.78800e-01, -2.45170e00, 1.46560e00],
            [1.50000e00, -6.42600e-01, 1.85640e00, 5.63600e-01],
        ],
    ]
)


def get_coeff(ci: np.ndarray, delta: np.ndarray, sza: np.ndarray) -> np.ndarray:
    """
    Get the 5 model coefficients according to the clearness index

    :param ci: (ndarray of shape (N,)) clearness index
    :param delta: (ndarray of shape (N,)) delta parameter
    :param sza: (ndarray of shape (N,)) solar zenith angles

    :return coeff: (ndarray of shape (5, N))
    """

    # add some reshapes so that broadcasting rules force coeff to be of
    # shape (5, N)
    coeff = (
        rdcoeff[:, ci, 0]
        + rdcoeff[:, ci, 1] * sza
        + delta * (rdcoeff[:, ci, 2] + rdcoeff[:, ci, 3] * sza)
    )

    coeff[2, ci == 0] = (
        np.exp(
            delta[ci == 0]
            * (rdcoeff[2, ci[ci == 0], 0] + rdcoeff[2, ci[ci == 0], 1] * sza[ci == 0])
            ** rdcoeff[2, ci[ci == 0], 2]
        )
        - rdcoeff[2, ci[ci == 0], 3]
    )
    coeff[3, ci == 0] = (
        np.exp(
            delta[ci == 0]
            * (rdcoeff[3, ci[ci == 0], 0] + rdcoeff[3, ci[ci == 0], 1] * sza[ci == 0])
        )
        - rdcoeff[3, ci[ci == 0], 2]
        - delta[ci == 0] * rdcoeff[3, ci[ci == 0], 3]
    )
    # coeff is array of shape (5, N)
    return coeff


def perez_1993(
    bh: Union[float, np.ndarray],
    dh: Union[float, np.ndarray],
    sza: Union[float, np.ndarray],
    saa: Union[float, np.ndarray],
    pza: Union[float, np.ndarray],
    paa: Union[float, np.ndarray],
    albedo: Union[float, np.ndarray],
    doy: Union[int, float, np.ndarray],
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the irradiance for a tilted plane in an arbitrary shaded
    situation using the Perez 1993 sky radiance distribution model.

    N scenarios are calculated at once. Each with it's own set of solar
    variables and plane orientations.

    :param bh:(ndarray of shape (N,)) beam horizontal irradiance
    :param dh: (ndarray of shape (N,)) diffuse horizontal irradiance
    :param sza: (ndarray of shape (N,)) solar zenith angle (radians)
    :param saa: (ndarray of shape (N,)) solar azimuth angle (radians).
        Measured clockwise from North.
    :param pza: (ndarray of shape (N,)) plane tilt angle (radians)
    :param paa: (ndarray of shape (N,)) plane azimuth angle (radians).
        Measured clockwise from North.
    :param albedo: (ndarray of shape (N,)) Local ground reflectivity
    :param doy: (ndarray of shape (N,)) Day of year from 1 (Jan 1st) to 366

    :return gti: (array of shape (N,)) global tilted irradiance
    :return bti: (array of shape (N,)) beam tilted irradiance
    :return dti: (array of shape (N,)) diffuse + reflected tilted irradiance
    """
    bh, dh, sza, saa, pza, paa = np.broadcast_arrays(bh, dh, sza, saa, pza, paa)

    bt = beam_rad(bh, sza, saa, pza, paa)
    drt = diffuse_rad(bh, dh, sza, saa, pza, paa, albedo, doy)
    gt = bt + drt
    return gt, bt, drt


def beam_rad(
    bh: np.ndarray, sza: np.ndarray, saa: np.ndarray, pza: np.ndarray, paa: np.ndarray
) -> np.ndarray:
    """
    Beam radiance
    """
    cosaoi = calc_aoi([sza, saa], [pza, paa], cos=True)
    b = np.cos(sza)
    b[b < 0.08716] = 0.08716  # if sza is below the horizon bh is 0 already
    cosaoi[cosaoi < 0] = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        bt = bh / b * cosaoi

    return bt


def diffuse_rad(
    bh: np.ndarray,
    dh: np.ndarray,
    sza: np.ndarray,
    saa: np.ndarray,
    pza: np.ndarray,
    paa: np.ndarray,
    albedo: np.ndarray,
    doy: Union[int, float, np.ndarray],
    include_refrad=True,
) -> np.ndarray:
    """
    Calculate the diffuse irradiance falling on a tilted plane

    N scenarios (typically different times or locations) each with a
    different plane angle are the arguments. Return is of shape (N,).

    :param bh: (array of shape (N,)) beam horizontal irradiance
    :param dh: (array of shape (N,)) diffuse horizontal irradiance
    :param sza: (ndarray of shape (N,)) zenith angle (radians) of sun direction
    :param saa: (ndarray of shape (N,)) azimuth angle (radians) of sun direction
    :param pza: (ndarray of shape (N,)) zenith angle (radians) of plane direction
    :param paa: (ndarray of shape (N,)) azimuth angle (radians) of plane direction
    :param doy: (array of shape (N,)) day of year
    :param albedo: (array of shape (N,)) local surface albedo
    :param include_refrad: include surface reflectance (output is diffuse + reflected)
        otherwise reflected component is set to 0

    :return dt: (ndarray of shape (N,)) diffuse + reflected irradiance
    """

    # calc diffuse radiance on integration sphere. Result of shape
    # (N, L) where L is the number of solid angle integration elements)
    diffrad, thetai, phii, domega = calc_diffuse_radiances(bh, dh, sza, saa, doy)

    # ref rad of shape N, L
    refrad = calc_lambertian_reflected_radiances(bh + dh, albedo)
    if not include_refrad:
        refrad = refrad * 0

    n = len(dh)
    l = thetai.shape[0]

    # Now calc the angle of incidence between L elements and N plane
    # orientations. Result is of shape (N, L)
    cosaoiplane = calc_aoi(
        (thetai[np.newaxis, :], phii[np.newaxis, :]),
        (pza[:, np.newaxis], paa[:, np.newaxis]),
        cos=True,
    )

    dels = (diffrad + refrad) * cosaoiplane

    # don't get any incidence from behind the plane
    dels[cosaoiplane < 0] = 0

    # Do the spherical integration. Reduce to shape (N,)
    dt = np.dot(dels, domega)

    return dt


def calc_diffuse_shape(
    theta: np.ndarray,
    phi: np.ndarray,
    sza: np.ndarray,
    saa: np.ndarray,
    coeff: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate the diffuse radiance shape for N scenarios at L different
    viewing angles. Return arrays of shape (N, L).

    :param theta: (ndarray of shape (L,)) zenith angle (radians) of incoming radiation
    :param phi: (ndarray of shape (L,)) azimuth angle (radians) of incoming radiation.
    :param sza: (ndarray of shape (N,)) zenith angle (radians) of sun direction
    :param saa: (ndarray of shape (N,)) azimuth angle (radians) of sun direction.
    :param coeff: (ndarray of shape (5, N))
    """

    # Need to calculate the radiation for m elements for the spherical
    # integration So that we can use pure numpy (avoid for loops) we
    # broadcast the n requested scenarios and m angles into a mesh grid (of
    # shape (N, L) and use that to calculate the diffuse radiation
    thetab, phib, szab, saab = np.broadcast_arrays(
        theta[np.newaxis, :], phi[np.newaxis, :], sza[:, np.newaxis], saa[:, np.newaxis]
    )
    coeffb = np.broadcast_to(coeff[:, :, np.newaxis], (5, sza.shape[0], theta.shape[0]))

    cosaoi = calc_aoi((thetab, phib), (szab, saab), cos=True)
    costhetab = np.cos(thetab)
    diffuse_shape = (1 + coeffb[0] * np.exp(coeffb[1] / costhetab)) * (
        1 + coeffb[2] * np.exp(coeffb[3] * np.arccos(cosaoi)) + coeffb[4] * cosaoi**2
    )

    # don't get diffuse radiance below horizon
    diffuse_shape[costhetab < 0] = 0

    return diffuse_shape, thetab, phib


def calc_diffuse_radiances(
    bh: np.ndarray,
    dh: np.ndarray,
    sza: np.ndarray,
    saa: np.ndarray,
    doy: Union[int, float, np.ndarray],
    thetaphi: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Diffuse radiance contribution for radiance coming from multiple theta,
    phi directions. (per steradian of incoming angle variation, per m^2 of
    cross-sectional area of a sphere)

    As usual it's calculated for multiple scenarios so we can do numpy
    looping. For each of the N scenarios all the L thetas and phis (viewing
    directions) are calculated.

    ie spherical integral(ddir) * cross sectional area of sphere pi*R^2 =
        total radiation W incident on sphere

    :param bh: (ndarray of shape (N,)) beam horizontal irradiance
    :param dh: (ndarray of shape (N,)) diffuse horizontal irradiance
    :param sza: (ndarray of shape (N,)) zenith angle (radians) of sun direction
    :param saa: (ndarray of shape (N,)) azimuth angle (radians) of sun direction.
        (Defined as clockwise from North).
    :param doy: (ndarray of shape (N,)) day of year
    :param thetaphi: (ndarray of shape (2, L)) (zenith, azimuth) zenith angle of
        incoming radiation (viewing angle)

    :return: (ndarray of shape (N, L)) the radiances for each scenario and for
        each theta/phi combination

    And optionally if thetaphi is not given:
    :return theta: (ndarray of shape (L,)) the theta values of the viewing
        angles for each radiance scenario
    :return phi: (ndarray of shape (L,)) the phi values of the viewing angles
        for each radiance scenario
    :return domega: (ndarray of shape (L,)) the solid angles of the
        integration elements of the lebedev sphere at each of the theta/phi
        pairs.
    """

    # clearness and brightness params
    ci, delta = calc_ci_delta_array(bh, dh, sza, doy)

    # model coefficients
    coeff = get_coeff(ci, delta, sza)

    # First calculate the diffuse shape on the lebedev spherical integration
    # points. We're going to need this for the normalisation anyway and it
    # may be we can re use it for the returned diffuse points as well. We
    # use a high degree lebedev model because the function is typically
    # highly discontinuous due to plane edges, horizon edges and the sharp
    # circumsolar diffuse radiation. In future maybe we could set up the
    # integration elements based on where the discontinuities are.
    thetai, phii, domegas = spherical_integration_elements(lebedev_degree=131)

    n = sza.shape[0]
    l = thetai.shape[0]

    # Result of shape (N, L). Where L is number of integration elements
    # (viewing angles).
    diffuse_shape, thetab, phib = calc_diffuse_shape(thetai, phii, sza, saa, coeff)

    # Integrate over the sphere to calculate the normalisation factor.
    # Result of shape (N,)
    nint = np.dot(diffuse_shape * np.cos(thetab), domegas)
    norm = nint / dh

    if thetaphi is None:

        # return the diffuse radiances on the lebedev spherical integration
        # elements
        return diffuse_shape / norm[:, np.newaxis], thetai, phii, domegas

    else:

        # Calculate the diffuse_rad using the supplied angles
        theta, phi = thetaphi[0], thetaphi[1]
        diffuse_shape, thetab, phib = calc_diffuse_shape(theta, phi, sza, saa, coeff)

        # Return of shape (N, L) where L is thetaphi[0].shape[0]
        return diffuse_shape / norm[:, np.newaxis]


def calc_lambertian_reflected_radiances(
    gh: np.ndarray,
    albedo: Union[float, np.ndarray],
    thetaphi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reflected radiance contribution for radiance coming from multiple theta,
    phi directions. (per steradian of incoming angle variation, per m^2 of
    cross-sectional area of a sphere). This is assuming an infite plane of
    perfect lambertian reflection.

    ie spherical integral(ddir) * cross sectional area of sphere pi*R^2 =
        total radiation W incident on sphere

    :param gh: (ndarray of shape (N,)) global horizontal irradiance for
        target spectral filter
    :param albedo: local ground albedo
    :param thetaphi: array of vector angles to calculate radiance on
    """

    if thetaphi is None:
        theta, phi, domega = spherical_integration_elements(lebedev_degree=131)
    else:
        theta, phi = thetaphi[0], thetaphi[1]
    refrad = albedo * gh / np.pi
    refrad = refrad[:, np.newaxis] * (theta > np.pi / 2)[np.newaxis, :]
    return refrad
