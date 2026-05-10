"""Delayed-electron generation.

Given the dense interpolation of muon energy depositions, draw one delayed
electron per ``num_e_delayed`` count, jittering the (x, y) position with a
Gaussian and the arrival time with a power-law tail. Replaces
``DE.delayed_electrons_sorted``.
"""

from __future__ import annotations

import numpy as np

from relics_de_sim import constants as C
from relics_de_sim.geometry import fiducial_xy_mask
from relics_de_sim.muon import power_law_samples


def generate_delayed_electrons(
    dense_muon: np.ndarray,
    dead_time_s: float,
    diffuse_length_mm: float,
    fiducial_radius_mm: float = 139.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one delayed electron per ``num_e_delayed`` count along a muon track.

    Parameters
    ----------
    dense_muon :
        Output of :func:`relics_de_sim.muon.compute_dense_muon_points`.
    dead_time_s :
        Lower bound of the power-law arrival-time distribution.
    diffuse_length_mm :
        Variance of the per-axis Gaussian xy jitter applied around each
        muon point.
    fiducial_radius_mm :
        Radius outside which DEs are dropped (DE.py uses 139 mm).

    Returns
    -------
    np.ndarray
        Structured array with dtype ``DELAYED_ELECTRON_DTYPE``, sorted by
        ``e_time``.
    """
    rng = rng or np.random.default_rng()

    n_total = int(dense_muon["num_e_delayed"].sum())
    out = np.zeros(n_total, dtype=C.DELAYED_ELECTRON_DTYPE)

    counts = dense_muon["num_e_delayed"]
    out["eventId"] = np.repeat(dense_muon["eventId"], counts)
    out["xd"] = np.repeat(dense_muon["xd"], counts)
    out["yd"] = np.repeat(dense_muon["yd"], counts)
    out["muon_time"] = np.repeat(dense_muon["muon_time"], counts)

    # Power-law arrival-time samples.
    delay_time = power_law_samples(
        dead_time_s,
        C.DELAY_TIME_TMAX_S,
        C.DELAY_TIME_GAMMA,
        size=n_total,
        rng=rng,
    )

    # 2-D Gaussian xy jitter (covariance matrix ``[[diffuse, 0], [0, diffuse]]``,
    # mirroring DE.py exactly).
    cov = np.array([[diffuse_length_mm, 0.0], [0.0, diffuse_length_mm]])
    jitter = rng.multivariate_normal([0.0, 0.0], cov, size=n_total)
    out["xd"] += jitter[:, 0]
    out["yd"] += jitter[:, 1]
    out["e_time"] = out["muon_time"] + delay_time

    out = out[np.argsort(out["e_time"])]
    out = out[fiducial_xy_mask(out["xd"], out["yd"], fiducial_radius_mm)]
    return out
