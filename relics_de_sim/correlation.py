"""Space-time correlation between candidate events and recent muons.

Replaces ``DE.space_time_cor_new``. Computes, for every event, a scalar score::

    cor = Σ_p  (1 / 2π r²) · exp(-||Δp||² / 2r²) · Δt_p^{-γ} · n_e_delayed_p

where ``p`` runs over every dense muon point belonging to the previous
``look_ahead`` muons.

Implementation notes
--------------------
* The legacy code processed events in batches of 100 000 to bound memory.
  We keep that batching but vectorise the inner gather (``expanded_points``)
  via :func:`numpy.lib.stride_tricks.as_strided`-free slicing.
* Returns the same scalar per event plus optional per-batch (Δx, Δy, Δt)
  blobs that the legacy code emitted; downstream notebooks plot histograms
  from those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class CorrelationDeltas:
    """Aggregated Δ-coordinates between events and look-back muon points.

    Each list entry corresponds to one batch (so the user can drop the heaviest
    batches when memory is tight).
    """

    delta_x: List[np.ndarray]
    delta_y: List[np.ndarray]
    delta_t: List[np.ndarray]


def space_time_correlation(
    event_t: np.ndarray,
    event_xy: np.ndarray,
    dense_muon: np.ndarray,
    last_muon_id: np.ndarray,
    *,
    radius_mm: float = 20.0,
    gamma: float = 3.0,
    look_ahead: int = 20,
    interp_per_muon: int = 100,
    batch_size: int = 100_000,
    progress: bool = True,
) -> Tuple[np.ndarray, CorrelationDeltas]:
    """Compute per-event space-time correlation against recent muon points.

    Parameters
    ----------
    event_t : (N,) float
        Event arrival times (s).
    event_xy : (N, 2) float
        Event (x, y) positions (mm). Pre-broken-out from a structured array.
    dense_muon :
        Output of :func:`relics_de_sim.muon.compute_dense_muon_points`.
    last_muon_id : (N,) int
        Index of the most recent muon ``<= event_t``. Use
        ``np.searchsorted(event_time, event_t, side="right") - 1``.
    radius_mm, gamma, look_ahead :
        Parameters of the correlation kernel (DE.py defaults: 20, 3, 20).
    interp_per_muon :
        Must match the value used to build ``dense_muon`` (legacy 100).
    batch_size :
        Events processed per batch (legacy 100 000).

    Returns
    -------
    correlation : (N,) np.ndarray
    deltas : CorrelationDeltas
    """
    n_events = len(event_t)
    if event_xy.shape != (n_events, 2):
        raise ValueError("event_xy must have shape (N, 2).")
    if last_muon_id.shape != (n_events,):
        raise ValueError("last_muon_id length must match event_t.")

    correlation = np.zeros(n_events, dtype=np.float64)
    deltas = CorrelationDeltas([], [], [])

    # Pre-extract the columns we need; this avoids repeatedly indexing into
    # the structured array inside the inner loop.
    mu_x = dense_muon["xd"]
    mu_y = dense_muon["yd"]
    mu_t = dense_muon["muon_time"]
    mu_n = dense_muon["num_e_delayed"]

    n_window = interp_per_muon * look_ahead

    iterator = range(0, n_events, batch_size)
    if progress:
        iterator = tqdm(iterator, desc="space-time correlation", total=(n_events + batch_size - 1) // batch_size)

    for start in iterator:
        stop = min(n_events, start + batch_size)
        size = stop - start
        block_x = np.zeros((size, n_window), dtype=np.float64)
        block_y = np.zeros((size, n_window), dtype=np.float64)
        block_t = np.zeros((size, n_window), dtype=np.float64)
        block_n = np.zeros((size, n_window), dtype=np.float64)

        for i in range(size):
            muon_id = int(last_muon_id[start + i])
            muon_start = max(0, muon_id - (look_ahead - 1))
            muon_stop = muon_id  # inclusive
            point_start = muon_start * interp_per_muon
            point_stop = (muon_stop + 1) * interp_per_muon
            length = point_stop - point_start
            block_x[i, :length] = mu_x[point_start:point_stop]
            block_y[i, :length] = mu_y[point_start:point_stop]
            block_t[i, :length] = mu_t[point_start:point_stop]
            block_n[i, :length] = mu_n[point_start:point_stop]

        delta_x = event_xy[start:stop, 0:1] - block_x
        delta_y = event_xy[start:stop, 1:2] - block_y
        delta_t = event_t[start:stop, None] - block_t
        deltas.delta_x.append(delta_x)
        deltas.delta_y.append(delta_y)
        deltas.delta_t.append(delta_t)

        # Spatial Gaussian density.
        distance_sqr = delta_x * delta_x + delta_y * delta_y
        xy_density = (1.0 / (2.0 * np.pi * radius_mm**2)) * np.exp(
            -0.5 * distance_sqr / radius_mm**2
        )
        # Time prior, with safe handling of empty look-back rows.
        with np.errstate(divide="ignore", invalid="ignore"):
            time_prior = np.where(delta_t > 0, np.power(delta_t, -gamma), 0.0)

        density = xy_density * time_prior * block_n
        # Rows of all-zero block_t come from "no preceding muon" -- mask out.
        density[block_t == 0.0] = 0.0
        correlation[start:stop] = density.sum(axis=1)

    return correlation, deltas
