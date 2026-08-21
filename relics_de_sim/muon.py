"""Muon track ingestion + dense interpolation.

The legacy ``DE.generate_muon_points`` interleaved file IO, time-stamping,
electron-yield calculation and dense-point interpolation in one method. We
split those concerns:

``load_muon_files``   reads a list of muon-track ``.npy`` files, concatenates
them with monotonically-increasing event ids, and assigns each event a random
arrival time within ``time_range = N_muon / muon_rate``.

``compute_dense_muon_points``   linearly interpolates ``interp_points_per_muon``
points along each track and attaches the depth-dependent expected DE count
(``num_e_delayed``).

A dataclass :class:`MuonRunSummary` collects the few scalars
(``time_range_s``, ``n_muons``, ``dead_time_ratio``) that downstream stages
need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from numpy.lib import recfunctions as rfn
from tqdm import tqdm

from relics_de_sim import constants as C


@dataclass
class MuonRunSummary:
    """Scalar metadata produced when ingesting a set of muon files."""

    n_muons: int
    time_range_s: float
    dead_time_ratio: float


def power_law_samples(
    a: float, b: float, gamma: float, size: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw samples from a power law ``pdf(x) ~ x ** (gamma - 1)`` on ``[a, b]``.

    Reproduces ``DE.rndm`` so the dead-time-ratio integral stays bit-compatible.
    """
    rng = rng or np.random.default_rng()
    r = rng.random(size=size)
    ag, bg = a**gamma, b**gamma
    return (ag + (bg - ag) * r) ** (1.0 / gamma)


def estimate_dead_time_ratio(
    dead_time_s: float, rng: np.random.Generator | None = None
) -> float:
    """Fraction of the natural DE arrival-time tail surviving the dead window.

    Identical to ``DE.DESimulation.__init__`` lines 130-135.
    """
    samples = power_law_samples(
        C.DEAD_TIME_RATIO_TMIN_S,
        C.DEAD_TIME_RATIO_TMAX_S,
        C.DEAD_TIME_RATIO_GAMMA,
        C.DEAD_TIME_RATIO_NSAMPLE,
        rng=rng,
    )
    return float((samples >= dead_time_s).sum() / samples.size)


def load_muon_files(
    file_list: Sequence[str | Path],
    muon_rate_hz: float,
    quenching_factor: float,
    rng: np.random.Generator | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, MuonRunSummary]:
    """Concatenate muon-track files, stamp each muon with a random arrival time.

    Parameters
    ----------
    file_list :
        Iterable of paths to muon_track ``.npy`` files (Geant4 export format
        with at least ``eventId`` and ``energy`` fields plus xd/yd/zd).
    muon_rate_hz :
        Mean muon rate; ``time_range = N_muons / muon_rate`` defines the
        simulated wall-clock window.
    quenching_factor :
        Mean ionisation electrons per keV deposited (legacy ``quenching_factor``).

    Returns
    -------
    merged : np.ndarray
        Structured array carrying every original field plus ``electronNum``
        (per-step electron yield) and ``muon_time`` (s).
    summary : MuonRunSummary
    """
    rng = rng or np.random.default_rng()

    iterator: Iterable = tqdm(list(enumerate(file_list)), desc="Reading muon files") if progress else enumerate(file_list)
    arrays: List[np.ndarray] = []
    pre_event_num = 0
    for index, path in iterator:
        arr = np.load(path)
        file_event_num = int(arr["eventId"][-1]) + 1
        arr = arr.copy()
        arr["eventId"] = arr["eventId"] + pre_event_num
        pre_event_num += file_event_num
        arrays.append(arr)
    merged = np.concatenate(arrays, axis=0)

    n_muons = int(merged["eventId"][-1]) + 1
    time_range = n_muons / muon_rate_hz
    event_time = np.sort(rng.uniform(0.0, time_range, size=n_muons))

    electron_num = (quenching_factor * merged["energy"]).astype("uint32")
    muon_time_per_step = event_time[merged["eventId"]]

    merged = rfn.append_fields(
        merged,
        names=["electronNum", "muon_time"],
        data=[electron_num, muon_time_per_step],
        usemask=False,
    )
    # Stash event_time inside the summary so callers can pass it to
    # ``compute_dense_muon_points`` without recomputing.
    summary = MuonRunSummary(
        n_muons=n_muons,
        time_range_s=float(time_range),
        dead_time_ratio=float("nan"),  # to be filled by the caller
    )
    summary.event_time = event_time  # type: ignore[attr-defined]
    return merged, summary


def compute_dense_muon_points(
    merged_muon: np.ndarray,
    event_time: np.ndarray,
    dead_time_ratio: float,
    interp_num: int = 100,
) -> np.ndarray:
    """Linearly interpolate dense points along each muon track.

    Reproduces ``DE.dense_muon_gen`` exactly; the only behavioural difference
    is that ``interp_num`` is an explicit argument (it was hard-wired to 100
    inside ``DE.generate_muon_points``).
    """
    if interp_num != 100:
        # The legacy code hard-codes 100 in several places (eg the eventId
        # broadcasting `np.arange(N) / 100`).  Loosening that requires a careful
        # audit of the downstream space-time correlation indexing; until then
        # we keep parity with the legacy choice.
        raise NotImplementedError(
            "interp_num != 100 would break the legacy correlation indexing."
        )

    muon_sim_num = int(merged_muon["eventId"][-1]) + 1
    dense = np.zeros(muon_sim_num * interp_num, dtype=C.DENSE_MUON_DTYPE)

    _, start_indices = np.unique(merged_muon["eventId"], return_index=True)
    end_indices = np.concatenate([start_indices[1:] - 1, [len(merged_muon) - 1]])
    start_point = merged_muon[start_indices]
    end_point = merged_muon[end_indices]
    gap_x = end_point["xd"] - start_point["xd"]
    gap_y = end_point["yd"] - start_point["yd"]
    gap_z = end_point["zd"] - start_point["zd"]

    # Legacy spacing: arange(0, 100/99, 1/99) -> [0, 1/99, 2/99, ..., 99/99],
    # i.e. the last interpolant lands exactly on the track end.
    arr = np.tile(np.arange(0, 100 / 99, 1 / 99), (muon_sim_num, 1))[:, :interp_num]

    x_dense = arr * gap_x[:, None] + start_point["xd"][:, None]
    y_dense = arr * gap_y[:, None] + start_point["yd"][:, None]
    z_dense = arr * gap_z[:, None] + start_point["zd"][:, None]

    dense["xd"] = x_dense.flatten()
    dense["yd"] = y_dense.flatten()
    dense["zd"] = z_dense.flatten()
    dense["eventId"] = (np.arange(muon_sim_num * interp_num) // interp_num).astype("uint32")

    # Per-event sums of step energy and electron yield, divided by interp_num.
    event_ids = merged_muon["eventId"]
    _, event_id_indices = np.unique(event_ids, return_inverse=True)
    muon_electron_sum = np.bincount(event_id_indices, weights=merged_muon["electronNum"])
    muon_step_energy = np.bincount(event_id_indices, weights=merged_muon["energy"])
    average_electron = (muon_electron_sum / interp_num).astype("uint32")
    average_energy = (muon_step_energy / interp_num).astype("uint32")

    dense["energy"] = average_energy[dense["eventId"]]
    dense["num_e"] = average_electron[dense["eventId"]]
    dense["muon_time"] = event_time[dense["eventId"]]

    # delay_ratio = (
    #     (dense["zd"] - C.DELAY_RATIO_Z_INTERCEPT_MM) / C.DELAY_RATIO_Z_SLOPE / 1000.0
    #     + C.DELAY_RATIO_OFFSET
    # ) / C.DELAY_RATIO_DENOMINATOR
    # delay_ratio = np.clip(delay_ratio, C.DELAY_RATIO_FLOOR, None)
    delay_ratio = 0.001
    dense["num_e_delayed"] = (dense["num_e"] * delay_ratio * dead_time_ratio).astype("uint32")
    return dense
