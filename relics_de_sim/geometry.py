"""Loaders + helpers for static detector geometry (PMT layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from relics_de_sim.constants import N_BOT_PMT, N_TOP_PMT, PMT_DTYPE


def load_pmt_layout(top_path: str | Path, bot_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read the top + bottom PMT geometry text files.

    The legacy code parsed both files with ``np.loadtxt(path, dtype=PMT_DTYPE)``;
    we forward that dtype so existing files continue to work unchanged.

    Returns
    -------
    pmt_top, pmt_bot : np.ndarray
        Structured arrays of length :data:`N_TOP_PMT` and :data:`N_BOT_PMT`.
    """
    pmt_top = np.loadtxt(top_path, dtype=PMT_DTYPE)
    pmt_bot = np.loadtxt(bot_path, dtype=PMT_DTYPE)
    if pmt_top.size != N_TOP_PMT:
        raise ValueError(f"Expected {N_TOP_PMT} top PMTs, got {pmt_top.size} from {top_path}")
    if pmt_bot.size != N_BOT_PMT:
        raise ValueError(f"Expected {N_BOT_PMT} bottom PMTs, got {pmt_bot.size} from {bot_path}")
    return pmt_top, pmt_bot


def fiducial_xy_mask(xd: np.ndarray, yd: np.ndarray, radius_mm: float) -> np.ndarray:
    """True where (xd, yd) lies inside a disk of given radius (mm)."""
    return xd * xd + yd * yd < radius_mm * radius_mm


def sample_disk_uniform(
    radius_mm: float, n: int, rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample `n` 2D points uniformly within a disk of given radius."""
    rng = rng or np.random.default_rng()
    angle = 2.0 * np.pi * rng.random(size=n)
    distance = radius_mm * np.sqrt(rng.random(size=n))
    return distance * np.cos(angle), distance * np.sin(angle)
