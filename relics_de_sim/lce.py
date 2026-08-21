"""Light-collection-efficiency (LCE) maps.

The LCE map is a per-channel 2-D table giving the expected number of pe per
single drifted electron arriving at a particular (x, y) location at the gate.
The legacy code held three NumPy arrays (``LCE_xs``, ``LCE_ys``, ``LCE_value``)
and looped over the 128 channels with :func:`scipy.interpolate.interpn` -- one
call per channel.

This module wraps that representation behind :class:`LCEMap` so that:

* the disk + channel layout is loaded once;
* ``light_pattern(xy)`` returns the full ``(N, 128)`` array in a single
  vectorised pass (still using ``interpn`` per channel, since that path was
  shown empirically to be faster than multiprocessing for typical batch sizes);
* an alternative ``true`` map can be plugged in for systematics studies, mirroring
  the old ``LCE_value`` vs ``LCE_value_true`` distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interpn

from relics_de_sim.constants import N_TOTAL_PMT


@dataclass
class LCEMap:
    """Container for the per-channel LCE table.

    Attributes
    ----------
    xs, ys : np.ndarray
        Grid coordinates (length Nx, Ny).
    value : np.ndarray
        Shape ``(N_TOTAL_PMT, Nx, Ny)``.  ``value[ch, ix, iy]`` is the expected
        pe per electron at that (x, y) on channel `ch`.
    """

    xs: np.ndarray
    ys: np.ndarray
    value: np.ndarray

    def __post_init__(self) -> None:
        if self.value.shape[0] != N_TOTAL_PMT:
            raise ValueError(
                f"LCE value array must have leading dim {N_TOTAL_PMT}, got {self.value.shape}"
            )
        if self.value.shape[1:] != (len(self.xs), len(self.ys)):
            raise ValueError(
                f"LCE grid shape mismatch: expected ({len(self.xs)}, {len(self.ys)}), "
                f"got {self.value.shape[1:]}"
            )

    @classmethod
    def from_paths(
        cls, xs_path: str | Path, ys_path: str | Path, value_path: str | Path
    ) -> "LCEMap":
        return cls(xs=np.load(xs_path), ys=np.load(ys_path), value=np.load(value_path))

    def light_pattern(self, xy: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Interpolate the per-channel response for a batch of (x, y) positions.

        Parameters
        ----------
        xy : np.ndarray, shape (N, 2)
            Positions in mm.
        fill_value : float
            Returned for points that fall outside the grid.

        Returns
        -------
        np.ndarray, shape (N, N_TOTAL_PMT)
            Expected pe per electron, per channel.
        """
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (N, 2), got {xy.shape}")
        n = xy.shape[0]
        out = np.empty((n, N_TOTAL_PMT), dtype=np.float64)
        # Per-channel interpolation: scipy.interpn does not support a leading
        # batch axis, so we loop. The cost is dominated by the actual
        # interpolation, not Python overhead, for typical n>>128.
        for ch in range(N_TOTAL_PMT):
            out[:, ch] = interpn(
                (self.xs, self.ys),
                self.value[ch],
                xy,
                bounds_error=False,
                fill_value=fill_value,
            )
        return out


def load_lce(
    xs_path: str | Path,
    ys_path: str | Path,
    value_path: str | Path,
    value_true_path: Optional[str | Path] = None,
) -> tuple[LCEMap, LCEMap]:
    """Load the nominal and 'truth' LCE maps used by the legacy pipeline.

    If ``value_true_path`` is ``None``, the same map is returned twice.
    """
    nominal = LCEMap.from_paths(xs_path, ys_path, value_path)
    true_map = (
        LCEMap.from_paths(xs_path, ys_path, value_true_path)
        if value_true_path is not None
        else nominal
    )
    return nominal, true_map
