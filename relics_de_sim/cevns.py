"""CEvNS event sampling.

Replaces ``DE.generate_CEVNS_points`` and the per-multiplicity scaffolding
that lived in ``DE.DESimulation``.

The legacy code maintained one structured array per electron-multiplicity bin
(3..10 electrons). We keep that shape for compatibility with downstream
notebook code, but expose it through a plain :class:`CEvNSGenerator` that owns
the spectrum and supports re-seeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from relics_de_sim import constants as C
from relics_de_sim.geometry import sample_disk_uniform


@dataclass
class CEvNSSpectrum:
    """The CEvNS multiplicity spectrum loaded from disk."""

    e_num: np.ndarray  # electron multiplicity per bin (e.g. [1..10])
    counts: np.ndarray  # number of events to draw per bin

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        e_num_slice: slice = slice(1, 11),
    ) -> "CEvNSSpectrum":
        data = np.load(path)
        e_num = data["CEvNS_bins"][e_num_slice]
        counts = data["CEvNS"][e_num_slice].astype(int)
        return cls(e_num=e_num, counts=counts)


class CEvNSGenerator:
    """Sample CEvNS events uniformly in the fiducial volume.

    Parameters
    ----------
    fiducial_radius_mm, height_mm :
        Disk radius and TPC height for uniform sampling.
    time_range_s :
        Total simulated wall-clock window.
    spectrum :
        :class:`CEvNSSpectrum` controlling the number of events per bin.
    rng :
        Optional ``numpy.random.Generator``.
    """

    def __init__(
        self,
        fiducial_radius_mm: float,
        height_mm: float,
        time_range_s: float,
        spectrum: CEvNSSpectrum,
        rng: np.random.Generator | None = None,
        time_buffer_s: float = 2.0,
    ) -> None:
        self.fiducial_radius_mm = fiducial_radius_mm
        self.height_mm = height_mm
        self.time_range_s = time_range_s
        self.spectrum = spectrum
        self.rng = rng or np.random.default_rng()
        self.time_buffer_s = time_buffer_s

    def sample(self) -> List[np.ndarray]:
        """Return one structured array per electron-multiplicity bin.

        Each array uses dtype :data:`relics_de_sim.constants.CEVNS_DTYPE`.
        """
        events_per_bin: List[np.ndarray] = []
        running_id = 0
        for e_num, count in zip(self.spectrum.e_num, self.spectrum.counts):
            if count <= 0:
                events_per_bin.append(np.zeros(0, dtype=C.CEVNS_DTYPE))
                continue

            arr = np.zeros(count, dtype=C.CEVNS_DTYPE)
            arr["cevnsID"] = np.arange(count) + running_id
            arr["num_e"] = e_num
            xs, ys = sample_disk_uniform(self.fiducial_radius_mm, count, self.rng)
            arr["pos_original"]["xd"] = xs
            arr["pos_original"]["yd"] = ys
            arr["pos_original"]["zd"] = self.rng.random(size=count) * self.height_mm
            t_low = self.time_buffer_s
            t_high = max(self.time_range_s - self.time_buffer_s, t_low)
            arr["t"] = self.rng.uniform(t_low, t_high, size=count)

            arr = np.sort(arr, order="t")
            events_per_bin.append(arr)
            running_id += count
        return events_per_bin
