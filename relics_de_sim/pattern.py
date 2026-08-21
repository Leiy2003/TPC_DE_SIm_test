"""Per-event S2 PMT pattern simulation.

Replaces 4-5 near-identical methods in the legacy ``DE.DESimulation`` class
(``generate_DE_pattern``, ``generate_DE_pattern_linear``,
``generate_DE_recon_pattern``, ``generate_CEVNS_pattern``,
``generate_CEVNS_recon_pattern``).

Pipeline per event::

    light_per_e[n_electrons, 128] = LCE(xy) * single_e_gain * gamma_jitter
    pe_per_e[n_electrons, 128]    = Poisson(light_per_e)
    light_event[n_events, 128]    = sum over n_electrons
    pe_event[n_events, 128]       = sum over n_electrons
    area_event[n_events, 128]     = sum_p pe_p * Normal(pe_gain, pe_gain_std)

The single :class:`PatternSimulator` exposes those steps so each caller can
combine them as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from relics_de_sim import constants as C
from relics_de_sim.config import ElectronicsConfig
from relics_de_sim.lce import LCEMap


@dataclass
class PatternResult:
    """Container for the per-event pattern arrays.

    Attributes
    ----------
    light_pattern : (n_events, 128)
        Expected pe per channel (continuous).
    pe_pattern : (n_events, 128)
        Poisson-sampled pe per channel.
    area_pattern : (n_events, 128)
        Per-pe area aggregated per channel.
    pe_by_area : (n_events, 128)
        ``area_pattern / pe_gain`` -- a smooth pe estimate.
    pe_info : (n_total_pe,) struct
        Long-format record of every individual pe used (event_id, area, channel,
        e_time, e_id).  Required for waveform synthesis.
    keep_mask : (n_events_in,)
        Boolean mask over the original (pre-cut) event list, telling the
        caller which events survived the area window.
    """

    light_pattern: np.ndarray
    pe_pattern: np.ndarray
    area_pattern: np.ndarray
    pe_by_area: np.ndarray
    pe_info: np.ndarray
    keep_mask: np.ndarray


class PatternSimulator:
    """Simulates the S2 PMT pattern given electron positions.

    Parameters
    ----------
    lce :
        :class:`LCEMap` providing the per-channel response.
    electronics :
        :class:`relics_de_sim.config.ElectronicsConfig` carrying gain constants.
    rng :
        Optional ``numpy.random.Generator``.
    """

    # def __init__(
    #     self,
    #     lce: LCEMap,
    #     electronics: ElectronicsConfig,
    #     rng: np.random.Generator | None = None,
    # ) -> None:
    #     self.lce = lce
    #     self.elec = electronics
    #     self.rng = rng or np.random.default_rng()

    def __init__(
        self,
        lce: LCEMap,
        electronics: ElectronicsConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.lce = lce
        self.rng = rng or np.random.default_rng()
        
        # 安全转换函数
        def safe_float(value, name):
            try:
                return float(value)
            except (TypeError, ValueError) as e:
                raise TypeError(f"{name} must be convertible to float, got {type(value)}: {value}") from e
        
        # 转换所有需要的数值
        self.pe_gain = safe_float(electronics.pe_gain, 'pe_gain')
        self.pe_gain_std = safe_float(electronics.pe_gain_std, 'pe_gain_std')
        self.se_gamma_sigma = safe_float(electronics.se_gamma_sigma, 'se_gamma_sigma')
        self.single_electron_gain_per_channel = safe_float(
            electronics.single_electron_gain_per_channel, 
            'single_electron_gain_per_channel'
        )
        self.average_efficiency = safe_float(electronics.average_efficiency, 'average_efficiency')
        
        # 保留原始 electronics 对象（以防其他地方需要）
        self.elec = electronics

    # ------------------------------------------------------------------ #
    # Stage 1 - per-electron expected pe per channel.
    # ------------------------------------------------------------------ #
    def per_electron_light(self, xy: np.ndarray) -> np.ndarray:
        """Return ``(n_electrons, 128)`` expected pe per channel.

        Includes the multiplicative SE-gain jitter (``Gaussian(1, gamma_sigma)``).
        Negative samples (rare, low-tail) are clipped to 0.
        """
        light = self.lce.light_pattern(xy)
        gamma = self.rng.normal(1.0, self.se_gamma_sigma, size=len(xy))
        light = light * self.single_electron_gain_per_channel * gamma[:, None]
        np.clip(light, 0.0, None, out=light)
        return light

    # ------------------------------------------------------------------ #
    # Stage 2 - aggregate per-event light + Poisson pe count.
    # ------------------------------------------------------------------ #
    def aggregate_per_event(
        self, per_electron_light: np.ndarray, n_per_event: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sum the per-electron arrays into per-event light + Poisson pe arrays.

        ``per_electron_light`` must have shape ``(n_events * n_per_event, 128)``;
        rows are assumed contiguous per event. Returns
        ``(light_per_event, pe_per_event, pe_per_electron)``.
        """
        if per_electron_light.shape[0] % n_per_event:
            raise ValueError(
                f"per_electron_light rows ({per_electron_light.shape[0]}) "
                f"not a multiple of n_per_event ({n_per_event})"
            )
        
        lambda_threshold = 1000  # 超过此值用正态近似
        mask_small = per_electron_light <= lambda_threshold
        mask_large = per_electron_light > lambda_threshold

        per_electron_pe = np.zeros_like(per_electron_light, dtype=np.int64)

        # 小λ用Poisson
        per_electron_pe[mask_small] = np.random.poisson(per_electron_light[mask_small])

        # 大λ用正态近似（泊松的均值和方差相等）
        if mask_large.any():
            large_lam = per_electron_light[mask_large]
            per_electron_pe[mask_large] = np.random.normal(
                large_lam, 
                np.sqrt(large_lam)
            ).round().astype(np.int64)
            
        # 确保非负
        per_electron_pe = np.maximum(per_electron_pe, 0)

        # per_electron_pe = self.rng.poisson(per_electron_light)
        light_event = per_electron_light.reshape(-1, n_per_event, C.N_TOTAL_PMT).sum(axis=1)
        pe_event = per_electron_pe.reshape(-1, n_per_event, C.N_TOTAL_PMT).sum(axis=1)
        return light_event, pe_event, per_electron_pe

    # ------------------------------------------------------------------ #
    # Stage 3 - per-event area pattern + area-window cut.
    # ------------------------------------------------------------------ #
    def area_pattern(
        self,
        pe_event: np.ndarray,
        per_electron_pe: np.ndarray,
        e_time_per_electron: np.ndarray,
        n_per_event: int,
        sim_low_pe: int,
        sim_up_pe: int,
        with_pe_info: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fold each event's pe count through the per-pe gain distribution.

        Returns
        -------
        area_pattern : (n_events_pass, 128)
        keep_mask : (n_events_in,) bool
        pe_info : (n_total_pe_pass,) structured  (or empty if ``with_pe_info=False``)
        """
        n_events = pe_event.shape[0]
        keep_mask = np.zeros(n_events, dtype=bool)

        area_patterns: list[np.ndarray] = []
        channel_indices_list: list[np.ndarray] = []
        pe_area_list: list[np.ndarray] = []
        event_id_list: list[np.ndarray] = []
        e_time_list: list[np.ndarray] = []
        
        pe_gain = float(self.pe_gain)

        # print(pe_gain.dtype)
        # The area window is expressed in pe in the legacy code (sim_low/up_limit)
        # but compared against an *area* sum.  Multiply by pe_gain to keep the
        # original logic: ``area > sim_low_pe * pe_gain``.
        area_low = sim_low_pe * pe_gain
        area_high = sim_up_pe * pe_gain

        out_event_id = 0
        for j in range(n_events):
            channel_indices = np.repeat(np.arange(C.N_TOTAL_PMT), pe_event[j])
            pe_area = self.rng.normal(
                self.pe_gain, self.pe_gain_std, size=len(channel_indices)
            )
            np.clip(pe_area, 0.0, None, out=pe_area)
            area_pattern = np.bincount(
                channel_indices, weights=pe_area, minlength=C.N_TOTAL_PMT
            )
            total_area = float(area_pattern.sum())
            if not (area_low < total_area < area_high):
                continue

            area_patterns.append(area_pattern)
            keep_mask[j] = True

            if with_pe_info:
                channel_indices_list.append(channel_indices)
                pe_area_list.append(pe_area)
                event_id_list.append(np.full(len(channel_indices), out_event_id, dtype=np.uint64))
                # Per-pe e_time tag: legacy code only computes this for DE pile-up.
                if e_time_per_electron is not None:
                    e_time_event = e_time_per_electron[
                        j * n_per_event : (j + 1) * n_per_event
                    ]
                    pe_per_electron = per_electron_pe[
                        j * n_per_event : (j + 1) * n_per_event
                    ].sum(axis=1)
                    e_time_list.append(np.repeat(e_time_event, pe_per_electron))
                out_event_id += 1

        area_pattern_arr = np.stack(area_patterns) if area_patterns else np.empty((0, C.N_TOTAL_PMT))

        if with_pe_info and channel_indices_list:
            pe_info = np.zeros(
                int(np.sum([len(x) for x in channel_indices_list])), dtype=C.PE_DTYPE
            )
            pe_info["channel"] = np.concatenate(channel_indices_list)
            pe_info["area"] = np.concatenate(pe_area_list)
            pe_info["event_id"] = np.concatenate(event_id_list)
            if e_time_list:
                pe_info["e_time"] = np.concatenate(e_time_list)
        else:
            pe_info = np.zeros(0, dtype=C.PE_DTYPE)

        return area_pattern_arr, keep_mask, pe_info

    # ------------------------------------------------------------------ #
    # Convenience: full pipeline for one event group.
    # ------------------------------------------------------------------ #
    def simulate_event_group(
        self,
        xy: np.ndarray,
        n_per_event: int,
        e_time_per_electron: np.ndarray | None,
        sim_low_pe: int,
        sim_up_pe: int,
        with_pe_info: bool = True,
    ) -> PatternResult:
        """Complete pipeline for a single multiplicity bucket.

        Parameters
        ----------
        xy : (n_electrons, 2) ; rows are flattened per-event blocks.
        n_per_event : multiplicity (n).
        e_time_per_electron : (n_electrons,) electron arrival time. Pass ``None``
            for the CEvNS path (legacy CEVNS_pattern stores ``e_id`` instead of
            ``e_time``); the caller handles that case after the fact.
        sim_low_pe / sim_up_pe : area-window edges in pe.
        """
        per_e_light = self.per_electron_light(xy)
        light_event, pe_event, per_electron_pe = self.aggregate_per_event(
            per_e_light, n_per_event
        )

        if isinstance(sim_low_pe, (list, tuple, np.ndarray)):
            sim_low_pe = sim_low_pe[0] if len(sim_low_pe) > 0 else 0
        if isinstance(sim_up_pe, (list, tuple, np.ndarray)):
            sim_up_pe = sim_up_pe[0] if len(sim_up_pe) > 0 else float('inf')

        area_pattern, keep_mask, pe_info = self.area_pattern(
            pe_event,
            per_electron_pe,
            e_time_per_electron,
            n_per_event,
            sim_low_pe,
            sim_up_pe,
            with_pe_info=with_pe_info,
        )
        return PatternResult(
            light_pattern=light_event[keep_mask],
            pe_pattern=pe_event[keep_mask],
            area_pattern=area_pattern,
            pe_by_area=area_pattern / self.pe_gain,
            pe_info=pe_info,
            keep_mask=keep_mask,
        )

    # ------------------------------------------------------------------ #
    # Reconstructed-position pattern (no Poisson, no area cut).
    # ------------------------------------------------------------------ #
    def recons_light_pattern(
        self, xy: np.ndarray, total_pe_by_area: np.ndarray
    ) -> np.ndarray:
        """Compute the expected light pattern at reconstructed positions.

        Mirrors ``DE.generate_DE_recon_pattern`` body: scale the per-electron
        LCE light by ``sum(pe_by_area) / single_e_gain``, then by
        ``single_electron_gain_per_channel``.
        """
        recon_light = self.lce.light_pattern(xy)
        scale = total_pe_by_area[:, None] / (
            self.single_electron_gain_per_channel * self.average_efficiency
        )
        return scale * recon_light * self.single_electron_gain_per_channel
