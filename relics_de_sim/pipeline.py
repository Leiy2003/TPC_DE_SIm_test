"""High-level orchestration replacing ``DE.DESimulation``.

A :class:`Pipeline` owns the heavyweight detector objects (LCE map, CNN
position reconstructor) and offers a fluent stage-by-stage API:

    >>> pipe = Pipeline(cfg)
    >>> pipe.load_muon_tracks(file_list)
    >>> pipe.simulate_delayed_electrons()
    >>> pipe.simulate_pile_up_patterns()
    >>> pipe.recon_position_de()
    >>> pipe.simulate_cevns()
    >>> pipe.score()
    >>> pipe.save("outputs/my_run/")

Every method is idempotent and stores its result on the instance.  Stages are
small enough to be unit-testable in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from relics_de_sim import constants as C
from relics_de_sim import io as io_mod
from relics_de_sim.cevns import CEvNSGenerator, CEvNSSpectrum
from relics_de_sim.config import DESimConfig
from relics_de_sim.correlation import space_time_correlation
from relics_de_sim.cuts import pattern_likelihood
from relics_de_sim.delayed_electron import generate_delayed_electrons
from relics_de_sim.geometry import load_pmt_layout
from relics_de_sim.lce import load_lce
from relics_de_sim.muon import (
    compute_dense_muon_points,
    estimate_dead_time_ratio,
    load_muon_files,
)
from relics_de_sim.pattern import PatternResult, PatternSimulator
from relics_de_sim.pileup import group_pile_up
from relics_de_sim.recon import PositionReconstructor

import relics_de_sim.pos_rec_relics10 as posrec

class Pipeline:
    """Orchestrates every stage of the DE/CEvNS simulation."""

    def __init__(
        self,
        config: DESimConfig,
        rng: np.random.Generator | None = None,
        device: str = "auto",
    ) -> None:
        self.cfg = config
        self.rng = rng or np.random.default_rng()
        self.device = device

        # ----- detector objects ------------------------------------------ #
        paths = config.paths
        self.pmt_top, self.pmt_bot = load_pmt_layout(paths.pmt_top, paths.pmt_bot)
        self.lce_nominal, self.lce_truth = load_lce(
            xs_path=paths.lce_xs,
            ys_path=paths.lce_ys,
            value_path=paths.lce_value,
            value_true_path=paths.lce_value_true,
        )
        self.pattern_truth = PatternSimulator(
            self.lce_truth, config.electronics, rng=self.rng
        )
        self.pattern_recon = PatternSimulator(
            self.lce_nominal, config.electronics, rng=self.rng
        )

        self._position_reconstructor: Optional[PositionReconstructor] = None

        # ----- stage outputs ---------------------------------------------- #
        self.merged_muon: Optional[np.ndarray] = None
        self.event_time: Optional[np.ndarray] = None
        self.dense_muon: Optional[np.ndarray] = None
        self.delayed_electron: Optional[np.ndarray] = None
        self.time_range_s: float = 0.0

        self.pile_up_orders: List[int] = list(config.simulation.pile_up_orders)
        self.pile_up_events: List[np.ndarray] = []
        self.pile_up_indices: List[np.ndarray] = []
        self.pile_up_results: List[PatternResult] = []
        self.pile_up_pos_recon: List[np.ndarray] = []
        self.pile_up_recons_light_pattern: List[np.ndarray] = []
        self.pile_up_st_cor: List[np.ndarray] = []
        self.pile_up_pattern_coef: List[np.ndarray] = []

        self.cevns_spectrum: Optional[CEvNSSpectrum] = None
        self.cevns_points: List[np.ndarray] = []
        self.cevns_pe_info: List[np.ndarray] = []

    # ---------------------------------------------------------------- #
    # Lazy helpers
    # ---------------------------------------------------------------- #
    @property
    def position_reconstructor(self) -> PositionReconstructor:
        if self._position_reconstructor is None:
            self._position_reconstructor = PositionReconstructor(
                self.cfg.paths.position_reconstruction_model, device=self.device
            )
        return self._position_reconstructor

    # ---------------------------------------------------------------- #
    # Stage 1: muon track ingestion
    # ---------------------------------------------------------------- #
    def load_muon_tracks(self, file_list: Sequence[str | Path]) -> "Pipeline":
        """Load muon tracks, time-stamp events, build dense interpolation."""
        merged, summary = load_muon_files(
            file_list,
            muon_rate_hz=self.cfg.simulation.muon_rate_hz,
            quenching_factor=self.cfg.detector.quenching_factor,
            rng=self.rng,
        )
        # Recover the assigned event_time (stashed on the summary).
        event_time = summary.event_time  # type: ignore[attr-defined]
        dead_ratio = estimate_dead_time_ratio(self.cfg.simulation.dead_time_s, rng=self.rng)
        summary.dead_time_ratio = dead_ratio

        self.merged_muon = merged
        self.event_time = event_time
        self.time_range_s = summary.time_range_s
        self.dense_muon = compute_dense_muon_points(
            merged,
            event_time,
            dead_time_ratio=dead_ratio,
            interp_num=self.cfg.simulation.interp_points_per_muon,
        )
        return self

    # ---------------------------------------------------------------- #
    # Stage 2: delayed-electron sampling + pile-up grouping
    # ---------------------------------------------------------------- #
    def simulate_delayed_electrons(self) -> "Pipeline":
        if self.dense_muon is None:
            raise RuntimeError("Call load_muon_tracks first.")
        self.delayed_electron = generate_delayed_electrons(
            self.dense_muon,
            dead_time_s=self.cfg.simulation.dead_time_s,
            diffuse_length_mm=self.cfg.simulation.diffuse_length_mm,
            fiducial_radius_mm=self.cfg.detector.radius_mm,
            rng=self.rng,
        )
        events_list, indices_list = group_pile_up(
            self.delayed_electron,
            gap_s=self.cfg.simulation.pile_up_gap_s,
            orders=self.pile_up_orders,
        )
        self.pile_up_events = events_list
        self.pile_up_indices = indices_list
        return self

    # ---------------------------------------------------------------- #
    # Stage 3: pattern simulation for each pile-up multiplicity
    # ---------------------------------------------------------------- #
    def simulate_pile_up_patterns(self) -> "Pipeline":
        if not self.pile_up_events:
            raise RuntimeError("Call simulate_delayed_electrons first.")
        sim = self.cfg.simulation
        self.pile_up_results = []
        sim_low_pe=sim.sim_low_limit_pe
        sim_up_pe=sim.sim_up_limit_pe

        for idx, (n, events) in enumerate(zip(self.pile_up_orders, self.pile_up_events)):
            if len(events) == 0:
                self.pile_up_results.append(_empty_pattern_result(n))
                continue
            xy = np.stack(
                [events["xd"].reshape(-1), events["yd"].reshape(-1)], axis=1
            )
            e_time = events["e_time"].reshape(-1)
            res = self.pattern_truth.simulate_event_group(
                xy=xy,
                n_per_event=n,
                e_time_per_electron=e_time,
                sim_low_pe = sim_low_pe,
                sim_up_pe = sim_up_pe,
            )
            self.pile_up_events[idx] = events[res.keep_mask]
            self.pile_up_results.append(res)
        return self

    # ---------------------------------------------------------------- #
    # Stage 4: position reconstruction + recons-light-pattern
    # ---------------------------------------------------------------- #
    def recon_position_de(self) -> "Pipeline":
        if not self.pile_up_results:
            raise RuntimeError("Call simulate_pile_up_patterns first.")
        recon = self.position_reconstructor
        self.pile_up_pos_recon = []
        self.pile_up_recons_light_pattern = []
        for res in self.pile_up_results:
            if len(res.pe_by_area) == 0:
                self.pile_up_pos_recon.append(np.zeros((0, 2), dtype=np.float32))
                self.pile_up_recons_light_pattern.append(
                    np.zeros((0, C.N_TOTAL_PMT), dtype=np.float64)
                )
                continue
            pos = posrec.position_construction(res.pe_by_area[:, : C.N_TOP_PMT])
            self.pile_up_pos_recon.append(pos)
            recons_light = self.pattern_recon.recons_light_pattern(
                pos.astype(np.float64), total_pe_by_area=res.pe_by_area.sum(axis=1)
            )
            self.pile_up_recons_light_pattern.append(recons_light)
        return self

    # ---------------------------------------------------------------- #
    # Stage 5: scoring (pattern likelihood + space-time correlation)
    # ---------------------------------------------------------------- #
    def score(self) -> "Pipeline":
        if not self.pile_up_recons_light_pattern:
            raise RuntimeError("Call recon_position_de first.")
        self.pile_up_pattern_coef = []
        self.pile_up_st_cor = []
        for events, res, recons_light, pos in zip(
            self.pile_up_events,
            self.pile_up_results,
            self.pile_up_recons_light_pattern,
            self.pile_up_pos_recon,
        ):
            if len(res.pe_by_area) == 0:
                self.pile_up_pattern_coef.append(np.zeros(0))
                self.pile_up_st_cor.append(np.zeros(0))
                continue
            coef = pattern_likelihood(res.pe_by_area, recons_light, n_top=C.N_TOP_PMT)
            event_t = events[:, 0]["e_time"]
            event_xy = pos.astype(np.float64)
            last_muon_id = (
                np.searchsorted(self.event_time, event_t, side="right") - 1
            )
            cor, _ = space_time_correlation(
                event_t,
                event_xy,
                self.dense_muon,
                last_muon_id,
                radius_mm=self.cfg.simulation.correlation_radius_mm,
                gamma=self.cfg.simulation.correlation_gamma,
                look_ahead=self.cfg.simulation.correlation_look_ahead,
                interp_per_muon=self.cfg.simulation.interp_points_per_muon,
            )
            self.pile_up_pattern_coef.append(coef)
            self.pile_up_st_cor.append(cor)
        return self

    # ---------------------------------------------------------------- #
    # CEvNS pipeline
    # ---------------------------------------------------------------- #
    def simulate_cevns(self) -> "Pipeline":
        if self.dense_muon is None:
            raise RuntimeError("Call load_muon_tracks first (CEvNS times use the muon time range).")
        self.cevns_spectrum = CEvNSSpectrum.from_npz(self.cfg.paths.cevns_e_spectrum)
        gen = CEvNSGenerator(
            fiducial_radius_mm=self.cfg.detector.fiducial_radius_mm,
            height_mm=self.cfg.detector.height_mm,
            time_range_s=self.time_range_s,
            spectrum=self.cevns_spectrum,
            rng=self.rng,
        )
        self.cevns_points = gen.sample()
        self._simulate_cevns_patterns()
        self._cevns_position_recon()
        self._score_cevns()
        return self

    # -------------- helpers ------------------------------------------ #
    def _simulate_cevns_patterns(self) -> None:
        sim = self.cfg.simulation
        self.cevns_pe_info = []
        for bin_idx, arr in enumerate(self.cevns_points):
            if len(arr) == 0:
                self.cevns_pe_info.append(np.zeros(0, dtype=C.PE_DTYPE))
                continue
            n = int(arr["num_e"][0])
            xy = np.repeat(
                np.stack([arr["pos_original"]["xd"], arr["pos_original"]["yd"]], axis=1),
                n,
                axis=0,
            )
            res = self.pattern_truth.simulate_event_group(
                xy=xy,
                n_per_event=n,
                e_time_per_electron=None,
                sim_low_pe=sim.sim_low_limit_pe,
                sim_up_pe=sim.sim_up_limit_pe,
            )
            arr_keep = arr[res.keep_mask].copy()
            arr_keep["light_pattern"] = res.light_pattern
            arr_keep["pe_pattern"] = res.pe_pattern
            arr_keep["area_pattern"] = res.area_pattern
            arr_keep["pe_by_area"] = res.pe_by_area
            self.cevns_points[bin_idx] = arr_keep
            self.cevns_pe_info.append(res.pe_info)

    def _cevns_position_recon(self) -> None:
        recon = self.position_reconstructor
        for arr in self.cevns_points:
            if len(arr) == 0:
                continue
            pos = posrec.position_construction(arr["pe_by_area"][:, : C.N_TOP_PMT])
            arr["pos_recon"]["xd"] = pos[:, 0]
            arr["pos_recon"]["yd"] = pos[:, 1]
            recons_light = self.pattern_recon.recons_light_pattern(
                pos.astype(np.float64), total_pe_by_area=arr["pe_by_area"].sum(axis=1)
            )
            arr["recons_light_pattern"] = recons_light

    def _score_cevns(self) -> None:
        for arr in self.cevns_points:
            if len(arr) == 0:
                continue
            cor, _ = space_time_correlation(
                arr["t"],
                np.stack([arr["pos_recon"]["xd"], arr["pos_recon"]["yd"]], axis=1),
                self.dense_muon,
                last_muon_id=np.searchsorted(self.event_time, arr["t"], side="right") - 1,
                radius_mm=self.cfg.simulation.correlation_radius_mm,
                gamma=self.cfg.simulation.correlation_gamma,
                look_ahead=self.cfg.simulation.correlation_look_ahead,
                interp_per_muon=self.cfg.simulation.interp_points_per_muon,
            )
            arr["st_cor"] = cor

    # ---------------------------------------------------------------- #
    # Saving
    # ---------------------------------------------------------------- #
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.pile_up_results:
            io_mod.save_de_result(
                out / "de_result.npz",
                pile_up_orders=self.pile_up_orders,
                pile_up_pe_by_area=[r.pe_by_area for r in self.pile_up_results],
                pile_up_recons_light_pattern=self.pile_up_recons_light_pattern,
                pile_up_pattern_coef=self.pile_up_pattern_coef
                or [np.zeros(0)] * len(self.pile_up_orders),
                pile_up_st_cor=self.pile_up_st_cor
                or [np.zeros(0)] * len(self.pile_up_orders),
                pile_up_pe_info=[r.pe_info for r in self.pile_up_results],
                pile_up_area=[r.pe_by_area.sum(axis=1) for r in self.pile_up_results],
            )
        if self.cevns_points:
            io_mod.save_cevns_result(out / "cevns_result.npz", self.cevns_points)
        io_mod.save_meta(out / "meta.json", self.cfg)
        return out


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _empty_pattern_result(_n: int) -> PatternResult:
    return PatternResult(
        light_pattern=np.zeros((0, C.N_TOTAL_PMT)),
        pe_pattern=np.zeros((0, C.N_TOTAL_PMT)),
        area_pattern=np.zeros((0, C.N_TOTAL_PMT)),
        pe_by_area=np.zeros((0, C.N_TOTAL_PMT)),
        pe_info=np.zeros(0, dtype=C.PE_DTYPE),
        keep_mask=np.zeros(0, dtype=bool),
    )
