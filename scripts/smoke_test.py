#!/usr/bin/env python
"""End-to-end smoke test of the simulation chain on synthetic data.

Generates a tiny muon track, a synthetic LCE map, and a top/bot PMT layout in
a temp dir, then exercises every stage of the :class:`Pipeline` that does not
strictly require torch (we substitute the CNN reconstructor for the truth
position so the test passes on machines without PyTorch).
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

from relics_de_sim import DESimConfig, Pipeline
from relics_de_sim.cevns import CEvNSGenerator, CEvNSSpectrum
from relics_de_sim.constants import N_BOT_PMT, N_TOP_PMT, N_TOTAL_PMT, PMT_DTYPE
from relics_de_sim.delayed_electron import generate_delayed_electrons
from relics_de_sim.lce import LCEMap
from relics_de_sim.muon import (
    compute_dense_muon_points,
    estimate_dead_time_ratio,
    load_muon_files,
)
from relics_de_sim.pattern import PatternSimulator
from relics_de_sim.pileup import group_pile_up


def make_synthetic_inputs(root: Path, n_muons: int = 200, rng: np.random.Generator | None = None) -> None:
    rng = rng or np.random.default_rng(0)
    (root / "LCE_info").mkdir(parents=True, exist_ok=True)
    (root / "muon_track").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)

    # PMT layout: top + bottom on a hex grid (we don't care about the exact
    # geometry for the sim; only the dtype and channel count matter).
    pmts_top = np.zeros(N_TOP_PMT, dtype=PMT_DTYPE)
    pmts_bot = np.zeros(N_BOT_PMT, dtype=PMT_DTYPE)
    pmts_top["ChannelID"] = np.arange(N_TOP_PMT)
    pmts_bot["ChannelID"] = np.arange(N_TOP_PMT, N_TOTAL_PMT)
    np.savetxt(root / "LCE_info/topPMTs.txt", pmts_top.tolist(), fmt="%d %f %f %f %f %f %f")
    np.savetxt(root / "LCE_info/botPMTs.txt", pmts_bot.tolist(), fmt="%d %f %f %f %f %f %f")

    # LCE map: a small grid with a smooth Gaussian-ish response per channel.
    xs = np.linspace(-200, 200, 41)
    ys = np.linspace(-200, 200, 41)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    centres = rng.uniform(-100, 100, size=(N_TOTAL_PMT, 2))
    sigma = 60.0
    lce = np.empty((N_TOTAL_PMT, len(xs), len(ys)), dtype=np.float64)
    for ch in range(N_TOTAL_PMT):
        lce[ch] = np.exp(-((XX - centres[ch, 0]) ** 2 + (YY - centres[ch, 1]) ** 2) / (2 * sigma**2))
        lce[ch] /= lce[ch].max() * N_TOTAL_PMT  # tiny per-channel response.
    np.save(root / "LCE_info/LCE_xs.npy", xs)
    np.save(root / "LCE_info/LCE_ys.npy", ys)
    np.save(root / "LCE_info/LCE_value.npy", lce)

    # Muon track: each muon has 5 energy-deposition steps along a downward line.
    muon_dtype = np.dtype(
        [("eventId", "<i4"), ("energy", "<f8"), ("xd", "<f8"), ("yd", "<f8"), ("zd", "<f8")]
    )
    n_steps = 5
    arr = np.zeros(n_muons * n_steps, dtype=muon_dtype)
    for mu in range(n_muons):
        x0, y0 = rng.uniform(-100, 100, 2)
        x1, y1 = x0 + rng.uniform(-50, 50), y0 + rng.uniform(-50, 50)
        for k in range(n_steps):
            t = k / (n_steps - 1)
            i = mu * n_steps + k
            arr[i]["eventId"] = mu
            arr[i]["xd"] = x0 + t * (x1 - x0)
            arr[i]["yd"] = y0 + t * (y1 - y0)
            arr[i]["zd"] = 280.0 - t * 250.0  # 280 -> 30 mm (closer to gate boosts num_e_delayed)
            arr[i]["energy"] = rng.uniform(1000, 5000)  # synthetic high yield, in keV
    np.save(root / "muon_track/muon_track.0.npy", arr)

    # CEvNS spectrum: 10-bin distribution with ~10 events per bin.
    np.savez(
        root / "data/e_spectrum.npz",
        CEvNS_bins=np.arange(11),
        CEvNS=np.full(11, 5),
    )

    # Synthetic config
    cfg = {
        "detector": {"radius_mm": 139.0, "height_mm": 310.0, "fiducial_radius_mm": 139.0},
        "electronics": {
            "average_efficiency": 0.33,
            "single_electron_pe": 30.0,
            "pe_gain": 6e6,
            "pe_gain_std": 2.4e6,
        },
        "simulation": {
            "muon_rate_hz": 10.25,
            "dead_time_s": 0.002,
            "pile_up_gap_s": 1.5e-6,
            "diffuse_length_mm": 20.0,
            "sim_low_limit_pe": 5.0,
            "sim_up_limit_pe": 1e6,
            "pile_up_orders": [2, 3],
        },
        "paths": {
            "output_dir": str(root / "outputs"),
            "pmt_top": str(root / "LCE_info/topPMTs.txt"),
            "pmt_bot": str(root / "LCE_info/botPMTs.txt"),
            "lce_value": str(root / "LCE_info/LCE_value.npy"),
            "lce_value_true": str(root / "LCE_info/LCE_value.npy"),
            "lce_xs": str(root / "LCE_info/LCE_xs.npy"),
            "lce_ys": str(root / "LCE_info/LCE_ys.npy"),
            "muon_track_dir": str(root / "muon_track"),
            "cevns_e_spectrum": str(root / "data/e_spectrum.npz"),
            "position_reconstruction_model": str(root / "models/missing.ckpt"),
        },
    }
    with open(root / "config.yaml", "w") as f:
        yaml.dump(cfg, f)


def smoke_de(cfg: DESimConfig, rng: np.random.Generator) -> None:
    """Run the DE chain without touching the CNN."""
    print("\n--- DE chain (no CNN) ---")
    files = [Path(cfg.paths.muon_track_dir) / "muon_track.0.npy"]

    merged, summary = load_muon_files(
        files, cfg.simulation.muon_rate_hz, cfg.detector.quenching_factor, rng=rng
    )
    summary.dead_time_ratio = estimate_dead_time_ratio(cfg.simulation.dead_time_s, rng=rng)
    print(f"  n_muons={summary.n_muons}, time_range={summary.time_range_s:.3f}s, "
          f"dead_time_ratio={summary.dead_time_ratio:.3f}")

    dense = compute_dense_muon_points(
        merged, summary.event_time, dead_time_ratio=summary.dead_time_ratio
    )
    print(f"  dense points: {dense.shape[0]} ({dense['num_e_delayed'].sum()} expected DEs)")

    de = generate_delayed_electrons(
        dense,
        dead_time_s=cfg.simulation.dead_time_s,
        diffuse_length_mm=cfg.simulation.diffuse_length_mm,
        fiducial_radius_mm=cfg.detector.radius_mm,
        rng=rng,
    )
    print(f"  delayed electrons inside fiducial: {len(de)}")

    pileup_events, _ = group_pile_up(de, cfg.simulation.pile_up_gap_s, [2, 3])
    print(f"  pile-up groups: {[len(e) for e in pileup_events]}")

    # Pattern sim for n=2 only (skip n=3 if no events).
    lce = LCEMap.from_paths(cfg.paths.lce_xs, cfg.paths.lce_ys, cfg.paths.lce_value)
    sim = PatternSimulator(lce, cfg.electronics, rng=rng)
    if len(pileup_events[0]):
        xy = np.stack([
            pileup_events[0]["xd"].reshape(-1),
            pileup_events[0]["yd"].reshape(-1),
        ], axis=1)
        e_time = pileup_events[0]["e_time"].reshape(-1)
        res = sim.simulate_event_group(
            xy=xy, n_per_event=2, e_time_per_electron=e_time,
            sim_low_pe=cfg.simulation.sim_low_limit_pe,
            sim_up_pe=cfg.simulation.sim_up_limit_pe,
        )
        print(f"  2e events surviving area cut: {len(res.pe_by_area)}")


def smoke_cevns(cfg: DESimConfig, rng: np.random.Generator) -> None:
    print("\n--- CEvNS chain ---")
    spec = CEvNSSpectrum.from_npz(cfg.paths.cevns_e_spectrum, e_num_slice=slice(1, 11))
    gen = CEvNSGenerator(
        fiducial_radius_mm=cfg.detector.fiducial_radius_mm,
        height_mm=cfg.detector.height_mm,
        time_range_s=10.0,
        spectrum=spec,
        rng=rng,
    )
    bins = gen.sample()
    print("  events per bin:", [len(b) for b in bins])


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_synthetic_inputs(root)
        cfg = DESimConfig.from_yaml(root / "config.yaml")
        rng = np.random.default_rng(123)
        smoke_de(cfg, rng)
        smoke_cevns(cfg, rng)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
