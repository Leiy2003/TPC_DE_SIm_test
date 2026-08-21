#!/usr/bin/env python
"""Synthesise S2 waveforms for events surviving the pattern x ST cut.

Single parameterised replacement for the eight legacy ``3e_cut_wf_gen.py`` …
``10e_cut_wf_gen.py`` files.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from relics_de_sim import DESimConfig, Pipeline
from relics_de_sim.cuts import (
    PatternSTCut,
    calculate_poisson_log_likelihood,
)
from relics_de_sim.waveform import WaveformParams, synthesize_event_waveforms
from scripts._common import (
    build_muon_file_list,
    common_argparser,
    configure_logging,
    resolve_out_dir,
)

LOG = logging.getLogger("relics-de-sim.run_waveform_gen")


def main() -> None:
    parser = common_argparser("Generate per-event S2 waveforms after the pattern cut.")
    parser.add_argument(
        "--n-electrons",
        type=int,
        required=True,
        help="Single electron multiplicity to generate waveforms for (e.g. 5).",
    )
    parser.add_argument(
        "--z-mm",
        type=float,
        default=None,
        help="Fixed z (mm) for the drift-diffusion sigma. If omitted, z is sampled uniformly in [0, 24].",
    )
    args = parser.parse_args()
    configure_logging(args.verbose)

    cfg = DESimConfig.from_yaml(args.config)
    rng = np.random.default_rng(args.seed)
    muon_files = build_muon_file_list(cfg, args.batch_index, args.files_per_batch)
    out_dir = resolve_out_dir(cfg, args.out_dir, args.batch_index)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(cfg, rng=rng, device=args.device)
    pipe.load_muon_tracks(muon_files)
    pipe.simulate_cevns()

    if cfg.cuts.k_st_coefficients_path is None or cfg.cuts.b_coefficients_path is None:
        raise SystemExit("Cut coefficients are required for the waveform generation step.")
    cut = PatternSTCut.from_npz(
        cfg.cuts.k_st_coefficients_path, cfg.cuts.b_coefficients_path
    )
    wf_params = WaveformParams.from_configs(cfg.detector, cfg.electronics)

    target_n = args.n_electrons
    bin_idx = next(
        (
            i
            for i, arr in enumerate(pipe.cevns_points)
            if len(arr) and int(arr["num_e"][0]) == target_n
        ),
        None,
    )
    if bin_idx is None:
        raise SystemExit(f"No CEvNS events found with n_electrons == {target_n} in this batch.")

    arr = pipe.cevns_points[bin_idx]
    pe_info = pipe.cevns_pe_info[bin_idx]

    st_mask = arr["st_cor"] > 0
    valid = arr[st_mask]
    if len(valid) == 0:
        LOG.warning("No CEvNS events with positive st_cor for n=%d; nothing to write.", target_n)
        return

    area = valid["pe_by_area"].sum(axis=1)
    log_st_cor = np.log(valid["st_cor"])
    pattern_coef = np.sum(
        calculate_poisson_log_likelihood(
            valid["pe_by_area"][:, :64],
            valid["recons_light_pattern"][:, :64],
        ),
        axis=1,
    )
    passes = cut.passes(pattern_coef, log_st_cor, area)
    LOG.info(
        "n=%d: %d / %d events pass the pattern-ST cut (acc=%.3f)",
        target_n,
        int(passes.sum()),
        len(valid),
        float(passes.mean()),
    )
    if not passes.any():
        return

    passing_event_ids = np.where(st_mask)[0][passes]
    if args.z_mm is None:
        z_array = np.linspace(0.0, 24.0, int(passes.sum()))
    else:
        z_array = np.full(int(passes.sum()), args.z_mm, dtype=np.float64)

    waveforms, stds = synthesize_event_waveforms(
        pe_info, passing_event_ids, z_array, wf_params, rng=rng
    )
    out_path = out_dir / f"waveforms_e{target_n}.npz"
    np.savez_compressed(
        out_path,
        z=z_array,
        wf_pass_pattern=waveforms,
        wf_std=stds,
        area=area[passes],
        st_cor=log_st_cor[passes],
        pattern=pattern_coef[passes],
        e_num=np.full(int(passes.sum()), target_n, dtype=np.uint32),
    )
    LOG.info("Wrote %d waveforms to %s", int(passes.sum()), out_path)


if __name__ == "__main__":
    main()
