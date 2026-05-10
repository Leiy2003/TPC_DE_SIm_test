#!/usr/bin/env python
"""CEvNS acceptance scan: simulate CEvNS, apply the pattern x ST cut, save waveforms.

Replaces the legacy ``Acceptance_test.py`` and the per-N ``*e_cut_wf_gen.py`` files.
"""

from __future__ import annotations

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

LOG = logging.getLogger("relics-de-sim.run_acceptance")


def main() -> None:
    parser = common_argparser("Run the CEvNS acceptance scan with waveform synthesis.")
    parser.add_argument(
        "--n-electrons",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the scan to the given electron multiplicities (default: all bins in the spectrum).",
    )
    parser.add_argument(
        "--z-mm",
        type=float,
        default=12.0,
        help="Vertical position used to derive the per-event drift sigma. The legacy code "
             "scanned 0..24mm uniformly; pass a single number here to fix it.",
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
        raise SystemExit("Cut coefficients are required: set cuts.k_st_coefficients_path / cuts.b_coefficients_path in the YAML.")
    cut = PatternSTCut.from_npz(
        cfg.cuts.k_st_coefficients_path, cfg.cuts.b_coefficients_path
    )
    LOG.info("Loaded cut: %s", cut)

    wf_params = WaveformParams.from_configs(cfg.detector, cfg.electronics)

    for bin_idx, arr in enumerate(pipe.cevns_points):
        if len(arr) == 0:
            continue
        e_num = int(arr["num_e"][0])
        if args.n_electrons is not None and e_num not in args.n_electrons:
            continue

        st_mask = arr["st_cor"] > 0
        valid = arr[st_mask]
        if len(valid) == 0:
            continue

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
            "e_num=%d: %d/%d events pass the pattern-ST cut (acc=%.3f)",
            e_num,
            int(passes.sum()),
            len(valid),
            float(passes.mean()) if len(valid) else 0.0,
        )
        if not passes.any():
            continue

        pe_info = pipe.cevns_pe_info[bin_idx]
        passing_event_ids = np.where(st_mask)[0][passes]
        z_array = np.full(passing_event_ids.shape, args.z_mm, dtype=np.float64)
        waveforms, stds = synthesize_event_waveforms(
            pe_info, passing_event_ids, z_array, wf_params, rng=rng
        )
        np.savez_compressed(
            out_dir / f"acceptance_e{e_num}.npz",
            area=area[passes],
            st_cor=log_st_cor[passes],
            pattern=pattern_coef[passes],
            e_num=np.full(int(passes.sum()), e_num, dtype=np.uint32),
            z=z_array,
            waveform=waveforms,
            wf_std=stds,
        )

    pipe.save(out_dir)
    LOG.info("Done; outputs in %s", out_dir)


if __name__ == "__main__":
    main()
