#!/usr/bin/env python
"""Fit the area-dependent (pattern, log st_cor) cut from a DE batch's outputs.

Aggregates one or more ``de_result.npz`` files produced by
:mod:`scripts.run_de_sim` and fits the four-coefficient :class:`PatternSTCut`
via :func:`relics_de_sim.cuts.fit_pattern_st_cut`.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import numpy as np

from relics_de_sim.cuts import PatternSTCut, fit_pattern_st_cut, pattern_likelihood
from relics_de_sim.io import load_de_result

LOG = logging.getLogger("relics-de-sim.fit_cuts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the pattern x ST cut from DE batches.")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more de_result.npz files (typically the output of run_de_sim).",
    )
    parser.add_argument(
        "--k-st-out",
        type=Path,
        default=Path("cut_config/k_st_coefficients.npz"),
        help="Output path for the slope coefficients.",
    )
    parser.add_argument(
        "--b-out",
        type=Path,
        default=Path("cut_config/b_coefficients.npz"),
        help="Output path for the intercept coefficients.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Per-area-bin upper-tail quantile used to set the threshold (default: 0.99).",
    )
    parser.add_argument(
        "--n-area-bins",
        type=int,
        default=11,
        help="Number of area bins used to fit slope/intercept vs area (default: 11).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    pattern_list, st_cor_list, area_list = [], [], []
    for inp in args.inputs:
        data = load_de_result(inp)
        pattern_list.append(data["pile_up_pattern_coef"])
        st_cor_list.append(np.log(np.where(data["pile_up_st_cor"] > 0, data["pile_up_st_cor"], 1e-300)))
        area_list.append(data["pile_up_area"])
    pattern = np.concatenate(pattern_list)
    st_cor = np.concatenate(st_cor_list)
    area = np.concatenate(area_list)
    LOG.info("Aggregated %d events across %d files.", len(area), len(args.inputs))

    bins = np.linspace(area.min(), area.max(), args.n_area_bins)
    cut, diagnostics = fit_pattern_st_cut(
        area, pattern, st_cor, area_bins=bins, quantile=args.quantile
    )
    LOG.info("Fitted cut: %s", cut)
    cut.to_npz(args.k_st_out, args.b_out)
    print(
        f"Saved coefficients:\n"
        f"  k_st = {[cut.k_st_a, cut.k_st_b]} -> {args.k_st_out}\n"
        f"  b    = {[cut.b_a, cut.b_b]} -> {args.b_out}"
    )


if __name__ == "__main__":
    main()
