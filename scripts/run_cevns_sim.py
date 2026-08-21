#!/usr/bin/env python
"""Run the CEvNS event simulation for one batch of muon files.

Replaces the CEvNS-only path of the legacy ``CEvNS_acceptance.py`` /
``Acceptance_test.py``. The DE pile-up is *not* simulated here -- if you need
both DE and CEvNS in the same run, run :mod:`scripts.run_de_sim` first and pass
its output_dir to :mod:`scripts.run_acceptance`.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import numpy as np

from relics_de_sim import DESimConfig, Pipeline
from scripts._common import (
    build_muon_file_list,
    common_argparser,
    configure_logging,
    resolve_out_dir,
)

LOG = logging.getLogger("relics-de-sim.run_cevns_sim")


def main() -> None:
    parser = common_argparser("Run the CEvNS simulation for one muon batch.")
    args = parser.parse_args()
    configure_logging(args.verbose)

    cfg = DESimConfig.from_yaml(args.config)
    rng = np.random.default_rng(args.seed)
    muon_files = build_muon_file_list(cfg, args.batch_index, args.files_per_batch)
    out_dir = resolve_out_dir(cfg, args.out_dir, args.batch_index)

    LOG.info("Loading muon tracks from %d files", len(muon_files))
    pipe = Pipeline(cfg, rng=rng, device=args.device)
    pipe.load_muon_tracks(muon_files)
    LOG.info("Simulating CEvNS events (spectrum=%s)", cfg.paths.cevns_e_spectrum)
    pipe.simulate_cevns()

    LOG.info("Writing output to %s", out_dir)
    pipe.save(out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
