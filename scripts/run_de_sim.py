#!/usr/bin/env python
"""Run the full delayed-electron pile-up simulation for one batch of muon files.

Replaces the legacy ``sim_script/sim.py``. Output schema: see
:mod:`relics_de_sim.io`.
"""

from __future__ import annotations

import logging

import numpy as np

from relics_de_sim import DESimConfig, Pipeline
from scripts._common import (
    build_muon_file_list,
    common_argparser,
    configure_logging,
    resolve_out_dir,
)

LOG = logging.getLogger("relics-de-sim.run_de_sim")


def main() -> None:
    parser = common_argparser("Run the DE pile-up simulation for one muon batch.")
    args = parser.parse_args()
    configure_logging(args.verbose)

    cfg = DESimConfig.from_yaml(args.config)
    rng = np.random.default_rng(args.seed)
    muon_files = build_muon_file_list(cfg, args.batch_index, args.files_per_batch)
    out_dir = resolve_out_dir(cfg, args.out_dir, args.batch_index)

    LOG.info("Loading muon tracks from %d files", len(muon_files))
    pipe = Pipeline(cfg, rng=rng, device=args.device)
    pipe.load_muon_tracks(muon_files)
    LOG.info("Simulating delayed electrons (dead_time=%.4fs)", cfg.simulation.dead_time_s)
    pipe.simulate_delayed_electrons()
    LOG.info("Simulating pile-up patterns")
    pipe.simulate_pile_up_patterns()
    LOG.info("Reconstructing DE positions")
    pipe.recon_position_de()
    LOG.info("Scoring (pattern likelihood + space-time correlation)")
    pipe.score()

    LOG.info("Writing output to %s", out_dir)
    pipe.save(out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
