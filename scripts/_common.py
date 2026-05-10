"""Shared CLI helpers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

from relics_de_sim.config import DESimConfig


def common_argparser(description: str) -> argparse.ArgumentParser:
    """Return an :class:`argparse.ArgumentParser` with the conventional flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML config (see configs/).",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Integer batch id; multiplied by --files-per-batch to select muon files.",
    )
    parser.add_argument(
        "--files-per-batch",
        type=int,
        default=10,
        help="Number of muon_track.<i>.npy files consumed per run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the output directory (defaults to <config.paths.output_dir>/<batch_index>).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device for the position-reconstruction CNN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def build_muon_file_list(
    cfg: DESimConfig, batch_index: int, files_per_batch: int
) -> List[Path]:
    """Construct the list of muon_track files for a given batch index."""
    muon_dir = Path(cfg.paths.muon_track_dir)
    files: List[Path] = []
    for i in range(files_per_batch):
        idx = batch_index * files_per_batch + i
        files.append(muon_dir / f"muon_track.{idx}.npy")
    return files


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def resolve_out_dir(cfg: DESimConfig, override: Path | None, batch_index: int) -> Path:
    if override is not None:
        return override
    return Path(cfg.paths.output_dir) / f"batch_{batch_index:05d}"
