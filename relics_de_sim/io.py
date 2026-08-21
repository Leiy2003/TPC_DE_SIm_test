"""Standardised on-disk schema for pipeline outputs.

Each ``Pipeline.save(<dir>)`` call produces a directory containing:

* ``de_result.npz``       -- DE pile-up arrays (one chunk per multiplicity).
* ``cevns_result.npz``    -- CEvNS structured arrays (one chunk per e-bin).
* ``meta.json``           -- the YAML config the run used + provenance.

The legacy ``sim.py``/``Acceptance_test.py``/``*e_cut_wf_gen.py`` each used a
slightly different ad-hoc layout. This module gives them a single schema so
downstream notebooks can rely on a stable contract.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from relics_de_sim.config import DESimConfig


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _config_dump(cfg: DESimConfig) -> Dict[str, Any]:
    """Pydantic v2 dump."""
    return cfg.model_dump()


def save_meta(path: str | Path, cfg: DESimConfig, **extra: Any) -> None:
    """Write a ``meta.json`` describing the run."""
    meta = {
        "config": _config_dump(cfg),
        "git_sha": _git_sha(),
        "extra": extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def save_de_result(
    path: str | Path,
    pile_up_orders: list[int],
    pile_up_pe_by_area: list[np.ndarray],
    pile_up_recons_light_pattern: list[np.ndarray],
    pile_up_pattern_coef: list[np.ndarray],
    pile_up_st_cor: list[np.ndarray],
    pile_up_pe_info: list[np.ndarray],
    pile_up_area: list[np.ndarray],
) -> None:
    """Persist DE pile-up results in a flat ``npz`` file.

    Each array is concatenated across multiplicities and accompanied by an
    ``e_num`` column so the bin assignment is recoverable.
    """
    e_num_per_event = np.concatenate(
        [np.full(len(area), n, dtype=np.uint32) for n, area in zip(pile_up_orders, pile_up_area)]
    )
    np.savez_compressed(
        path,
        pile_up_e_num=e_num_per_event,
        pile_up_pe_by_area=np.concatenate(pile_up_pe_by_area)
        if pile_up_pe_by_area else np.zeros((0, 128)),
        pile_up_recons_light_pattern=np.concatenate(pile_up_recons_light_pattern)
        if pile_up_recons_light_pattern else np.zeros((0, 128)),
        pile_up_pattern_coef=np.concatenate(pile_up_pattern_coef)
        if pile_up_pattern_coef else np.zeros(0),
        pile_up_st_cor=np.concatenate(pile_up_st_cor) if pile_up_st_cor else np.zeros(0),
        pile_up_area=np.concatenate(pile_up_area) if pile_up_area else np.zeros(0),
        pile_up_pe_info=np.concatenate(pile_up_pe_info)
        if pile_up_pe_info else np.zeros(0, dtype=pile_up_pe_info[0].dtype if pile_up_pe_info else "f8"),
        orders=np.array(pile_up_orders, dtype=np.uint32),
    )


def load_de_result(path: str | Path) -> Dict[str, np.ndarray]:
    """Inverse of :func:`save_de_result`."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save_cevns_result(path: str | Path, cevns_points: list[np.ndarray]) -> None:
    """Persist CEvNS structured arrays (one per multiplicity bin)."""
    if cevns_points:
        merged = np.concatenate(cevns_points)
    else:
        merged = np.zeros(0)
    np.savez_compressed(path, cevns_points=merged)


def load_cevns_result(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as f:
        return f["cevns_points"]
