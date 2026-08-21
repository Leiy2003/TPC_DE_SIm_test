"""Detector-level constants and shared dtypes used across modules.

These were previously scattered through DE.py / position_rec.py / Acceptance_test.py
as magic numbers. Centralising them makes the physics assumptions explicit.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# PMT array
# --------------------------------------------------------------------------- #

N_TOP_PMT: int = 28
N_BOT_PMT: int = 28
N_TOTAL_PMT: int = N_TOP_PMT + N_BOT_PMT  # 56

# Mapping from the linear top-PMT index (0..63) to a (row, col) coordinate on the
# 10x10 grid expected by the CNN position reconstructor.  Reproduced verbatim
# from the legacy ``position_rec.CNN_output_vec`` so that the CNN checkpoint is
# bit-compatible.
_TOP_PMT_LAYERS = [
    [18, 17, 16, 15, 14],
    [19, 40, 39, 38, 37, 13],
    [20, 41, 56, 55, 54, 36, 12],
    [21, 42, 57, 4, 5, 6, 53, 35, 11],
    [22, 43, 58, 0, 1, 2, 3, 52, 34, 10],
    [23, 44, 59, 7, 8, 9, 63, 51, 33],
    [24, 45, 60, 61, 62, 50, 32],
    [25, 46, 47, 48, 49, 31],
    [26, 27, 28, 29, 30],
]


def top_pmt_grid_index() -> np.ndarray:
    """Return a length-64 array mapping channel id -> position on a 10x10 grid.

    Output convention matches the legacy code: ``index[ch] = row * 10 + col``.
    """
    index = np.empty(N_TOP_PMT, dtype=np.int64)
    for row, layer in enumerate(_TOP_PMT_LAYERS):
        col_start = (10 - len(layer)) // 2
        for k, ch in enumerate(layer):
            index[ch] = row * 10 + (col_start + k)
    return index


# --------------------------------------------------------------------------- #
# Numpy dtypes shared across modules
# --------------------------------------------------------------------------- #

PMT_DTYPE = np.dtype(
    [
        ("ChannelID", "<i4"),
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("rot_x", "<f8"),
        ("rot_y", "<f8"),
        ("rot_z", "<f8"),
    ]
)

DENSE_MUON_DTYPE = np.dtype(
    [
        ("eventId", "<i4"),
        ("energy", "<f8"),
        ("xd", "<f8"),
        ("yd", "<f8"),
        ("zd", "<f8"),
        ("muon_time", "<f8"),
        ("num_e", "<u4"),
        ("num_e_delayed", "<u4"),
    ]
)

DELAYED_ELECTRON_DTYPE = np.dtype(
    [
        ("eventId", "<i4"),
        ("xd", "<f8"),
        ("yd", "<f8"),
        ("muon_time", "<f8"),
        ("e_time", "<f8"),
    ]
)

PE_DTYPE = np.dtype(
    [
        ("event_id", "uint64"),
        ("area", "<f8"),
        ("channel", "uint16"),
        ("e_time", "<f8"),
        ("e_id", "uint32"),
    ]
)

CEVNS_DTYPE = np.dtype(
    [
        ("cevnsID", "<i4"),
        ("pos_original", np.dtype([("xd", "<f8"), ("yd", "<f8"), ("zd", "<f8")])),
        ("t", "<f8"),
        ("num_e", "<f8"),
        ("light_pattern", "<f8", (N_TOTAL_PMT,)),
        ("pe_pattern", "uint16", (N_TOTAL_PMT,)),
        ("area_pattern", "<f8", (N_TOTAL_PMT,)),
        ("pe_by_area", "<f8", (N_TOTAL_PMT,)),
        ("pos_recon", np.dtype([("xd", "<f8"), ("yd", "<f8")])),
        ("recons_light_pattern", "<f8", (N_TOTAL_PMT,)),
        ("st_cor", "<f8"),
    ]
)


# --------------------------------------------------------------------------- #
# DE-specific numerical defaults (legacy)
# --------------------------------------------------------------------------- #

# Range used to sample the dead-time fraction (i.e. the share of the natural
# DE arrival-time tail that survives the per-event dead window). Matches the
# call ``rndm(0.002, 2, g=-0.1, size=int(1e7))`` in DE.py.
DEAD_TIME_RATIO_TMIN_S: float = 0.002
DEAD_TIME_RATIO_TMAX_S: float = 2.0
DEAD_TIME_RATIO_GAMMA: float = -0.1
DEAD_TIME_RATIO_NSAMPLE: int = int(1e7)

# Same power law used to draw individual delayed-electron arrival times.
DELAY_TIME_TMAX_S: float = 2.0
DELAY_TIME_GAMMA: float = -0.1

# Affine model of the surviving-fraction-vs-depth (DE.py lines 1224-1225).
DELAY_RATIO_Z_INTERCEPT_MM: float = 100.0
DELAY_RATIO_Z_SLOPE: float = -1.7
DELAY_RATIO_OFFSET: float = 0.02
DELAY_RATIO_FLOOR: float = 0.001
DELAY_RATIO_DENOMINATOR: float = 100.0
