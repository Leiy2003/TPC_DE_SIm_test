"""Pile-up grouping of delayed electrons.

The legacy code (``DE.generate_Pile_UP_2e``, ``..._3e``, ``..._multi_e``) hand-
unrolls a fixed-window AND-reduction over the adjacency mask once per
multiplicity. Here we replace those six near-copies with one routine,
:func:`group_n_consecutive`, parameterised on ``n``.

Selection rule (preserved):

* Two electrons are "adjacent" if their ``e_time`` differ by less than the
  pile-up gap.
* An n-tuple is a pile-up event iff
  - the (n-1) adjacencies *inside* the window are all True, AND
  - the boundary on the left is False (or absent), AND
  - the boundary on the right is False (or absent).

Equivalent to the legacy ``adjacent_Ne`` chain plus ``start_mask`` / ``end_mask``
bookends. One subtle correction: the legacy ``end_mask`` was built with
``np.insert(arr, -1, True)`` which inserts *before* the last element rather
than at the very end, miscategorising at most one event at the tail of each
run. The implementation here uses ``np.r_[~adj, True]`` so the bookend lands
in the correct position.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def adjacency_mask(electrons: np.ndarray, gap_s: float) -> np.ndarray:
    """Return ``adjacency_mask[i] == True`` iff electron i+1 is within ``gap_s`` of i."""
    if len(electrons) < 2:
        return np.zeros(0, dtype=bool)
    return (electrons["e_time"][1:] - electrons["e_time"][:-1]) < gap_s


def group_n_consecutive(
    electrons: np.ndarray, gap_s: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Find every block of exactly `n` consecutive electrons within ``gap_s``.

    Parameters
    ----------
    electrons :
        Sorted array with at least an ``e_time`` field.
    gap_s :
        Maximum inter-electron separation considered "adjacent".
    n :
        Pile-up multiplicity (``n >= 2``).

    Returns
    -------
    events : np.ndarray, shape (M, n)
        Stacked structured arrays of the participating electrons.
    indices : np.ndarray, shape (M, n)
        Original indices in ``electrons`` for traceability.
    """
    if n < 2:
        raise ValueError("group_n_consecutive requires n >= 2.")
    if len(electrons) < n:
        return (
            np.empty((0, n), dtype=electrons.dtype),
            np.empty((0, n), dtype=np.int64),
        )

    adj = adjacency_mask(electrons, gap_s)
    # adjacency mask of the (n-1) inner edges starting at index i.
    if n == 2:
        inner_runs = adj
    else:
        inner_runs = np.lib.stride_tricks.sliding_window_view(adj, n - 1).all(axis=1)

    # Boundary conditions.
    not_adj_left = np.r_[True, ~adj]      # length len(electrons)
    not_adj_right = np.r_[~adj, True]     # length len(electrons)

    # Window starts at i, spans i..i+n-1. Need:
    #   * inner_runs[i]                    (adjacencies at i, i+1, ..., i+n-2 all True)
    #   * not_adj_left[i]                  (boundary on the left of i is False)
    #   * not_adj_right[i + n - 1]         (boundary on the right of i+n-1 is False)
    starts = not_adj_left[: len(inner_runs)]
    ends = not_adj_right[n - 1 : n - 1 + len(inner_runs)]
    mask = inner_runs & starts & ends

    start_idx = np.flatnonzero(mask)
    indices = start_idx[:, None] + np.arange(n)
    events = electrons[indices]
    return events, indices


def group_pile_up(
    electrons: np.ndarray, gap_s: float, orders: List[int]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Apply :func:`group_n_consecutive` for several multiplicities.

    Returns ``(events_per_n, indices_per_n)`` lists ordered by ``orders``.
    """
    events_list, indices_list = [], []
    for n in orders:
        ev, idx = group_n_consecutive(electrons, gap_s, n)
        events_list.append(ev)
        indices_list.append(idx)
    return events_list, indices_list
