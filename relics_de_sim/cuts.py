"""Pattern × space-time-correlation 2-D cut.

Replaces the bookkeeping that lives in ``3D-cut.ipynb`` /
``Acceptance_test.py`` / ``cevns_cut_wf_gen.py``. The cut takes the form::

    score(area)        = pattern_likelihood
                       - (k_st_a * area + k_st_b) * log_st_cor
    threshold(area)    = b_a * area + b_b
    keep := score(area) > threshold(area)

The two ``(a, b)`` coefficient pairs are usually fitted on a high-statistics
DE sample; we expose ``fit_pattern_st_cut`` to do that fit and
``apply_pattern_st_cut`` for inference. Loading and saving the legacy
``k_st_coefficients.npz`` / ``b_coefficients.npz`` files is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln


@dataclass
class PatternSTCut:
    """Linear-in-area cut coefficients."""

    k_st_a: float
    k_st_b: float
    b_a: float
    b_b: float

    # ------------------------------------------------------------------ #
    @classmethod
    def from_npz(
        cls, k_st_path: str | Path, b_path: str | Path
    ) -> "PatternSTCut":
        k_st = np.load(k_st_path)["arr_0"]
        b = np.load(b_path)["arr_0"]
        if len(k_st) != 2 or len(b) != 2:
            raise ValueError("Coefficient files must contain length-2 arrays under 'arr_0'.")
        return cls(k_st_a=float(k_st[0]), k_st_b=float(k_st[1]),
                   b_a=float(b[0]), b_b=float(b[1]))

    def to_npz(self, k_st_path: str | Path, b_path: str | Path) -> None:
        Path(k_st_path).parent.mkdir(parents=True, exist_ok=True)
        Path(b_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(k_st_path, arr_0=np.array([self.k_st_a, self.k_st_b]))
        np.savez(b_path, arr_0=np.array([self.b_a, self.b_b]))

    # ------------------------------------------------------------------ #
    def slope(self, area: np.ndarray) -> np.ndarray:
        return self.k_st_a * area + self.k_st_b

    def intercept(self, area: np.ndarray) -> np.ndarray:
        return self.b_a * area + self.b_b

    def score(self, pattern: np.ndarray, log_st_cor: np.ndarray, area: np.ndarray) -> np.ndarray:
        return pattern - self.slope(area) * log_st_cor

    def threshold(self, area: np.ndarray) -> np.ndarray:
        return self.intercept(area)

    def passes(
        self, pattern: np.ndarray, log_st_cor: np.ndarray, area: np.ndarray
    ) -> np.ndarray:
        return self.score(pattern, log_st_cor, area) > self.threshold(area)


def calculate_poisson_log_likelihood(
    observed: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    """Element-wise log-Poisson PMF, vectorised."""
    return -expected + observed * np.log(np.where(expected > 0, expected, 1.0)) - gammaln(observed + 1)


def pattern_likelihood(
    pe_by_area: np.ndarray,
    recons_light_pattern: np.ndarray,
    n_top: int = 64,
) -> np.ndarray:
    """Sum of log-Poisson over the top ``n_top`` channels."""
    return np.sum(
        calculate_poisson_log_likelihood(
            pe_by_area[:, :n_top], recons_light_pattern[:, :n_top]
        ),
        axis=1,
    )


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def _line(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def fit_pattern_st_cut(
    area: np.ndarray,
    pattern: np.ndarray,
    log_st_cor: np.ndarray,
    *,
    area_bins: Optional[np.ndarray] = None,
    quantile: float = 0.99,
) -> Tuple[PatternSTCut, dict]:
    """Fit a linear-in-area cut to a labelled DE distribution.

    For each ``area`` bin we fit a 1-D linear model
    ``pattern = slope_bin * log_st_cor + intercept_bin`` to the bin's data, then
    fit a 1-D linear model of ``slope_bin`` and ``intercept_bin`` vs the bin
    centre. Returns the four-coefficient :class:`PatternSTCut` plus an
    diagnostics dict with the per-bin fit results.
    """
    if area_bins is None:
        area_bins = np.linspace(area.min(), area.max(), 11)
    bin_centres = 0.5 * (area_bins[:-1] + area_bins[1:])

    slopes, intercepts = [], []
    for lo, hi in zip(area_bins[:-1], area_bins[1:]):
        mask = (area >= lo) & (area < hi)
        if mask.sum() < 10:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue
        x = log_st_cor[mask]
        y = pattern[mask]
        try:
            (slope, intercept), _ = curve_fit(_line, x, y)
        except RuntimeError:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue
        # Use a high quantile of the residual to set the threshold conservatively.
        pred = _line(x, slope, intercept)
        residual = y - pred
        intercept = intercept + np.quantile(residual, 1.0 - quantile)
        slopes.append(slope)
        intercepts.append(intercept)
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    valid = np.isfinite(slopes) & np.isfinite(intercepts)
    if valid.sum() < 2:
        raise RuntimeError("Not enough valid area bins for the global fit.")
    (k_st_a, k_st_b), _ = curve_fit(_line, bin_centres[valid], slopes[valid])
    (b_a, b_b), _ = curve_fit(_line, bin_centres[valid], intercepts[valid])

    cut = PatternSTCut(
        k_st_a=float(k_st_a),
        k_st_b=float(k_st_b),
        b_a=float(b_a),
        b_b=float(b_b),
    )
    diagnostics = {
        "area_bins": area_bins,
        "bin_centres": bin_centres,
        "slopes": slopes,
        "intercepts": intercepts,
    }
    return cut, diagnostics
