"""Position reconstruction via a 10x10 CNN trained on the top PMT pattern.

Replaces the legacy ``position_rec.py``. Key fixes:

* Drops the file-based CUDA lock (``cuda_status.txt``). The original code had
  one process per chunk poll a flag file before running inference. This module
  exposes a per-instance :class:`PositionReconstructor` that simply runs torch
  on whichever device the caller asked for, with an optional in-process lock
  for multi-threaded callers.
* Keeps the exact CNN architecture and channel→grid mapping from the legacy
  code so existing ``CnnRelics.ckpt`` checkpoints load unchanged.

``torch`` is imported lazily so :mod:`relics_de_sim` can be imported even on
machines where torch is not installed (e.g. for CI lint jobs that don't run
inference).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from relics_de_sim.constants import N_TOP_PMT, top_pmt_grid_index


def _import_torch() -> Any:
    try:
        import torch  # noqa: WPS433 - intentional local import
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "PyTorch is required for position reconstruction. "
            "Install with `pip install torch` (CPU build is fine)."
        ) from exc
    return torch


def _build_relics_convnet() -> Any:
    """Construct the legacy ``ConvNet`` architecture (lazy-imported)."""
    torch = _import_torch()
    nn = torch.nn

    class _RelicsConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
            )
            self.fc1 = nn.Linear(2 * 2 * 64, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):  # type: ignore[no-untyped-def]
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out

    return _RelicsConvNet()


def pe_to_grid(pe_per_channel: np.ndarray) -> np.ndarray:
    """Map a ``(N, 64)`` pe pattern onto a normalised ``(N, 10, 10)`` grid.

    Reproduces ``position_rec.CNN_output_vec``: zero-fills the 10×10 grid at
    the positions defined by :func:`top_pmt_grid_index`, then row-normalises
    by the total grid sum.
    """
    if pe_per_channel.ndim != 2 or pe_per_channel.shape[1] != N_TOP_PMT:
        raise ValueError(
            f"Expected (N, {N_TOP_PMT}) array, got shape {pe_per_channel.shape}"
        )
    n = pe_per_channel.shape[0]
    grid = np.zeros((n, 100), dtype=np.float64)
    grid[np.arange(n)[:, None], top_pmt_grid_index()] = pe_per_channel
    grid = grid.reshape(n, 10, 10)
    norm = grid.sum(axis=(1, 2), keepdims=True)
    np.divide(grid, norm, out=grid, where=norm > 0)
    return grid


class PositionReconstructor:
    """CNN inference helper for top-PMT pe patterns -> (x, y).

    Parameters
    ----------
    checkpoint_path :
        Path to a ``state_dict()`` compatible torch ``.ckpt`` file.
    device :
        ``"cuda"``, ``"cpu"`` or any valid torch device string.
        ``"auto"`` picks ``cuda`` if available.
    batch_size :
        Maximum number of events fed to the CNN per forward pass.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        batch_size: int = 4096,
    ) -> None:
        self._torch = _import_torch()
        if device == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.device = self._torch.device(device)
        self.batch_size = batch_size
        self._lock = threading.Lock()

        self.model = _build_relics_convnet().to(self.device)
        state = self._torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def __call__(self, pe_per_channel: np.ndarray) -> np.ndarray:
        """Run inference on a ``(N, 64)`` pe pattern. Returns ``(N, 2)``."""
        if len(pe_per_channel) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        torch = self._torch
        grid = pe_to_grid(pe_per_channel).astype(np.float32)
        out = np.empty((grid.shape[0], 2), dtype=np.float32)
        with self._lock, torch.no_grad():
            for start in range(0, len(grid), self.batch_size):
                stop = start + self.batch_size
                tensor = (
                    torch.from_numpy(grid[start:stop]).unsqueeze(1).to(self.device)
                )
                pred = self.model(tensor)
                out[start:stop] = pred.detach().cpu().numpy()
        return out


__all__ = ["PositionReconstructor", "pe_to_grid"]
