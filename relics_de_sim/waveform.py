"""S2 waveform synthesis + (optional) classifier wrapper.

Replaces the duplicated ``cevns_waveform_gen`` / ``top_waveform_gen`` helpers
in ``Acceptance_test.py``, ``CEvNS_acceptance.py`` and ``*e_cut_wf_gen.py``.

Waveform synthesis is plain NumPy. The optional CNN classifier defers its
torch import to construction time so the rest of the package can be imported
without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from relics_de_sim.config import DetectorConfig, ElectronicsConfig


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #

@dataclass
class WaveformParams:
    """Compact bundle of constants used inside the inner waveform loop."""

    drift_velocity_mm_per_s: float
    longitudinal_diffusion: float
    sigma_se_s: float
    sample_dt_s: float
    waveform_length: int
    pe_gain: float

    @classmethod
    def from_configs(
        cls, detector: DetectorConfig, electronics: ElectronicsConfig
    ) -> "WaveformParams":
        return cls(
            drift_velocity_mm_per_s=detector.drift_velocity_mm_per_s,
            longitudinal_diffusion=detector.longitudinal_diffusion,
            sigma_se_s=electronics.sigma_se_s,
            sample_dt_s=electronics.sample_dt_s,
            waveform_length=electronics.waveform_length,
            pe_gain=electronics.pe_gain,
        )


def _drift_sigma_s(z_mm: float, drift_velocity_mm_per_s: float, dl: float) -> float:
    """Per-event longitudinal-diffusion sigma in seconds."""
    if z_mm <= 0:
        return 0.0
    t = z_mm / drift_velocity_mm_per_s
    return float(np.sqrt(2.0 * dl * t / drift_velocity_mm_per_s**2))


def synthesize_event_waveform(
    pe_info: np.ndarray,
    z_mm: float,
    params: WaveformParams,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    """Return ``(waveform, std_s)`` for a single event.

    Parameters
    ----------
    pe_info :
        Subset of the global pe array belonging to this event (must contain
        ``area`` and ``e_id`` fields).
    z_mm :
        Vertical position used to scale the drift sigma.
    """
    rng = rng or np.random.default_rng()
    n_pe = len(pe_info)
    waveform = np.zeros(params.waveform_length, dtype=np.float64)
    if n_pe == 0:
        return waveform, 0.0

    sigma = _drift_sigma_s(
        z_mm, params.drift_velocity_mm_per_s, params.longitudinal_diffusion
    )
    pe_e_time = np.zeros(n_pe, dtype=np.float64)
    if sigma > 0:
        for e_id in np.unique(pe_info["e_id"]):
            shift = rng.normal(0.0, sigma)
            pe_e_time[pe_info["e_id"] == e_id] = shift

    se_jitter = rng.normal(0.0, params.sigma_se_s, size=n_pe)
    pe_time = pe_e_time + se_jitter
    pe_time -= pe_time.mean()
    sample_index = (pe_time / params.sample_dt_s).astype(np.int64)
    centre = params.waveform_length // 2
    # np.add.at(waveform, sample_index + centre, pe_info["area"] / params.pe_gain)
    # np.add.at(waveform, sample_index + centre, pe_info["area"].astype(np.float64) / params.pe_gain)
    # np.clip(waveform, 0.0, None, out=waveform)
    # std_s = float(np.std(sample_index) * params.sample_dt_s)

    # 逐个添加，避免 NumPy 的类型检查问题
    gain = float(params.pe_gain)  # 确保是 Python float
    indices = sample_index + centre
    areas = pe_info["area"]
    
    for idx, area in zip(indices, areas):
        if 0 <= idx < params.waveform_length:
            waveform[idx] += float(area) / gain
    
    np.clip(waveform, 0.0, None, out=waveform)
    std_s = float(np.std(sample_index) * params.sample_dt_s)

    return waveform, std_s


def synthesize_event_waveforms(
    pe_info: np.ndarray,
    event_ids: np.ndarray,
    z_mm: np.ndarray,
    params: WaveformParams,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised batch wrapper around :func:`synthesize_event_waveform`."""
    rng = rng or np.random.default_rng()
    n = len(event_ids)
    waveforms = np.zeros((n, params.waveform_length), dtype=np.float64)
    stds = np.zeros(n, dtype=np.float64)
    for k, ev_id in enumerate(event_ids):
        mask = pe_info["event_id"] == ev_id
        waveforms[k], stds[k] = synthesize_event_waveform(
            pe_info[mask], float(z_mm[k]), params, rng=rng
        )
    return waveforms, stds


# --------------------------------------------------------------------------- #
# Classifier (lazy torch import)
# --------------------------------------------------------------------------- #

def _import_torch() -> Any:
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the waveform classifier."
        ) from exc
    return torch


def _build_default_conv1d_net(input_length: int) -> Any:
    """Best-guess 1-D CNN matching the legacy waveform classifier checkpoint."""
    torch = _import_torch()
    nn = torch.nn

    class _Conv1dNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    return _Conv1dNet()


class WaveformClassifier:
    """Optional CNN classifier that scores synthesised waveforms.

    The legacy code in ``3D-cut.ipynb`` defined the network architecture inline;
    this wrapper accepts a custom ``model`` argument so users can plug in any
    nn.Module that matches the checkpoint they have on disk.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model: Any | None = None,
        input_length: int = 3500,
        device: str = "auto",
    ) -> None:
        self._torch = _import_torch()
        if device == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.device = self._torch.device(device)
        self.model = (model or _build_default_conv1d_net(input_length)).to(self.device)
        state = self._torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def __call__(self, waveforms: np.ndarray) -> np.ndarray:
        if len(waveforms) == 0:
            return np.zeros(0, dtype=np.float32)
        torch = self._torch
        with torch.no_grad():
            tensor = (
                torch.from_numpy(waveforms.astype(np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            return self.model(tensor).detach().cpu().numpy()
