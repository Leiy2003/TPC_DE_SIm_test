"""Configuration objects for RELICS DE simulations.

The legacy code parsed each YAML directly inside ``DESimulation.__init__``,
so the field set drifted between configs (e.g. ``CEVNS_radius_range`` only
existed in ``Single_Electron.yaml``).  This module gives every field a
nested home, validates types at construction time and supports a single
``extends:`` key for YAML inheritance.

Implemented with stdlib :mod:`dataclasses` so the package has no hard
dependency on a specific pydantic major version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml


# --------------------------------------------------------------------------- #
# Detector + electronics
# --------------------------------------------------------------------------- #


@dataclass
class DetectorConfig:
    """Geometric / physical constants of the TPC."""

    radius_mm: float = 139.0
    height_mm: float = 310.0
    fiducial_radius_mm: float = 139.0
    drift_velocity_mm_per_s: float = 0.174e6
    longitudinal_diffusion: float = 12.0
    quenching_factor: float = 50.0

    def __post_init__(self) -> None:
        for fld in (
            "radius_mm",
            "height_mm",
            "fiducial_radius_mm",
            "drift_velocity_mm_per_s",
        ):
            if getattr(self, fld) <= 0:
                raise ValueError(f"DetectorConfig.{fld} must be > 0 (got {getattr(self, fld)!r})")


@dataclass
class ElectronicsConfig:
    """SE / PMT response constants."""

    average_efficiency: float = 0.3326716371190126
    single_electron_pe: float = 30.0
    pe_gain: float = 6e6
    pe_gain_std: float = 2.4e6
    se_gamma_sigma: float = 0.265
    sigma_se_s: float = 190e-9
    sample_dt_s: float = 4e-9
    waveform_length: int = 3500

    @property
    def single_electron_gain_per_channel(self) -> float:
        """Mean expected pe per channel before LCE -- SE pe / avg LCE."""
        if self.average_efficiency <= 0:
            raise ValueError("average_efficiency must be > 0")
        return self.single_electron_pe / self.average_efficiency


# --------------------------------------------------------------------------- #
# Simulation parameters (muon, DE, CEvNS)
# --------------------------------------------------------------------------- #


@dataclass
class SimulationConfig:
    """Parameters that control the toy MC."""

    muon_rate_hz: float = 10.25
    dead_time_s: float = 0.002
    pile_up_gap_s: float = 1500e-9
    diffuse_length_mm: float = 20.0
    interp_points_per_muon: int = 100
    correlation_radius_mm: float = 20.0
    correlation_gamma: float = 3.0
    correlation_look_ahead: int = 20
    sim_low_limit_pe: float = 90.0
    sim_up_limit_pe: float = 300.0
    roi_low_limit_pe: float = 120.0
    roi_up_limit_pe: float = 300.0
    pile_up_orders: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7])
    save_muon_track: bool = False

    def __post_init__(self) -> None:
        if any(n < 2 for n in self.pile_up_orders):
            raise ValueError(
                "pile_up_orders entries must be >= 2 (single-electron events are not pile-up)"
            )
        self.pile_up_orders = sorted(set(int(n) for n in self.pile_up_orders))


# --------------------------------------------------------------------------- #
# Cut analysis
# --------------------------------------------------------------------------- #


@dataclass
class CutsConfig:
    """Coefficients of the area-dependent linear cut in the (pattern, log st_cor) plane."""

    k_st_coefficients_path: Optional[str] = None
    b_coefficients_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# Paths to external data
# --------------------------------------------------------------------------- #


@dataclass
class PathsConfig:
    """Filesystem locations for inputs and outputs.

    Paths can be absolute or relative to the YAML file (resolution is performed
    in :meth:`DESimConfig.resolve_paths`).
    """

    output_dir: str = "outputs/"
    pmt_top: str = "LCE_info/topPMTs.txt"
    pmt_bot: str = "LCE_info/botPMTs.txt"
    lce_value: str = "LCE_info/LCE_value.npy"
    lce_value_true: str = "LCE_info/LCE_value.npy"
    lce_xs: str = "LCE_info/LCE_xs.npy"
    lce_ys: str = "LCE_info/LCE_ys.npy"
    muon_track_dir: str = "muon_track/"
    cevns_e_spectrum: str = "data/e_spectrum_relics.npz"
    position_reconstruction_model: str = "models/CnnRelics.ckpt"
    waveform_classifier_model: Optional[str] = "models/waveform_classifier.pth"


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


T = TypeVar("T")


@dataclass
class DESimConfig:
    """Container for every parameter the pipeline needs.

    Construct via :meth:`from_yaml` or by passing nested dicts directly.
    """

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    electronics: ElectronicsConfig = field(default_factory=ElectronicsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    cuts: CutsConfig = field(default_factory=CutsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    source_yaml: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DESimConfig":
        path = Path(path).resolve()
        merged = _load_with_extends(path)
        merged = _normalise_legacy_keys(merged)
        merged.setdefault("source_yaml", str(path))
        cfg = _build_dataclass(cls, merged)
        cfg.resolve_paths(base_dir=path.parent)
        return cfg

    def resolve_paths(self, base_dir: Path) -> None:
        """Make every path absolute relative to ``base_dir`` (in-place)."""
        for fld in fields(self.paths):
            value = getattr(self.paths, fld.name)
            if value is None:
                continue
            p = Path(value)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            setattr(self.paths, fld.name, str(p))

    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible alias for :func:`dataclasses.asdict`."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Implementation helpers
# --------------------------------------------------------------------------- #

_SECTIONS: Dict[str, Type[Any]] = {
    "detector": DetectorConfig,
    "electronics": ElectronicsConfig,
    "simulation": SimulationConfig,
    "cuts": CutsConfig,
    "paths": PathsConfig,
}


def _build_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Construct a top-level :class:`DESimConfig` from a nested-dict payload."""
    kwargs: Dict[str, Any] = {}
    for fld in fields(cls):
        if fld.name in _SECTIONS and fld.name in data:
            section_cls = _SECTIONS[fld.name]
            section_data = data[fld.name] or {}
            valid_fields = {f.name for f in fields(section_cls)}
            unknown = set(section_data) - valid_fields
            if unknown:
                raise ValueError(
                    f"Unknown field(s) in section {fld.name!r}: {sorted(unknown)}"
                )
            kwargs[fld.name] = section_cls(**section_data)
        elif fld.name in data:
            kwargs[fld.name] = data[fld.name]
    unknown_top = set(data) - {f.name for f in fields(cls)}
    if unknown_top:
        raise ValueError(f"Unknown top-level field(s): {sorted(unknown_top)}")
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with ``override`` merged on top of ``base`` recursively."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_with_extends(path: Path) -> Dict[str, Any]:
    """Recursively resolve a chain of ``extends:`` declarations."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    parent_data = _load_with_extends(parent_path)
    return _deep_merge(parent_data, data)


# Legacy-flat-keyed YAMLs from ``sim_config/`` map onto the nested layout via
# this table.
_LEGACY_KEY_MAP: Dict[str, "tuple[str, str]"] = {
    "CEVNS_radius_range": ("detector", "fiducial_radius_mm"),
    "quenching_factor": ("detector", "quenching_factor"),
    "average_efficiency": ("electronics", "average_efficiency"),
    "single_electron_gain": ("electronics", "single_electron_pe"),
    "pe_gain": ("electronics", "pe_gain"),
    "pe_gain_std": ("electronics", "pe_gain_std"),
    "sigma_se": ("electronics", "sigma_se_s"),
    "muon_rate": ("simulation", "muon_rate_hz"),
    "dead_time": ("simulation", "dead_time_s"),
    "gap": ("simulation", "pile_up_gap_s"),
    "diffuse_length": ("simulation", "diffuse_length_mm"),
    "sim_up_limit": ("simulation", "sim_up_limit_pe"),
    "sim_low_limit": ("simulation", "sim_low_limit_pe"),
    "roi_up_limit": ("simulation", "roi_up_limit_pe"),
    "roi_low_limit": ("simulation", "roi_low_limit_pe"),
    "save_muon_track": ("simulation", "save_muon_track"),
    "output_pattern": ("paths", "output_dir"),
    "pmt_top": ("paths", "pmt_top"),
    "pmt_bot": ("paths", "pmt_bot"),
    "LCE_value": ("paths", "lce_value"),
    "LCE_value_true": ("paths", "lce_value_true"),
    "LCE_xs": ("paths", "lce_xs"),
    "LCE_ys": ("paths", "lce_ys"),
    "muon_track_file": ("paths", "muon_track_dir"),
    "cevns_e_spectrum": ("paths", "cevns_e_spectrum"),
    "position_reconstruction_model": ("paths", "position_reconstruction_model"),
}


def _normalise_legacy_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Promote any flat legacy keys into the nested schema (in-place semantically)."""
    out: Dict[str, Any] = {
        section: dict(data.get(section, {}) or {})
        for section in _SECTIONS
    }
    for legacy_key, (section, fld) in list(_LEGACY_KEY_MAP.items()):
        if legacy_key in data:
            out[section][fld] = data.pop(legacy_key)

    merged_data: Dict[str, Any] = {k: v for k, v in data.items() if k not in _SECTIONS}
    for section in _SECTIONS:
        existing = data.get(section, {}) or {}
        if existing or out[section]:
            merged_data[section] = _deep_merge(existing, out[section])
    return merged_data
