"""RELICS Delayed Electron simulation package.

Top-level entry points:
    DESimConfig: pydantic-validated YAML config (see relics_de_sim.config).
    Pipeline:    high-level orchestrator that chains every physics stage.

Example:
    >>> from relics_de_sim import DESimConfig, Pipeline
    >>> cfg = DESimConfig.from_yaml("configs/de_sim.yaml")
    >>> pipe = Pipeline(cfg)
    >>> pipe.load_muon_tracks(["muon_track/muon_track.0.npy"])
    >>> pipe.simulate_delayed_electrons()
    >>> pipe.simulate_pile_up_patterns()
    >>> pipe.score()
    >>> pipe.save("outputs/de_sim/run_001/")
"""

from relics_de_sim.config import DESimConfig
from relics_de_sim.pipeline import Pipeline

__all__ = ["DESimConfig", "Pipeline", "__version__"]

__version__ = "0.2.0"
