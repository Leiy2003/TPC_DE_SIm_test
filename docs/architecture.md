# Architecture reference

This document describes the module layout of `relics_de_sim`, the call graph
between the modules, and the on-disk schema for each output type.

## Module map

```
relics_de_sim/
├── __init__.py             re-exports DESimConfig, Pipeline
├── __main__.py             tiny dispatch CLI (`python -m relics_de_sim de ...`)
├── config.py               nested dataclasses + YAML loading with `extends:`
├── constants.py            magic numbers + structured dtypes
├── geometry.py             PMT layout + fiducial helpers
├── lce.py                  LCEMap with batched per-channel interpolation
├── muon.py                 load_muon_files, dense_muon_points, dead-time ratio
├── delayed_electron.py     power-law dt, Gaussian xy diffusion
├── pileup.py               group_n_consecutive (replaces 6 unrolled copies)
├── pattern.py              PatternSimulator (light → pe → area patterns)
├── recon.py                PositionReconstructor (CNN, lazy torch import)
├── correlation.py          space_time_correlation (vectorised batches)
├── cevns.py                CEvNSGenerator + CEvNSSpectrum
├── waveform.py             S2 waveform synthesis + (optional) CNN classifier
├── cuts.py                 PatternSTCut, fit_pattern_st_cut, pattern_likelihood
├── pipeline.py             Pipeline (replaces DESimulation)
└── io.py                   standardised npz schema + meta.json sibling
```

### Call graph

```
Pipeline.load_muon_tracks
  └─ muon.load_muon_files
  └─ muon.estimate_dead_time_ratio
  └─ muon.compute_dense_muon_points

Pipeline.simulate_delayed_electrons
  └─ delayed_electron.generate_delayed_electrons
  └─ pileup.group_pile_up

Pipeline.simulate_pile_up_patterns
  └─ pattern.PatternSimulator.simulate_event_group  ─── for each multiplicity

Pipeline.recon_position_de
  └─ recon.PositionReconstructor                    ─── (lazy torch import)
  └─ pattern.PatternSimulator.recons_light_pattern

Pipeline.score
  └─ cuts.pattern_likelihood
  └─ correlation.space_time_correlation

Pipeline.simulate_cevns
  └─ cevns.CEvNSGenerator.sample
  └─ pattern.PatternSimulator.simulate_event_group  (per multiplicity bin)
  └─ recon.PositionReconstructor                    (per multiplicity bin)
  └─ correlation.space_time_correlation             (per multiplicity bin)

Pipeline.save
  └─ io.save_de_result, save_cevns_result, save_meta
```

Each method on `Pipeline` is idempotent and stores its result in an attribute
of the same name (e.g. `pipe.delayed_electron`, `pipe.pile_up_results`).
Stages can be run independently from a notebook for debugging.

## On-disk schemas

### `de_result.npz`

Produced by `scripts/run_de_sim.py` (and `Pipeline.save`):

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `pile_up_e_num`               | (M,)         | uint32 | electron multiplicity per pile-up event |
| `pile_up_pe_by_area`          | (M, 128)     | float64 | per-channel `area / pe_gain` |
| `pile_up_recons_light_pattern`| (M, 128)     | float64 | expected pe per channel from CNN-reconstructed (x, y) |
| `pile_up_pattern_coef`        | (M,)         | float64 | log-likelihood of `pe_by_area` given `recons_light_pattern` |
| `pile_up_st_cor`              | (M,)         | float64 | scalar space-time correlation against last `look_ahead` muons |
| `pile_up_area`                | (M,)         | float64 | total S2 area (Σ over channels) |
| `pile_up_pe_info`             | (P,)         | structured (`PE_DTYPE`) | per-pe long-format record (channel, area, e_id, e_time, event_id) |
| `orders`                      | (n_orders,)  | uint32 | the multiplicities present in this run |

Where `M = Σ_n M_n` (total events across multiplicities) and `P` is the total
number of pe.

### `cevns_result.npz`

Produced by `scripts/run_cevns_sim.py`:

| Field | Shape | Dtype |
|---|---|---|
| `cevns_points` | (M_cevns,) | structured (`CEVNS_DTYPE`) |

The structured dtype carries: `cevnsID`, `pos_original` (x/y/z), `t`, `num_e`,
`light_pattern[128]`, `pe_pattern[128]`, `area_pattern[128]`,
`pe_by_area[128]`, `pos_recon` (x/y), `recons_light_pattern[128]`, `st_cor`.

### `acceptance_e<N>.npz`

Produced per multiplicity by `scripts/run_acceptance.py`:

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `area`     | (K,) | float64 | total S2 area for events passing the cut |
| `st_cor`   | (K,) | float64 | log(st_cor) |
| `pattern`  | (K,) | float64 | pattern likelihood |
| `e_num`    | (K,) | uint32  | electron multiplicity (constant within file) |
| `z`        | (K,) | float64 | drift depth used for waveform synth |
| `waveform` | (K, L) | float64 | synthesised S2 waveform (L = `electronics.waveform_length`) |
| `wf_std`   | (K,) | float64 | per-event waveform std-dev (s) |

### `meta.json`

JSON sibling that captures provenance:

```json
{
  "config": {
    "detector": {...}, "electronics": {...},
    "simulation": {...}, "cuts": {...}, "paths": {...},
    "source_yaml": "/path/to/de_sim.yaml"
  },
  "git_sha": "<commit>",
  "extra": {}
}
```

### Cut coefficients (`cut_config/k_st_coefficients.npz`, `b_coefficients.npz`)

Each file is a single `arr_0` array of length 2:

* `k_st_coefficients.npz["arr_0"] = [k_st_a, k_st_b]`
* `b_coefficients.npz["arr_0"] = [b_a, b_b]`

The cut is

```
pattern - (k_st_a * area + k_st_b) * log(st_cor) > b_a * area + b_b
```

## Dtypes

All structured dtypes are defined in `relics_de_sim/constants.py`:

* `PMT_DTYPE` — PMT layout text-file rows (`ChannelID`, `x`, `y`, `z`, …).
* `DENSE_MUON_DTYPE` — per-step muon data (`eventId`, `xd`, `yd`, `zd`, …).
* `DELAYED_ELECTRON_DTYPE` — `(eventId, xd, yd, muon_time, e_time)`.
* `PE_DTYPE` — `(event_id, area, channel, e_time, e_id)` for pe-level info.
* `CEVNS_DTYPE` — full CEvNS event record (see `cevns.py`).

## Refactor highlights

* **Pile-up**: 6 hand-unrolled functions collapse into the single
  `pileup.group_n_consecutive(n)` (with a small boundary-handling fix; see
  the docstring).
* **Patterns**: 5 near-identical methods collapse into the single
  `pattern.PatternSimulator` exposing `per_electron_light`,
  `aggregate_per_event`, `area_pattern`, `simulate_event_group` and
  `recons_light_pattern`.
* **CUDA lock**: removed. `recon.PositionReconstructor` runs on whichever
  device the caller asked for and uses an in-process `threading.Lock`
  for multi-threaded callers.
* **YAML**: 5 drift-prone configs collapse to a `_base.yaml` + 5 small
  topic configs that only override what differs.
* **CLI**: 8 per-N waveform scripts collapse into one
  `scripts/run_waveform_gen.py --n-electrons N`. All `*.sh` submitters
  collapse into `scripts/submit_array.sh`.

See `archive/NOTE.md` for the legacy → new mapping at file granularity.
