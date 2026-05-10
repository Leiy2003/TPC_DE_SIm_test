# Archived legacy code

This folder preserves the pre-`v0.2` codebase verbatim so that the new
:mod:`relics_de_sim` package and the seven curated notebooks under
`notebooks/` can be cross-checked against the original sources. Nothing here
is maintained.

If you need a feature that lives only here, port it into the package and add
a notebook section, rather than running these files directly. They were
written as one-off lab notebooks / scripts and depend on hard-coded paths
(`./file_to_load/`, `cuda_status.txt`, `DE_result_0822/`,
`waveform_result_0824/`, `0824_20000_sigma.npy`) that are not part of the new
contract.

## Contents

### Top-level Python sources

| File | Status | Replacement |
|---|---|---|
| `DE.py` | reference (1 540-line monolithic class) | split across `relics_de_sim/{config,muon,delayed_electron,pileup,pattern,recon,correlation,cevns,cuts,pipeline}.py` |
| `position_rec.py` | reference (CNN + file-based CUDA lock) | `relics_de_sim/recon.py` (no `cuda_status.txt`, configurable device) |
| `Acceptance_test.py` / `CEvNS_acceptance.py` | scratch | `scripts/run_acceptance.py` |
| `acceptance_waveform_gen.sh` | legacy submitter | `scripts/submit_array.sh acceptance ...` |
| `cuda_status.txt` | obsolete file lock | removed (PositionReconstructor uses an in-process lock) |

### `archive/sim_script/`

* `2e.ipynb`–`7e.ipynb` (six near-copies parameterised on electron count) →
  consolidated into `notebooks/05_waveform_classifier.ipynb` (now a
  parameterised notebook).
* `3e_cut_wf_gen.py` … `10e_cut_wf_gen.py` (eight ~400-line copies) →
  consolidated into `scripts/run_waveform_gen.py --n-electrons N`.
* `cevns_cut_wf_gen.py` → `scripts/run_waveform_gen.py` plus
  `scripts/run_cevns_sim.py`.
* `sim.py` + `sim.sh` → `scripts/run_de_sim.py` + `scripts/submit_array.sh de`.
* `wf_gen*.sh` → all replaced by the single `scripts/submit_array.sh waveform`.
* `cuda_check.py` → removed; the CNN now runs without a file lock.
* `DEMO.ipynb` + `Run_test.ipynb` → `notebooks/02_de_simulation.ipynb`.
* `CEvNS_Sim.ipynb` → `notebooks/03_cevns_simulation.ipynb`.
* `Muon_signal_sim.ipynb` → `notebooks/01_muon_signal.ipynb`.
* `Muon_waveform.ipynb`, `new_pattern_gpu_method.ipynb`, `Rate_time.ipynb`,
  `Result.ipynb`, `SE_sim_without_out.ipynb`, `waveform_result.ipynb`,
  `Untitled.ipynb` → kept here as scratch / lab logs (no replacement).

### `archive/sim_config/`

Every legacy YAML lives here for reference; the new
[configs/](../configs) directory carries the curated set. The legacy schema
still loads through `DESimConfig.from_yaml` (legacy flat keys are mapped onto
the nested layout in `relics_de_sim/config.py`).

### Root-level analysis notebooks (moved here)

| File | Status |
|---|---|
| `3D-cut.ipynb`, `3D-cut_multi_process.ipynb` | superseded by `notebooks/04_pattern_st_cor_cut.ipynb` |
| `Acceptance_analyst.ipynb` | superseded by `notebooks/06_acceptance_study.ipynb` |
| `Data_load_reshape.ipynb` | superseded by `notebooks/03_cevns_simulation.ipynb` |
| `Illustration_pattern.ipynb`, `Illustration_waveform.ipynb`, `Ilustration.ipynb` | superseded by `notebooks/illustrations.ipynb` |
| `Ilustration-muon.ipynb` | broken stub (referenced undefined symbols); kept for reference |
| `knn_test.ipynb`, `reshape_test.ipynb` | scratch tests; not maintained |
| `Untitled{,1,2,6}.ipynb` | scratch; not maintained |

### `archive/legacy_results/`

Pre-computed npy / npz outputs that the legacy notebooks emitted to the repo
root (`DE_rate_pattern.npy`, `DE_spectrum_pattern_wf.npy`,
`b_coefficients.npz`, `k_st_coefficients.npz`,
`cevns_pass_pattern_waveform_ratio.npy`, …). These are kept so existing
publication figures can be reproduced without re-running the simulation. The
canonical, regenerable copies of the cut coefficients live under
[`cut_config/`](../cut_config/).
