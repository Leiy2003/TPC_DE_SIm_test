"""Re-generate the canonical notebooks under ``notebooks/``.

Each notebook is described as a flat list of (kind, source) tuples. We use
:mod:`nbformat` to assemble them rather than maintaining hand-edited JSON.

Run from the repo root:

    python notebooks/_build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent

PREAMBLE = """\
# Auto-discover the package even if the notebook is launched from outside the repo.
import sys, os
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 110
"""


def _markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def _save(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    out = NOTEBOOKS_DIR / name
    nbf.write(nb, out)
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- #
# 01 - Muon signal
# --------------------------------------------------------------------------- #

def build_01() -> None:
    cells = [
        _markdown(
            """
            # 01 · Muon S2 signal walk-through

            This notebook reproduces the per-channel **muon S2** characterisation that
            used to live in `sim_script/Muon_signal_sim.ipynb`. We

            1. load a small muon-track sample,
            2. interpolate dense energy-deposition points along each track,
            3. build the **expected S2 light pattern** for the top PMT array,
            4. compute the per-channel pe rate and mean charge.

            All paths come from `configs/muon_only.yaml`; edit that file to point at your
            local detector data before running.
            """
        ),
        _code(PREAMBLE),
        _code(
            """
            from relics_de_sim import DESimConfig, Pipeline
            from relics_de_sim.muon import compute_dense_muon_points

            cfg = DESimConfig.from_yaml('configs/muon_only.yaml')
            print('muon_track_dir:', cfg.paths.muon_track_dir)
            print('output_dir:', cfg.paths.output_dir)
            """
        ),
        _markdown(
            """
            ## 1. Load a small muon batch

            We load the first few `muon_track.<i>.npy` files. Adjust ``files_per_batch``
            to scale the simulation horizon.
            """
        ),
        _code(
            """
            from pathlib import Path
            files_per_batch = 2
            muon_files = [
                Path(cfg.paths.muon_track_dir) / f'muon_track.{i}.npy'
                for i in range(files_per_batch)
            ]
            pipe = Pipeline(cfg, rng=np.random.default_rng(0))
            pipe.load_muon_tracks(muon_files)
            print(f'n muons: {len(pipe.event_time)}')
            print(f'simulated wall-clock: {pipe.time_range_s:.2f} s')
            print(f'dense muon points:    {pipe.dense_muon.shape[0]}')
            """
        ),
        _markdown(
            """
            ## 2. Inspect a single muon track

            The dense interpolation is the input to every downstream stage.
            """
        ),
        _code(
            """
            ev_id = 0
            mask = pipe.dense_muon['eventId'] == ev_id
            pts = pipe.dense_muon[mask]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(pts['xd'], pts['yd'], c=pts['zd'], cmap='viridis', s=4)
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
            ax.set_title(f'Muon track {ev_id} (color = z [mm])')
            plt.colorbar(ax.collections[0], ax=ax, label='z [mm]')
            plt.show()
            """
        ),
        _markdown(
            """
            ## 3. Per-event expected light pattern

            The `PatternSimulator` exposes the LCE × SE-gain product directly.  We use
            it here without applying the pile-up grouping or the area window.
            """
        ),
        _code(
            """
            from relics_de_sim.pattern import PatternSimulator
            from relics_de_sim.lce import LCEMap

            xy = np.stack([pts['xd'], pts['yd']], axis=1)
            light_per_e = pipe.pattern_truth.per_electron_light(xy)
            print('light_per_e shape:', light_per_e.shape)

            channel_mean = light_per_e.mean(axis=0)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(np.arange(64), channel_mean[:64], label='top')
            ax.bar(np.arange(64, 128), channel_mean[64:], color='C3', label='bottom')
            ax.set_xlabel('PMT channel'); ax.set_ylabel('<pe / electron>')
            ax.set_title('Mean expected pe per channel for one muon track')
            ax.legend()
            plt.show()
            """
        ),
        _markdown(
            """
            ## 4. Save metadata

            Everything we computed lives in memory; we typically persist the per-event
            light pattern alongside the muon summary.  See `scripts/run_de_sim.py` for
            the canonical save path.
            """
        ),
    ]
    _save("01_muon_signal.ipynb", cells)


# --------------------------------------------------------------------------- #
# 02 - DE simulation walk-through
# --------------------------------------------------------------------------- #

def build_02() -> None:
    cells = [
        _markdown(
            """
            # 02 · Delayed-electron pile-up walk-through

            This is the canonical end-to-end DE notebook. It reproduces the contents of
            the legacy `DEMO.ipynb` / `Run_test.ipynb` on top of the refactored
            :class:`relics_de_sim.Pipeline`.

            Stages:

            1. Muon tracks  → dense interpolation
            2. Delayed electrons (power-law dt + Gaussian xy)
            3. Pile-up grouping (n=2..7)
            4. S2 pattern simulation (light, pe, area)
            5. CNN position reconstruction
            6. Reconstructed-position S2 pattern
            7. Pattern likelihood + space-time correlation
            """
        ),
        _code(PREAMBLE),
        _code(
            """
            from relics_de_sim import DESimConfig, Pipeline
            cfg = DESimConfig.from_yaml('configs/de_sim.yaml')
            print('dead_time_s:', cfg.simulation.dead_time_s)
            print('pile_up_orders:', cfg.simulation.pile_up_orders)
            """
        ),
        _markdown("## Stage 1 – Load muons"),
        _code(
            """
            from pathlib import Path
            files_per_batch = 5
            muon_files = [Path(cfg.paths.muon_track_dir) / f'muon_track.{i}.npy'
                          for i in range(files_per_batch)]
            pipe = Pipeline(cfg, rng=np.random.default_rng(42))
            pipe.load_muon_tracks(muon_files)
            print(f'simulated time range: {pipe.time_range_s/3600:.2f} h')
            print(f'dead-time ratio (DE survival fraction): we estimate per-config')
            """
        ),
        _markdown("## Stage 2 – Delayed electrons"),
        _code(
            """
            pipe.simulate_delayed_electrons()
            n_de = len(pipe.delayed_electron)
            print(f'{n_de:_} delayed electrons survived the fiducial cut')
            print(f'  rate: {n_de / pipe.time_range_s:.1f} / s')

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(pipe.delayed_electron['e_time'], bins=60)
            axes[0].set_xlabel('e_time [s]'); axes[0].set_ylabel('count')
            axes[1].scatter(pipe.delayed_electron['xd'][:5000],
                            pipe.delayed_electron['yd'][:5000], s=2, alpha=0.3)
            axes[1].set_xlabel('x [mm]'); axes[1].set_ylabel('y [mm]'); axes[1].set_aspect('equal')
            axes[0].set_title('DE arrival times')
            axes[1].set_title('first 5k DE positions')
            plt.tight_layout(); plt.show()
            """
        ),
        _markdown("## Stage 3 – Pile-up grouping (n = 2..7)"),
        _code(
            """
            pipe.simulate_pile_up_patterns()
            for n, res in zip(pipe.pile_up_orders, pipe.pile_up_results):
                print(f'  n={n}: {len(res.pe_by_area)} surviving events')
            """
        ),
        _markdown("## Stage 4-5 – Position reconstruction + scoring"),
        _code(
            """
            try:
                pipe.recon_position_de()
                pipe.score()
                for n, coef, st in zip(pipe.pile_up_orders, pipe.pile_up_pattern_coef, pipe.pile_up_st_cor):
                    if len(coef):
                        print(f'  n={n}: <pattern_coef>={coef.mean():.2f}, <st_cor>={st.mean():.2e}')
            except ImportError as exc:
                print('Skipping CNN inference (torch unavailable):', exc)
            """
        ),
        _markdown(
            """
            ## Stage 6 – Save output

            ``Pipeline.save`` writes the standardised ``de_result.npz`` + ``meta.json``
            (see :mod:`relics_de_sim.io`).
            """
        ),
        _code(
            """
            out = pipe.save('outputs/de_sim/notebook_demo')
            print('wrote', out)
            """
        ),
    ]
    _save("02_de_simulation.ipynb", cells)


# --------------------------------------------------------------------------- #
# 03 - CEvNS simulation
# --------------------------------------------------------------------------- #

def build_03() -> None:
    cells = [
        _markdown(
            """
            # 03 · CEvNS event simulation

            Replaces `sim_script/CEvNS_Sim.ipynb` and the CEvNS-only path of
            `Data_load_reshape.ipynb`.  We

            1. load the CEvNS multiplicity spectrum,
            2. sample events uniformly inside the fiducial volume,
            3. simulate the S2 pattern + pe info,
            4. CNN-reconstruct the position and compute the pattern likelihood + the
               space-time correlation against the muon record.
            """
        ),
        _code(PREAMBLE),
        _code(
            """
            from relics_de_sim import DESimConfig, Pipeline
            cfg = DESimConfig.from_yaml('configs/cevns_sim.yaml')
            print('cevns_e_spectrum:', cfg.paths.cevns_e_spectrum)
            """
        ),
        _markdown("## 1 – Load muons (provides the time range CEvNS samples in)"),
        _code(
            """
            from pathlib import Path
            muon_files = [Path(cfg.paths.muon_track_dir) / f'muon_track.{i}.npy' for i in range(2)]
            pipe = Pipeline(cfg, rng=np.random.default_rng(7))
            pipe.load_muon_tracks(muon_files)
            print(f'time range: {pipe.time_range_s:.1f} s')
            """
        ),
        _markdown("## 2 – Sample CEvNS events"),
        _code(
            """
            try:
                pipe.simulate_cevns()
                for arr in pipe.cevns_points:
                    if len(arr):
                        print(f'n_e={int(arr["num_e"][0])}: {len(arr)} events')
            except ImportError as exc:
                print('CNN inference unavailable:', exc)
            """
        ),
        _markdown(
            """
            ## 3 – Quick look at the CEvNS distribution

            We stack the per-bin arrays and plot total area vs reconstructed radius.
            """
        ),
        _code(
            """
            if pipe.cevns_points:
                arr = np.concatenate(pipe.cevns_points)
                area = arr['pe_by_area'].sum(axis=1)
                rec_x = arr['pos_recon']['xd']
                rec_y = arr['pos_recon']['yd']
                rec_r = np.sqrt(rec_x**2 + rec_y**2)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].hist(area, bins=80)
                axes[0].set_xlabel('Σ pe_by_area'); axes[0].set_ylabel('count')
                axes[1].scatter(rec_r, np.log(np.maximum(arr['st_cor'], 1e-30)), s=4, alpha=0.4)
                axes[1].set_xlabel('reconstructed r [mm]'); axes[1].set_ylabel('log(st_cor)')
                plt.tight_layout(); plt.show()
            """
        ),
        _markdown("## 4 – Save"),
        _code("pipe.save('outputs/cevns_sim/notebook_demo')"),
    ]
    _save("03_cevns_simulation.ipynb", cells)


# --------------------------------------------------------------------------- #
# 04 - Pattern x ST cut fit
# --------------------------------------------------------------------------- #

def build_04() -> None:
    cells = [
        _markdown(
            """
            # 04 · Fitting the pattern × space-time-correlation cut

            Replaces `3D-cut.ipynb` + `3D-cut_multi_process.ipynb`.

            Inputs: a list of ``de_result.npz`` files produced by
            ``scripts/run_de_sim.py``.

            Outputs: ``cut_config/k_st_coefficients.npz`` and
            ``cut_config/b_coefficients.npz`` (legacy schema preserved).
            """
        ),
        _code(PREAMBLE),
        _code(
            """
            from pathlib import Path
            from relics_de_sim.cuts import (
                PatternSTCut,
                fit_pattern_st_cut,
                pattern_likelihood,
            )
            from relics_de_sim.io import load_de_result
            """
        ),
        _markdown("## 1 – Load the DE batch outputs"),
        _code(
            """
            de_files = sorted(Path('outputs/de_sim').glob('batch_*/de_result.npz'))
            assert de_files, 'No DE outputs found - run scripts/run_de_sim.py first.'
            print(f'aggregating {len(de_files)} files')

            pattern, st_cor, area = [], [], []
            for fp in de_files:
                d = load_de_result(fp)
                pattern.append(d['pile_up_pattern_coef'])
                st_cor.append(np.log(np.maximum(d['pile_up_st_cor'], 1e-300)))
                area.append(d['pile_up_area'])
            pattern = np.concatenate(pattern); st_cor = np.concatenate(st_cor); area = np.concatenate(area)
            print(f'{len(area):_} pile-up events total')
            """
        ),
        _markdown("## 2 – 2-D distribution"),
        _code(
            """
            fig, ax = plt.subplots(figsize=(6, 5))
            h = ax.hist2d(st_cor, pattern, bins=80, cmap='viridis')
            ax.set_xlabel('log(st_cor)'); ax.set_ylabel('pattern likelihood')
            plt.colorbar(h[3], ax=ax)
            plt.show()
            """
        ),
        _markdown("## 3 – Fit the area-dependent linear cut"),
        _code(
            """
            cut, diag = fit_pattern_st_cut(area, pattern, st_cor, quantile=0.99)
            print(cut)
            cut.to_npz('cut_config/k_st_coefficients.npz', 'cut_config/b_coefficients.npz')
            """
        ),
        _markdown("## 4 – Visualise the fit"),
        _code(
            """
            bins = diag['area_bins']
            centres = diag['bin_centres']
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(centres, diag['slopes'], 'o', label='per-bin slope')
            axes[0].plot(centres, cut.slope(centres), label='global fit')
            axes[0].set_xlabel('area'); axes[0].set_ylabel('slope k_st(area)'); axes[0].legend()
            axes[1].plot(centres, diag['intercepts'], 'o', label='per-bin intercept')
            axes[1].plot(centres, cut.intercept(centres), label='global fit')
            axes[1].set_xlabel('area'); axes[1].set_ylabel('threshold(area)'); axes[1].legend()
            plt.tight_layout(); plt.show()
            """
        ),
    ]
    _save("04_pattern_st_cor_cut.ipynb", cells)


# --------------------------------------------------------------------------- #
# 05 - Parametric per-N waveform classifier
# --------------------------------------------------------------------------- #

def build_05() -> None:
    cells = [
        _markdown(
            """
            # 05 · Parametric per-N S2 waveform synthesis & classification

            Single notebook that covers what the legacy `2e.ipynb` … `7e.ipynb` series
            +  `*e_cut_wf_gen.py` scripts each duplicated. The electron multiplicity
            is now a parameter (`N_ELECTRONS`).

            Pipeline: CEvNS sim → pattern×ST cut → waveform synthesis →
            (optional) CNN classifier.
            """
        ),
        _code(PREAMBLE),
        _code(
            """
            N_ELECTRONS = 5  # change this and re-run the rest of the notebook
            CONFIG_PATH = 'configs/cevns_sim.yaml'
            """
        ),
        _markdown("## 1 – Run a CEvNS batch"),
        _code(
            """
            from pathlib import Path
            from relics_de_sim import DESimConfig, Pipeline
            cfg = DESimConfig.from_yaml(CONFIG_PATH)
            pipe = Pipeline(cfg, rng=np.random.default_rng(N_ELECTRONS))
            muon_files = [Path(cfg.paths.muon_track_dir) / f'muon_track.{i}.npy' for i in range(2)]
            pipe.load_muon_tracks(muon_files)
            pipe.simulate_cevns()
            arr_idx = next(i for i, a in enumerate(pipe.cevns_points)
                           if len(a) and int(a['num_e'][0]) == N_ELECTRONS)
            arr = pipe.cevns_points[arr_idx]
            pe_info = pipe.cevns_pe_info[arr_idx]
            print(f'{len(arr)} events with n_e={N_ELECTRONS}')
            """
        ),
        _markdown("## 2 – Apply the pattern × ST cut"),
        _code(
            """
            from relics_de_sim.cuts import PatternSTCut, calculate_poisson_log_likelihood
            cut = PatternSTCut.from_npz(cfg.cuts.k_st_coefficients_path,
                                        cfg.cuts.b_coefficients_path)

            st_mask = arr['st_cor'] > 0
            valid = arr[st_mask]
            area = valid['pe_by_area'].sum(axis=1)
            log_st_cor = np.log(valid['st_cor'])
            pattern_coef = np.sum(
                calculate_poisson_log_likelihood(
                    valid['pe_by_area'][:, :64],
                    valid['recons_light_pattern'][:, :64]),
                axis=1)
            passes = cut.passes(pattern_coef, log_st_cor, area)
            print(f'pattern-cut acceptance for n={N_ELECTRONS}: {passes.mean():.3f}')
            """
        ),
        _markdown("## 3 – Synthesise S2 waveforms for the survivors"),
        _code(
            """
            from relics_de_sim.waveform import (
                WaveformParams,
                synthesize_event_waveforms,
            )
            wf_params = WaveformParams.from_configs(cfg.detector, cfg.electronics)
            passing_event_ids = np.where(st_mask)[0][passes]
            z_array = np.linspace(0.0, 24.0, int(passes.sum()))
            waveforms, stds = synthesize_event_waveforms(
                pe_info, passing_event_ids, z_array, wf_params,
                rng=np.random.default_rng(0))
            print('waveforms shape:', waveforms.shape)

            fig, ax = plt.subplots(figsize=(8, 3))
            for w in waveforms[:5]:
                ax.plot(w[1500:2000])
            ax.set_xlabel('sample (centre-trimmed)'); ax.set_ylabel('amplitude (pe/sample)')
            ax.set_title(f'first 5 surviving waveforms, n_e={N_ELECTRONS}')
            plt.show()
            """
        ),
        _markdown(
            """
            ## 4 – (Optional) classifier

            Skip if you don't have ``models/waveform_classifier.pth`` or PyTorch.
            """
        ),
        _code(
            """
            try:
                from relics_de_sim.waveform import WaveformClassifier
                clf = WaveformClassifier(cfg.paths.waveform_classifier_model)
                scores = clf(waveforms)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(scores, bins=40)
                ax.set_xlabel('classifier score'); ax.set_ylabel('count')
                plt.show()
            except (ImportError, FileNotFoundError, Exception) as exc:  # pragma: no cover
                print('Skipping classifier:', exc)
            """
        ),
    ]
    _save("05_waveform_classifier.ipynb", cells)


# --------------------------------------------------------------------------- #
# 06 - CEvNS acceptance study
# --------------------------------------------------------------------------- #

def build_06() -> None:
    cells = [
        _markdown(
            """
            # 06 · CEvNS acceptance study

            Replaces `Acceptance_analyst.ipynb`. Aggregates the per-batch
            ``acceptance_e<N>.npz`` files produced by ``scripts/run_acceptance.py`` and
            plots acceptance vs S2 area for each electron multiplicity.
            """
        ),
        _code(PREAMBLE),
        _code(
            """
            from pathlib import Path
            ACC_DIR = Path('outputs/acceptance')
            files = sorted(ACC_DIR.glob('batch_*/acceptance_e*.npz'))
            assert files, 'Run scripts/run_acceptance.py first to populate outputs/acceptance/.'
            """
        ),
        _markdown("## 1 – Aggregate"),
        _code(
            """
            from collections import defaultdict
            per_n = defaultdict(list)
            for fp in files:
                d = np.load(fp)
                per_n[int(d['e_num'][0])].append(d)

            for n in sorted(per_n):
                area = np.concatenate([d['area'] for d in per_n[n]])
                print(f'n_e={n}: {len(area)} events surviving the pattern cut')
            """
        ),
        _markdown("## 2 – Acceptance vs area"),
        _code(
            """
            bins = np.linspace(0, 300, 31)
            fig, ax = plt.subplots(figsize=(7, 5))
            for n in sorted(per_n):
                area = np.concatenate([d['area'] for d in per_n[n]])
                hist, edges = np.histogram(area, bins=bins)
                ax.step(0.5*(edges[1:]+edges[:-1]), hist, label=f'n_e={n}')
            ax.set_xlabel('S2 area [pe]')
            ax.set_ylabel('events passing cut')
            ax.set_yscale('log')
            ax.legend()
            plt.show()
            """
        ),
        _markdown(
            """
            ## 3 – Combined acceptance figure

            We typically combine the bins into a single acceptance-vs-area curve weighted
            by the assumed CEvNS spectrum.  Adapt the snippet below to your weighting.
            """
        ),
        _code(
            """
            spectrum_path = 'data/e_spectrum_relics.npz'
            spec = np.load(spectrum_path)
            print('spectrum bins:', spec['CEvNS_bins'])
            print('spectrum counts:', spec['CEvNS'])
            """
        ),
    ]
    _save("06_acceptance_study.ipynb", cells)


# --------------------------------------------------------------------------- #
# illustrations notebook
# --------------------------------------------------------------------------- #

def build_illustrations() -> None:
    cells = [
        _markdown(
            """
            # Illustrations · TPC + S2 pattern + waveform GIFs

            Consolidates `Illustration_pattern.ipynb`, `Illustration_waveform.ipynb`,
            `Ilustration.ipynb` and `Untitled1.ipynb`. Each section produces a small,
            self-contained demo figure or animation; they are independent so you can run
            only the parts you care about.
            """
        ),
        _code(PREAMBLE),
        _markdown("## 1 – TPC volume + PMT layout"),
        _code(
            """
            from relics_de_sim import DESimConfig
            from relics_de_sim.geometry import load_pmt_layout
            cfg = DESimConfig.from_yaml('configs/de_sim.yaml')
            pmt_top, pmt_bot = load_pmt_layout(cfg.paths.pmt_top, cfg.paths.pmt_bot)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(pmt_top['x'], pmt_top['y'], s=80, label='top')
            ax.scatter(pmt_bot['x'], pmt_bot['y'], s=80, marker='x', label='bottom')
            for ch, x, y in zip(pmt_top['ChannelID'], pmt_top['x'], pmt_top['y']):
                ax.text(x, y, str(ch), fontsize=6, ha='center', va='center')
            ax.set_aspect('equal'); ax.legend()
            ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
            ax.set_title('Top + bottom PMT layout')
            plt.show()
            """
        ),
        _markdown(
            """
            ## 2 – Animated S1 + S2 toy waveform

            A pure-NumPy illustration replicating the original `Illustration_waveform.ipynb`
            without depending on any detector simulation.
            """
        ),
        _code(
            """
            import matplotlib.animation as animation
            from scipy.stats import norm
            t = np.linspace(-1.5e-6, 1.5e-6, 750)
            s1 = norm.pdf(t, loc=-1e-6, scale=2e-8)
            s2 = norm.pdf(t, loc=5e-7, scale=1.5e-7)
            base = (s1 + s2) / np.max(s1 + s2)

            fig, ax = plt.subplots(figsize=(7, 3))
            line, = ax.plot([], [], lw=2)
            ax.set_xlim(t.min(), t.max())
            ax.set_ylim(-0.1, 1.2)
            ax.set_xlabel('t [s]'); ax.set_ylabel('normalised amplitude')
            ax.set_title('S1 → S2 toy waveform')

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                line.set_data(t[: frame * 10], base[: frame * 10])
                return line,

            anim = animation.FuncAnimation(fig, update, init_func=init, frames=80, blit=True, interval=40)
            plt.close(fig)  # display via Jupyter HTML widget
            from IPython.display import HTML
            HTML(anim.to_jshtml())
            """
        ),
        _markdown(
            """
            ## 3 – Per-electron PMT pattern preview

            Quick demo of `PatternSimulator.per_electron_light` for a fixed event
            position; useful for sanity-checking the LCE map.
            """
        ),
        _code(
            """
            xy = np.array([[10.0, 5.0]])
            light = cfg.electronics  # to access constants
            from relics_de_sim.lce import LCEMap
            lce = LCEMap.from_paths(cfg.paths.lce_xs, cfg.paths.lce_ys, cfg.paths.lce_value)
            pattern = lce.light_pattern(xy)[0]
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(np.arange(64), pattern[:64], label='top')
            ax.bar(np.arange(64, 128), pattern[64:], color='C3', label='bottom')
            ax.set_xlabel('PMT'); ax.set_ylabel('expected pe')
            ax.set_title(f'LCE response at xy={xy[0].tolist()}')
            ax.legend()
            plt.show()
            """
        ),
    ]
    _save("illustrations.ipynb", cells)


def main() -> None:
    build_01()
    build_02()
    build_03()
    build_04()
    build_05()
    build_06()
    build_illustrations()


if __name__ == "__main__":
    main()
