#!/usr/bin/env python
"""在 ``run_20260829_background_cevns.py`` 基础上增加 pattern-likelihood 保存。

与原脚本保持相同的抽样 / DE / CEvNS 模拟流程，仅额外把 DE 与 CEvNS 的
``pattern_likelihood`` 结果写入每次 run 的 ``analysis_result_run_<k>.npz``：

  * ``de_pattern_likelihood``：DE pile-up 事件基于重建光斑计算的
    pattern log-likelihood（等价于原 ``de_pattern``）。
  * ``cevns_pattern_likelihood``：CEvNS 事件基于重建光斑计算的
    pattern log-likelihood。

同时保留原字段 ``de_pattern`` / ``cevns_pattern`` 作为别名，因此原有的
结果读取脚本仍然兼容。

原脚本 ``run_20260829_background_cevns.py`` 不会被修改。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../TPC_DE_SIm-test/scripts
_PROJ_ROOT = _HERE.parent                         # .../TPC_DE_SIm-test

for _path in (_HERE, _PROJ_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

# 复用原脚本中已经实现并验证过的辅助函数；不会执行原脚本的 main()。
import run_20260829_background_cevns as _orig


LOG = logging.getLogger("relics-de-sim.run_20260829_background_cevns_with_pattern")


def parse_args() -> argparse.Namespace:
    """与原脚本参数一致，但默认输出目录单独命名，避免覆盖已有结果。"""
    p = argparse.ArgumentParser(
        description=(
            "Background runner replicating 20260730_sampled.ipynb; "
            "additionally saves DE/CEvNS pattern_likelihood arrays."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ_ROOT / "output" / "20260829_cevns_with_pattern",
        help="Directory to write all results. "
             "[default: <project>/output/20260829_cevns_with_pattern]",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First muon_events/gamma_events file index used as sampling sample.",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of independent sample+simulate runs. [default: 10]",
    )
    p.add_argument(
        "--sim-time",
        type=float,
        default=10000.0,
        help="Simulated wall-clock duration in seconds. [default: 10000]",
    )
    p.add_argument(
        "--muon-rate",
        type=float,
        default=6.0,
        help="Muon event rate in Hz. [default: 6]",
    )
    p.add_argument(
        "--gamma-rate",
        type=float,
        default=1.0,
        help="Gamma event rate in Hz. [default: 1]",
    )
    p.add_argument(
        "--n-muon-files",
        type=int,
        default=1,
        help="Number of muon_events files used as the sampling sample per run.",
    )
    p.add_argument(
        "--n-gamma-files",
        type=int,
        default=6,
        help="Number of gamma_events files used as the sampling sample per run.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device for the position-reconstruction CNN.",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    p.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("Project root: %s", _PROJ_ROOT)
    LOG.info("Output dir  : %s", out)

    # ---------- 加载配置并解析为项目根下的绝对路径 ----------
    print("=" * 60, flush=True)
    print("项目根 _PROJ_ROOT =", _PROJ_ROOT, flush=True)
    cfg_de = _orig.DESimConfig.from_yaml(_PROJ_ROOT / "configs" / "muon_only.yaml")
    _orig.resolve_paths_absolute(cfg_de)
    cfg_cevns = _orig.DESimConfig.from_yaml(_PROJ_ROOT / "configs" / "cevns_sim.yaml")
    _orig.resolve_paths_absolute(cfg_cevns)

    for _key in (
        "pmt_top",
        "pmt_bot",
        "lce_value",
        "lce_value_true",
        "position_reconstruction_model",
    ):
        _p = Path(getattr(cfg_de.paths, _key))
        if not _p.exists():
            print(f"[FATAL] 缺少 {_key} -> {_p}", flush=True)
            raise FileNotFoundError(f"缺少关键文件 {_key}: {_p}")

    # ---------- 定位可用的 muon/gamma 样本文件 ----------
    muon_root = _PROJ_ROOT / "muon_track"
    muon_candidates = _orig.collect_event_files(muon_root, "muon")
    gamma_candidates = _orig.collect_event_files(muon_root, "gamma")
    LOG.info("可用的样本文件: muon=%d, gamma=%d",
             len(muon_candidates), len(gamma_candidates))
    if not muon_candidates or not gamma_candidates:
        raise FileNotFoundError("未找到 muon_events 或 gamma_events 样本文件。")
    n_c_muon = len(muon_candidates)
    n_c_gamma = len(gamma_candidates)

    # ---------- 抽样参数 ----------
    n_muon_target = int(round(args.sim_time * args.muon_rate))
    n_gamma_target = int(round(args.sim_time * args.gamma_rate))
    LOG.info("抽样目标: muon %d 个 eventId, gamma %d 个 eventId; 模拟时长 %.0f s",
             n_muon_target, n_gamma_target, args.sim_time)
    rng = np.random.default_rng(args.seed)

    tmp_dir = out / "_tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    for run in range(args.n_runs):
        LOG.info("\n===== run [%d/%d] =====", run + 1, args.n_runs)

        # 1) 选择抽样样本文件
        muon_start = run + (args.start_index - 1)
        muon_sample = [
            muon_candidates[(muon_start + j) % n_c_muon]
            for j in range(args.n_muon_files)
        ]
        gamma_start = args.start_index - 1
        gamma_sample = [
            gamma_candidates[(gamma_start + j) % n_c_gamma]
            for j in range(args.n_gamma_files)
        ]
        LOG.info("  样本 muon : %s", [str(p.name) for p in muon_sample])
        LOG.info("  样本 gamma: %s", [str(p.name) for p in gamma_sample])

        # 2) 加载、抽样、合并
        muon_loaded = _orig.load_events_from_files(muon_sample)
        gamma_loaded = _orig.load_events_from_files(gamma_sample)
        muon_sampled = _orig.sample_events_by_eventid(
            muon_loaded, n_muon_target, rng, "muon"
        )
        gamma_sampled = _orig.sample_events_by_eventid(
            gamma_loaded, n_gamma_target, rng, "gamma"
        )
        merged_input = _orig.merge_muon_gamma(muon_sampled, gamma_sampled)

        tmp_npy = tmp_dir / f"input_run_{run}.npy"
        np.save(tmp_npy, merged_input)

        # 3) DE 仿真 + 提取 pattern likelihood
        pipe_de = _orig.build_de_pipeline(
            cfg_de,
            muon_root,
            muon_rate_hz=args.muon_rate + args.gamma_rate,
            device=args.device,
            seed=args.seed,
            muon_files=[tmp_npy],
        )
        de = _orig.extract_de_arrays(pipe_de)
        de_pattern_likelihood = np.asarray(de["pattern"])
        de_st_cor = np.asarray(de["st_cor"])
        de_area = np.asarray(de["area"])
        total_time = pipe_de.time_range_s
        LOG.info(
            "run %d: DE st_cor=%d pattern=%d area=%d time=%.3f s",
            run,
            len(de_st_cor),
            len(de_pattern_likelihood),
            len(de_area),
            total_time,
        )
        del pipe_de
        gc.collect()

        # 4) CEvNS 仿真 + 提取 pattern likelihood
        pipe_cevns = _orig.build_cevns_pipeline(
            cfg_cevns,
            muon_root,
            [tmp_npy],
            [],
            device=args.device,
            seed=args.seed,
        )
        cevns = _orig.extract_cevns_arrays(pipe_cevns)
        cevns_pattern_likelihood = np.asarray(cevns["pattern"])
        cevns_st_cor = np.asarray(cevns["st_cor"])
        cevns_area = np.asarray(cevns["area"])
        LOG.info(
            "run %d: CEvNS st_cor=%d pattern=%d area=%d",
            run,
            len(cevns_st_cor),
            len(cevns_pattern_likelihood),
            len(cevns_area),
        )
        del pipe_cevns, cevns
        gc.collect()

        # 5) 保存结果；新增 pattern_likelihood 字段
        tag = f"run_{run:02d}"
        np.savez_compressed(
            out / f"analysis_result_{tag}.npz",
            input_file=str(tmp_npy),
            sim_time=args.sim_time,
            muon_rate=args.muon_rate,
            gamma_rate=args.gamma_rate,
            time_range_s=total_time,
            n_de=len(de_area),
            de_st_cor=de_st_cor,
            de_pattern=de_pattern_likelihood,
            de_pattern_likelihood=de_pattern_likelihood,
            de_area=de_area,
            n_cevns=len(cevns_area),
            cevns_st_cor=cevns_st_cor,
            cevns_pattern=cevns_pattern_likelihood,
            cevns_pattern_likelihood=cevns_pattern_likelihood,
            cevns_area=cevns_area,
        )
        LOG.info("已保存 %s", out / f"analysis_result_{tag}.npz")

        all_summaries.append(
            {
                "run": run,
                "muon_sample_files": [str(p.name) for p in muon_sample],
                "gamma_sample_files": [str(p.name) for p in gamma_sample],
                "time_range_s": float(total_time),
                "n_de": int(len(de_area)),
                "n_de_pattern_likelihood": int(len(de_pattern_likelihood)),
                "n_cevns": int(len(cevns_area)),
                "n_cevns_pattern_likelihood": int(len(cevns_pattern_likelihood)),
            }
        )

        # 释放大对象，控制单进程内存峰值
        del (
            de,
            de_pattern_likelihood,
            de_st_cor,
            de_area,
            cevns_pattern_likelihood,
            cevns_st_cor,
            cevns_area,
            merged_input,
            muon_loaded,
            gamma_loaded,
            muon_sampled,
            gamma_sampled,
        )
        gc.collect()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    LOG.info("全部 %d 次 run 处理完毕。结果目录: %s", args.n_runs, out)
    print("\n=============== %d 次任务全部完成 ===============" % args.n_runs)
    print(json.dumps(all_summaries, indent=2))
    print("结果保存在:", out)


if __name__ == "__main__":
    main()
