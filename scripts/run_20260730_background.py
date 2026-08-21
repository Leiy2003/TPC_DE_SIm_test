#!/usr/bin/env python
"""后台运行脚本 —— 抽样合并 muon/gamma 事件后跑 DE 仿真并分别保存结果（避免 OOM）。

针对大数据量 muon→DE 仿真（20260730.ipynb）：避免直接导入海量 muon/gamma 文件，
而是**从 muon_events 与 gamma_events 文件中按 eventId 随机抽样**，合并后作为
模拟数据输入，跑 DE 仿真后把关键结果单独保存为一个 numpy 文件。

抽样逻辑：
  1. 分别导入指定数量的 muon_events / gamma_events 文件作为抽样样本
     （例如每次 1 个 muon_events 文件 + 6 个 gamma_events 文件）。
  2. 指定模拟时长与事件率（例如 10000 s、muon 6 Hz、gamma 1 Hz）。
  3. 按 eventId 从对应文件中**随机抽取事件**：每次随机抽某个 eventId 的数据加入
     样本，直至 muon 数（按 eventId 计）达到 10000*6、gamma 数（按 eventId 计）
     达到 1*10000。
  4. 将抽样出的 muon 与 gamma 合并，合并后 eventId 从 0 重新排序。
  5. 合并数组替代原导入文件，作为 Pipeline 的模拟输入。

使用方法（后台运行）：
    Linux / macOS：
        nohup python scripts/run_20260730_background.py \
            --out-dir output/20260730 --start-index 1 --n-runs 10 \
            --sim-time 10000 --muon-rate 6 --gamma-rate 1 \
            --n-muon-files 1 --n-gamma-files 6 \
            --verbose > run.log 2>&1 &

    Windows PowerShell（隐藏窗口后台运行）：
        Start-Process python -ArgumentList 'scripts/run_20260730_background.py','--out-dir','output/20260730','--start-index','1','--n-runs','10','--sim-time','10000','--muon-rate','6','--gamma-rate','1','--n-muon-files','1','--n-gamma-files','6','--verbose' `
            -RedirectStandardOutput 'run.log' -RedirectStandardError 'run.err' -WindowStyle Hidden

关键点：
  * 每次运行做 n-runs 次独立抽样+模拟，每次关键结果存为 analysis_result_run_<k>.npz。
  * 每完成一次即 del + gc.collect() 释放内存，控制单进程峰值。
  * 所有结果写入 --out-dir；summary.json 汇总全部任务信息。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---- 自动把项目根目录（TPC_DE_SIm-test）加入 sys.path，保证两头都能 import ----
_HERE = Path(__file__).resolve().parent          # .../TPC_DE_SIm-test/scripts
_PROJ_ROOT = _HERE.parent                       # .../TPC_DE_SIm-test
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np

from relics_de_sim import DESimConfig, Pipeline
from relics_de_sim.cuts import pattern_likelihood


LOG = logging.getLogger("relics-de-sim.run_20260730_background")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Background runner replicating 20260730.ipynb.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ_ROOT / "output" / "20260730",
        help="Directory to write all results. [default: <project>/output/20260730]",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First muon_events/gamma_events file index used as sampling sample "
             "(e.g. 1 -> muon_events.1.npy, gamma_events.1.npy).",
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
        help="Muon event rate in Hz. Total muon targets = sim_time * muon_rate. [default: 6]",
    )
    p.add_argument(
        "--gamma-rate",
        type=float,
        default=1.0,
        help="Gamma event rate in Hz. Total gamma targets = sim_time * gamma_rate. [default: 1]",
    )
    p.add_argument(
        "--n-muon-files",
        type=int,
        default=1,
        help="Number of muon_events files used as the sampling sample per run. [default: 1]",
    )
    p.add_argument(
        "--n-gamma-files",
        type=int,
        default=6,
        help="Number of gamma_events files used as the sampling sample per run. [default: 6]",
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


def collect_event_files(base_dir: Path, event_type: str) -> list[Path]:
    """与 notebook 一致的查找逻辑：优先 *_events.*.npy，回退 *.npy。"""
    base = Path(base_dir)
    event_dir = base / f"{event_type}_events"
    matches = sorted(event_dir.glob(f"{event_type}_events.*.npy")) if event_dir.exists() else []
    if not matches:
        matches = sorted(base.rglob(f"{event_type}_events.*.npy"))
    if not matches:
        matches = sorted(base.rglob("*.npy"))
    return matches


import os

# from_yaml 可能把这些字段解析成 YAML 相对路径（如 ../LCE_info/topPMTs.txt），
# 这里显式列出需要基于项目根重解析为绝对路径的所有字段。
_PATH_FIELDS = (
    "pmt_top",
    "pmt_bot",
    "lce_value",
    "lce_value_true",
    "lce_xs",
    "lce_ys",
    "muon_track_dir",
    "cevns_e_spectrum",
    "position_reconstruction_model",
    "waveform_classifier_model",
    "output_dir",
)


def _join_under_root(rel: str | Path) -> Path:
    """把相对路径规范化为“项目根下”的绝对路径，且不依赖当前工作目录(cwd)。

    关键点：
      - 对相对路径绝不能直接 ``.resolve()``，否则会基于 **cwd** 展开导致错位
        （例如 cwd=TPC_DE_SIm-test 时，``../LCE_info/....`` 会跳到 /.../LCE_info）。
      - YAML 里的 ``../xxx`` 在 from_yaml 中是以 configs/ 为基准，
        configs/../xxx = 项目根/xxx。这里直接用 os.path.normpath 基于项目根处理
        ``..``，得到项目根下的确定路径。
    """
    p = Path(rel)
    if p.is_absolute():
        return Path(os.path.normpath(p))
    # 基于项目根拼接并做语法层归一化（不 resolve，避免 cwd 干扰）
    return Path(os.path.normpath(_PROJ_ROOT / p))


def resolve_paths_absolute(cfg: DESimConfig,
                          extra: dict[str, str] | None = None) -> None:
    """把 cfg 的关键 paths 字段重写为 **_PROJ_ROOT 下**的绝对路径。

    - 用 os.path.normpath 做语法层 . / .. 清理，并始终以项目根为基准。
    - extra 可按需追加覆盖某字段（例如 CEvNS 的 cevns_e_spectrum）。
    - 用 print + flush 直接输出，确保即使 logging 未 flush 也能在 run.log 看到。
    """
    overrides = dict(extra or {})
    for fld_name in _PATH_FIELDS:
        if not hasattr(cfg.paths, fld_name):
            continue
        val = overrides.get(fld_name, getattr(cfg.paths, fld_name))
        if isinstance(val, str) and val:
            abs_p = _join_under_root(val)
            setattr(cfg.paths, fld_name, str(abs_p))
            print(f"[path] {fld_name:28s} -> {abs_p}", flush=True)


def build_de_pipeline(cfg: DESimConfig, muon_root: Path, muon_rate_hz: float,
                      device: str, seed: int | None,
                      muon_files: list[Path] | None = None) -> Pipeline:
    """构造并运行 DE pipeline（对应 notebook cell 1~5 的 pipe）。

    只处理 **muon** 文件（为控制单次运行内存峰值，默认仅加载这一批 muon 文件，
    不加载 gamma）。muon_files 可显式传入；为 None 时用 collect_event_files 查找。
    """
    cfg.paths.muon_track_dir = str(muon_root)
    cfg.simulation.muon_rate_hz = muon_rate_hz

    if muon_files is None:
        muon_files = collect_event_files(muon_root, "muon")
    all_input_files = [str(p) for p in muon_files]
    LOG.info("DE: loading %d muon file(s):", len(all_input_files))
    for fl in all_input_files:
        LOG.info("  file: %s", fl)

    rng = np.random.default_rng(seed)
    pipe = Pipeline(cfg, rng=rng, device=device)
    pipe.load_muon_tracks(all_input_files)
    LOG.info("DE: loaded. time_range_s=%.3f, dense_muon.shape=%s",
             pipe.time_range_s, None if pipe.dense_muon is None else pipe.dense_muon.shape)

    LOG.info("DE: simulate_delayed_electrons() ...")
    pipe.simulate_delayed_electrons()
    LOG.info("DE: simulate_pile_up_patterns() ...")
    pipe.simulate_pile_up_patterns()
    LOG.info("DE: recon_position_de() ...")
    pipe.recon_position_de()
    LOG.info("DE: score() ...")
    pipe.score()
    for order, res in zip(pipe.pile_up_orders, pipe.pile_up_results):
        LOG.info("DE: order=%s kept=%d", order, len(res.pe_by_area))
    return pipe


def build_cevns_pipeline(cfg_cevns: DESimConfig, muon_root: Path,
                         muon_files: list[Path], gamma_files: list[Path],
                         device: str, seed: int | None) -> Pipeline:
    """构造并运行 CEvNS pipeline（对应 notebook cell 中 pipe_cevns 部分）。"""
    cfg_cevns.paths.muon_track_dir = str(muon_root)
    load_files = [str(p) for p in (muon_files + gamma_files)]

    pipe = Pipeline(cfg_cevns, rng=np.random.default_rng(seed), device=device)
    pipe.load_muon_tracks(load_files)
    LOG.info("CEvNS: loaded %d files, time_range_s=%.3f", len(load_files), pipe.time_range_s)

    LOG.info("CEvNS: simulate_cevns() ...")
    pipe.simulate_cevns()
    for arr in pipe.cevns_points:
        if len(arr):
            LOG.info("CEvNS: n_e=%d -> %d events", int(arr["num_e"][0]), len(arr))
    return pipe


def extract_de_arrays(pipe: Pipeline):
    """提取 DE 侧 key 数组：st_cor, pattern, area（对应 notebook 后半部分）。"""
    st_cor = np.concatenate(pipe.pile_up_st_cor)
    pattern_coef = np.concatenate(pipe.pile_up_pattern_coef)

    pe_by_area_list = [r.pe_by_area for r in pipe.pile_up_results]
    pe_by_area = np.concatenate(pe_by_area_list) if pe_by_area_list else np.zeros((0, 128))
    area = np.sum(pe_by_area, axis=1)
    return dict(st_cor=st_cor, pattern=pattern_coef, pe_by_area=pe_by_area, area=area)


def extract_cevns_arrays(pipe: Pipeline):
    """提取 CEvNS 侧 key 数组（对应 arr['st_cor'] / pattern_likelihood / area）。"""
    arr = np.concatenate(pipe.cevns_points)
    st_cor = arr["st_cor"]
    pattern = pattern_likelihood(arr["pe_by_area"], arr["recons_light_pattern"], n_top=28)
    area = np.sum(arr["pe_by_area"], axis=1)
    return dict(st_cor=st_cor, pattern=pattern, area=area, raw=arr)


def load_events_from_files(files: list[Path]) -> np.ndarray:
    """加载多个 *_events 文件并合并，同时把 eventId 偏移成全局唯一。

    每个 *_events 文件（如 h5reader 生成的 muon_events.<i>.npy）里的 eventId
    都是从 0 独立编号的，合并多个文件时需做偏移，否则不同文件的 eventId 会冲突。
    """
    arrays: list[np.ndarray] = []
    pre_event_num = 0
    for path in files:
        arr = np.load(path)
        file_event_num = int(arr["eventId"][-1]) + 1
        arr = arr.copy()
        arr["eventId"] = arr["eventId"] + pre_event_num
        pre_event_num += file_event_num
        arrays.append(arr)
    merged = np.concatenate(arrays, axis=0)
    return merged


def sample_events_by_eventid(merged: np.ndarray, n_target: int,
                             rng: np.random.Generator, event_type: str) -> np.ndarray:
    """按 eventId 从合并数组中抽取事件，直到抽满 n_target 个事件（eventId）。

    抽样策略：
      1) 优先**随机无放回**抽样：从候选 eventId 中逐个随机抽，每个只抽一次。
      2) 当候选事件数 **不足** n_target 时，先无放回抽完全部候选，再**有放回**
         补足剩余数量（每次从原有候选 eventId 中随机可重复抽取）。
      3) 被重复抽到的事件：由于它已在无放回阶段出现过，第二次及以上被抽到时，
         会为它复制一份数据并分配一个**全新的、不与任何已有 eventId 重复**的
         eventId，作为一个独立事件记入候选，最终保证不同 eventId 总数= n_target。
    """
    unique_ids = np.unique(merged["eventId"])
    n_avail = len(unique_ids)

    if n_avail >= n_target:
        # ---- 情况 1：候选充足，纯随机无放回 ----
        chosen = rng.choice(unique_ids, size=n_target, replace=False)
        mask = np.isin(merged["eventId"], chosen)
        picked = merged[mask]
        LOG.info("%s: 无放回抽样 %d 个 eventId，共 %d 行（候选 %d）",
                 event_type, n_target, len(picked), n_avail)
        return picked

    # ---- 情况 2：候选不足，先无放回抽全部，再有放回补足 ----
    LOG.warning("%s: 候选 eventId 数 %d < 目标 %d，先无放回抽全部，再有放回补足",
                event_type, n_avail, n_target)

    # 2.1 无放回部分：全部候选行，保持原 eventId
    base_mask = np.isin(merged["eventId"], unique_ids)
    base_part = merged[base_mask]

    # 2.2 有放回部分：再随机抽 need 次（可重复），每次复制一份并分配全新 eventId
    need = n_target - n_avail
    repeat_ids = rng.choice(unique_ids, size=need, replace=True)
    next_id = int(merged["eventId"].max()) + 1   # 从现有最大编号之后开始分配
    extra_parts = []
    for eid in repeat_ids:
        sub = merged[merged["eventId"] == eid].copy()
        sub["eventId"] = next_id                 # 给这次重复抽样一个全新独立 eventId
        next_id += 1
        extra_parts.append(sub)

    picked = np.concatenate([base_part] + extra_parts, axis=0) if extra_parts else base_part
    LOG.info("%s: 抽样 %d 个 eventId（无放回 %d + 有放回补足 %d），共 %d 行",
             event_type, n_target, n_avail, need, len(picked))
    return picked


def merge_muon_gamma(muon_arr: np.ndarray, gamma_arr: np.ndarray) -> np.ndarray:
    """把抽样得到的 muon 与 gamma 合并，并将 eventId 从 0 重新排序。

    关键：muon 与 gamma 是两个独立事件类型，它们的 eventId 编号都从 0 独立偏移，
    区间可能重叠。若直接合并再用 np.unique 重排，编号相同的 muon/gamma 事件会被
    误当成同一个事件归并。因此先把 gamma 的 eventId 整体偏移到 muon 的最大值之上，
    使两类事件的 eventId 区间互不重叠，再合并、排序、从 0 连续重排。
    """
    # 把 gamma 偏移到 muon 之上，保证两类事件的 eventId 区间不相交
    offset = int(muon_arr["eventId"].max()) + 1
    gamma_arr2 = gamma_arr.copy()
    gamma_arr2["eventId"] = gamma_arr2["eventId"] + offset

    merged = np.concatenate([muon_arr, gamma_arr2], axis=0)
    order = np.argsort(merged["eventId"], kind="stable")
    merged = merged[order]
    merged["eventId"] = np.unique(merged["eventId"], return_inverse=True)[1]
    LOG.info("合并后共 %d 行，%d 个不同 eventId",
             len(merged), int(merged["eventId"][-1]) + 1)
    return merged


def calculate_score(st_cor, area, k_area):
    return k_area * area - np.log(st_cor)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    import gc
    import json

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("Project root: %s", _PROJ_ROOT)
    LOG.info("Output dir  : %s", out)

    # ---------- 加载配置文件并解析绝对路径（只做一次）----------
    print("=" * 60, flush=True)
    print("项目根 _PROJ_ROOT =", _PROJ_ROOT, flush=True)
    cfg_de = DESimConfig.from_yaml(_PROJ_ROOT / "configs" / "muon_only.yaml")
    resolve_paths_absolute(cfg_de)  # 显式重写为项目根下的绝对路径
    # 预检关键文件；若缺文件直接给出绝对路径与目录清单，避免后半程再报错
    for _key in ("pmt_top", "pmt_bot", "lce_value", "lce_value_true",
                 "position_reconstruction_model"):
        _p = Path(getattr(cfg_de.paths, _key))
        if not _p.exists():
            print(f"[FATAL] 缺少 {_key} -> {_p}", flush=True)
            print(f"[INFO]  项目根={_PROJ_ROOT} 下 LCE_info/ 内容: "
                  f"{[str(x) for x in (_PROJ_ROOT / 'LCE_info').iterdir()]}", flush=True)
            raise FileNotFoundError(f"缺少关键文件 {_key}: {_p}")
    print("DE config pmt_top =", cfg_de.paths.pmt_top, flush=True)

    # ---------- 定位可用的 muon_events / gamma_events 样本文件 ----------
    muon_root = _PROJ_ROOT / "muon_track"
    muon_candidates = collect_event_files(muon_root, "muon")
    gamma_candidates = collect_event_files(muon_root, "gamma")
    LOG.info("可用的样本文件: muon=%d, gamma=%d", len(muon_candidates), len(gamma_candidates))
    if not muon_candidates or not gamma_candidates:
        raise FileNotFoundError("未找到 muon_events 或 gamma_events 样本文件。")
    n_c_muon = len(muon_candidates)
    n_c_gamma = len(gamma_candidates)

    # ---------- 抽样参数 ----------
    n_muon_target = int(round(args.sim_time * args.muon_rate))    # 如 10000*6 = 60000
    n_gamma_target = int(round(args.sim_time * args.gamma_rate))  # 如 10000*1 = 10000
    LOG.info("抽样目标: muon %d 个 eventId, gamma %d 个 eventId; 模拟时长 %.0f s",
             n_muon_target, n_gamma_target, args.sim_time)
    rng = np.random.default_rng(args.seed)

    # 临时目录：存放每次抽样合并后的输入数组
    tmp_dir = out / "_tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 循环 n-runs 次：抽样合并 → 模拟 → 分别保存结果 ----------
    all_summaries: list[dict] = []
    for run in range(args.n_runs):
        LOG.info("\n===== run [%d/%d] =====", run + 1, args.n_runs)

        # 1) 取抽样样本文件：
        #    muon  : 每次都取不同编号的 muon_events 文件（按 run 递增，越界回绕）
        #    gamma : 固定用同一批 n_gamma_files 个文件（编号不变）
        muon_start = run + (args.start_index - 1)
        muon_sample = [muon_candidates[(muon_start + j) % n_c_muon]
                       for j in range(args.n_muon_files)]
        gamma_start = args.start_index - 1
        gamma_sample = [gamma_candidates[(gamma_start + j) % n_c_gamma]
                        for j in range(args.n_gamma_files)]
        LOG.info("  样本 muon : %s", [str(p.name) for p in muon_sample])
        LOG.info("  样本 gamma: %s (固定)", [str(p.name) for p in gamma_sample])

        # 2) 加载样本文件，并按 eventId 随机抽样到目标数量
        muon_loaded = load_events_from_files(muon_sample)
        gamma_loaded = load_events_from_files(gamma_sample)
        muon_sampled = sample_events_by_eventid(muon_loaded, n_muon_target, rng, "muon")
        gamma_sampled = sample_events_by_eventid(gamma_loaded, n_gamma_target, rng, "gamma")

        # 3) 合并 muon+gamma，eventId 从 0 重新排序
        merged_input = merge_muon_gamma(muon_sampled, gamma_sampled)

        # 4) 把合并数组保存为临时文件，作为 Pipeline 的输入（替代原直接导入文件）
        tmp_npy = tmp_dir / f"input_run_{run}.npy"
        np.save(tmp_npy, merged_input)

        # 5) 跑 DE 仿真
        pipe_de = build_de_pipeline(
            cfg_de, muon_root, muon_rate_hz=args.muon_rate+args.gamma_rate,
            device=args.device, seed=args.seed,
            muon_files=[tmp_npy],   # 传入合并后的输入数组
        )
        # 6) 提取关键结果
        de = extract_de_arrays(pipe_de)
        total_time = pipe_de.time_range_s
        LOG.info("run %d: st_cor=%d pattern=%d area=%d time=%.3f s",
                 run, len(de["st_cor"]), len(de["pattern"]), len(de["area"]), total_time)

        # 7) 把本次关键结果单独保存为一个 numpy 文件
        #    （为减小空间占用，不保存体积较大的 de_pe_by_area）
        tag = f"run_{run:02d}"
        np.savez_compressed(
            out / f"analysis_result_{tag}.npz",
            input_file=str(tmp_npy),
            sim_time=args.sim_time,
            muon_rate=args.muon_rate,
            gamma_rate=args.gamma_rate,
            time_range_s=total_time,
            n_de=len(de["area"]),
            de_st_cor=de["st_cor"],
            de_pattern=de["pattern"],
            de_area=de["area"],
        )
        LOG.info("已保存 %s", out / f"analysis_result_{tag}.npz")

        # 8) 记录汇总信息
        all_summaries.append({
            "run": run,
            "muon_sample_files": [str(p.name) for p in muon_sample],
            "gamma_sample_files": [str(p.name) for p in gamma_sample],
            "time_range_s": float(total_time),
            "n_de": int(len(de["area"])),
        })

        # 9) 释放大对象，为下一次 run 腾出内存
        del de, pipe_de, merged_input, muon_loaded, gamma_loaded
        del muon_sampled, gamma_sampled
        gc.collect()

    # ---------- 清理临时输入文件，并汇总 ----------
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    LOG.info("全部 %d 次 run 处理完毕。结果目录: %s", args.n_runs, out)
    print("\n=============== %d 次任务全部完成 ===============" % args.n_runs)
    print(json.dumps(all_summaries, indent=2))
    print("结果保存在:", out)


if __name__ == "__main__":
    main()
