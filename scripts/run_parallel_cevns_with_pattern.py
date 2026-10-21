#!/usr/bin/env python
"""并行启动 ``run_20260829_background_cevns_with_pattern.py`` 的启动器。

为什么不能简单地把同一个命令起 N 遍：

  * 原脚本每个进程都把结果写成 ``analysis_result_run_<k>.npz``（k 是进程内
    的局部编号），多个进程写同一个 ``--out-dir`` 会互相覆盖同名文件；
  * 原脚本共用 ``out_dir/_tmp_inputs`` 和 ``out_dir/summary.json``，也会冲突。

因此本脚本给每个 worker 分配一个**独立的 out-dir** 和**独立的日志文件**，
并用不同的 seed（以及可选的 start-index）保证每次抽样不同。worker 结束后，
本脚本可以把每个 worker 的结果文件按全局唯一编号收集到 ``--collect-dir``。

本脚本不会修改原脚本 ``run_20260829_background_cevns_with_pattern.py``。

用法示例（与当前正在跑的作业参数一致，新增 4 个并行 worker）：

    nohup python scripts/run_parallel_cevns_with_pattern.py \
        --out-dir output/run_muon8_gamma2_deadtime_20ms_par \
        --n-workers 4 --n-runs 1000 \
        --sim-time 10000 --muon-rate 8 --gamma-rate 2 \
        --n-muon-files 2 --n-gamma-files 6 --device cuda \
        > parallel_launcher.log 2>&1 &

说明：

  * 每个 worker 的完整输出写到 ``<out-dir>/logs/<run_tag>/worker_XX.log``；
  * 每次 run 保存成功时，本启动器会把一条完成记录追加到
    ``<out-dir>/logs/<run_tag>/runs.log``；
  * 全部 worker 结束后，结果 npz 会被收集到 ``--collect-dir``（默认等于
    ``--out-dir``），文件名形如
    ``analysis_result_<run_tag>_run_<全局编号>.npz``，不会覆盖已有文件。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
DEFAULT_SCRIPT = _HERE / "run_20260829_background_cevns_with_pattern.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="并行启动多个 CEνNS+DE background worker。",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ_ROOT / "output" / "run_muon8_gamma2_deadtime_20ms_par",
        help="并行任务的总输出目录；worker 会写到其下的 workers/ 子目录。",
    )
    p.add_argument("--n-workers", type=int, default=4, help="并行 worker 数量。")
    p.add_argument("--n-runs", type=int, default=1000, help="每个 worker 模拟的 run 数。")
    p.add_argument(
        "--start-index-base",
        type=int,
        default=1,
        help="第 0 个 worker 的 --start-index；后续 worker 按 step 递增。",
    )
    p.add_argument(
        "--start-index-step",
        type=int,
        default=1,
        help="相邻 worker 的 start-index 间隔。",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="第 0 个 worker 的随机种子；请勿与正在运行的作业 seed 相同。",
    )
    p.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="相邻 worker 的 seed 间隔。",
    )
    p.add_argument("--sim-time", type=float, default=10000.0)
    p.add_argument("--muon-rate", type=float, default=8.0)
    p.add_argument("--gamma-rate", type=float, default=2.0)
    p.add_argument("--n-muon-files", type=int, default=2)
    p.add_argument("--n-gamma-files", type=int, default=6)
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="运行原脚本的 python 解释器；默认与当前解释器相同。",
    )
    p.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="原脚本路径。")
    p.add_argument(
        "--stagger-seconds",
        type=float,
        default=10.0,
        help="每个 worker 启动之间的间隔（秒），避免同时加载模型造成瞬时峰值。",
    )
    p.add_argument(
        "--cuda-devices",
        default=None,
        help="可选：按 worker 轮询分配 GPU，例如 '0,1,2,3'（通过 CUDA_VISIBLE_DEVICES）。",
    )
    p.add_argument(
        "--collect-dir",
        type=Path,
        default=None,
        help="全部结束后把结果收集到该目录；默认等于 --out-dir。",
    )
    p.add_argument(
        "--no-collect",
        action="store_true",
        help="不把结果归并到 --collect-dir，保留在每个 worker 子目录中。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不实际启动。",
    )
    return p.parse_args()


class LogTail:
    """增量读取一个正在被写入的日志文件。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh = path.open("r", encoding="utf-8", errors="replace")
        self.buf = ""

    def iter_lines(self):
        self.buf += self.fh.read()
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            yield line

    def flush_lines(self):
        if self.buf.strip():
            yield self.buf
            self.buf = ""

    def close(self) -> None:
        self.fh.close()


def build_worker_cmd(args: argparse.Namespace, start_index: int, seed: int, out_dir: Path) -> list[str]:
    return [
        args.python,
        str(args.script),
        "--out-dir",
        str(out_dir),
        "--start-index",
        str(start_index),
        "--n-runs",
        str(args.n_runs),
        "--sim-time",
        str(args.sim_time),
        "--muon-rate",
        str(args.muon_rate),
        "--gamma-rate",
        str(args.gamma_rate),
        "--n-muon-files",
        str(args.n_muon_files),
        "--n-gamma-files",
        str(args.n_gamma_files),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--verbose",
    ]


def launch_workers(args: argparse.Namespace, run_tag: str, out_dir: Path) -> list[dict]:
    workers_root = out_dir / "workers"
    logs_dir = out_dir / "logs" / run_tag
    workers_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cuda_devices = None
    if args.cuda_devices:
        cuda_devices = [x.strip() for x in args.cuda_devices.split(",") if x.strip()]

    workers: list[dict] = []
    for i in range(args.n_workers):
        start_index = args.start_index_base + i * args.start_index_step
        seed = args.seed_base + i * args.seed_step
        worker_dir = workers_root / (
            f"worker_{i:02d}_idx{start_index}_seed{seed}"
        )
        log_path = logs_dir / f"worker_{i:02d}.log"
        worker_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_worker_cmd(args, start_index, seed, worker_dir)
        info = {
            "id": i,
            "start_index": start_index,
            "seed": seed,
            "out_dir": str(worker_dir),
            "log_path": str(log_path),
            "cmd": cmd,
            "proc": None,
            "returncode": None,
        }
        workers.append(info)

        if args.dry_run:
            env_prefix = ""
            if cuda_devices:
                env_prefix = f"CUDA_VISIBLE_DEVICES={cuda_devices[i % len(cuda_devices)]} "
            print(f"[dry-run] worker {i:02d}: {env_prefix}" + " ".join(cmd))
            continue

        env = os.environ.copy()
        if cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices[i % len(cuda_devices)]

        with log_path.open("wb") as log_fh:
            log_fh.write(
                f"[launcher] worker {i:02d} start_index={start_index} seed={seed}\n".encode()
            )
            log_fh.write(f"[launcher] cmd: {' '.join(cmd)}\n\n".encode())
            log_fh.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(args.script.parent),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        info["proc"] = proc
        print(
            f"[launcher] worker {i:02d} started, pid={proc.pid}, "
            f"seed={seed}, start_index={start_index}, log={log_path}",
            flush=True,
        )

        if i < args.n_workers - 1 and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

    return workers


def monitor_workers(args: argparse.Namespace, workers: list[dict], runs_log: Path) -> None:
    tails = {
        i: LogTail(Path(info["log_path"]))
        for i, info in enumerate(workers)
    }

    try:
        while True:
            running = [w for w in workers if w["proc"] is not None and w["proc"].poll() is None]
            for i, info in enumerate(workers):
                for line in tails[i].iter_lines():
                    if "已保存" in line:
                        with runs_log.open("a", encoding="utf-8") as fh:
                            fh.write(
                                f"[worker_{i:02d}] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"{line.strip()}\n"
                            )
            if not running:
                break
            time.sleep(5)
    finally:
        for i, info in enumerate(workers):
            for line in tails[i].flush_lines():
                if "已保存" in line:
                    with runs_log.open("a", encoding="utf-8") as fh:
                        fh.write(
                            f"[worker_{i:02d}] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"{line.strip()}\n"
                        )
            tails[i].close()

    for info in workers:
        if info["proc"] is not None:
            info["returncode"] = info["proc"].poll()


def collect_results(
    args: argparse.Namespace,
    workers: list[dict],
    run_tag: str,
    out_dir: Path,
) -> tuple[int, Path]:
    collect_dir = args.collect_dir or out_dir
    collect_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    for i, info in enumerate(workers):
        worker_dir = Path(info["out_dir"])
        for r in range(args.n_runs):
            src = worker_dir / f"analysis_result_run_{r:02d}.npz"
            if not src.exists():
                continue
            global_id = i * args.n_runs + r
            dst = collect_dir / f"analysis_result_{run_tag}_run_{global_id:05d}.npz"
            if dst.exists():
                # run_tag 每次启动都不同，正常情况下不应发生；保留已有文件，不覆盖。
                dst = collect_dir / (
                    f"analysis_result_{run_tag}_run_{global_id:05d}_dup{i:02d}.npz"
                )
            import shutil

            shutil.move(str(src), str(dst))
            moved += 1

        summary_src = worker_dir / "summary.json"
        if summary_src.exists():
            summary_dst = collect_dir / f"summary_{run_tag}_worker_{i:02d}.json"
            if summary_dst.exists():
                summary_dst = collect_dir / f"summary_{run_tag}_worker_{i:02d}_dup.json"
            import shutil

            shutil.move(str(summary_src), str(summary_dst))

    return moved, collect_dir


def main() -> None:
    args = parse_args()
    if not args.script.exists():
        raise FileNotFoundError(f"原脚本不存在: {args.script}")
    if args.n_workers < 1:
        raise ValueError("--n-workers 必须 >= 1")
    if args.n_runs < 1:
        raise ValueError("--n-runs 必须 >= 1")

    out_dir = args.out_dir.resolve()
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    print(f"[launcher] project root: {_PROJ_ROOT}", flush=True)
    print(f"[launcher] out-dir      : {out_dir}", flush=True)
    print(f"[launcher] n-workers    : {args.n_workers}", flush=True)
    print(f"[launcher] n-runs/worker: {args.n_runs}", flush=True)
    print(
        f"[launcher] seeds        : {args.seed_base} .. "
        f"{args.seed_base + (args.n_workers - 1) * args.seed_step}",
        flush=True,
    )

    workers = launch_workers(args, run_tag, out_dir)
    if args.dry_run:
        print("[dry-run] 仅打印命令，未启动任何进程。", flush=True)
        return

    logs_dir = out_dir / "logs" / run_tag
    runs_log = logs_dir / "runs.log"
    runs_log.touch(exist_ok=True)

    manifest_path = out_dir / "manifests"
    manifest_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_tag": run_tag,
        "out_dir": str(out_dir),
        "n_workers": args.n_workers,
        "n_runs_per_worker": args.n_runs,
        "workers": [
            {
                "id": w["id"],
                "pid": w["proc"].pid if w["proc"] is not None else None,
                "start_index": w["start_index"],
                "seed": w["seed"],
                "out_dir": w["out_dir"],
                "log_path": w["log_path"],
            }
            for w in workers
        ],
    }
    (manifest_path / f"manifest_{run_tag}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    print(f"[launcher] 正在监控 worker，完成记录写入: {runs_log}", flush=True)
    monitor_workers(args, workers, runs_log)

    failed = [w for w in workers if w["returncode"] not in (0, None)]
    if failed:
        for w in failed:
            print(
                f"[launcher] WARNING worker {w['id']:02d} 退出码={w['returncode']}, "
                f"日志: {w['log_path']}",
                flush=True,
            )

    if not args.no_collect:
        moved, collect_dir = collect_results(args, workers, run_tag, out_dir)
        print(
            f"[launcher] 全部 worker 结束；已收集 {moved} 个结果文件到 {collect_dir}",
            flush=True,
        )
    else:
        print("[launcher] 全部 worker 结束；结果保留在各 worker 子目录。", flush=True)

    print(
        f"[launcher] 每个 run 的完成记录: {runs_log}\n"
        f"[launcher] manifest: {manifest_path / f'manifest_{run_tag}.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
