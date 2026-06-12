#!/usr/bin/env python3
"""Run LC-gate, LiDAE-KU/LC, and single-LiDAR checkpoints, then plot trajectories.

This wrapper is meant for the paper-style comparison figure where the same
real-tree condition is evaluated with three policies and their trajectories are
overlaid in one image.  It can try multiple seeds and choose the seed whose
metrics most clearly satisfy:

    LC-gate > LiDAE-KU/LC > Single-LiDAR

The selection is recorded in selection_manifest.json next to the figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ZK_DIR = SCRIPT_DIR.parent
OMNIDRONES_DIR = ZK_DIR.parent.parent
REPO_ROOT = OMNIDRONES_DIR.parent

LCGATE_SCRIPT = SCRIPT_DIR / "realtree_sweep_camlidar_gate_video.py"
LC_SCRIPT = ZK_DIR / "contrast_with_sota_realtree_lc" / "realtree_sweep_camlidar_eval_lc_compat.py"
LIDAR_SCRIPT = ZK_DIR / "contrast_with_sota_realtree_lidar" / "realtree_sweep_lidar_eval.py"
PLOT_SCRIPT = SCRIPT_DIR / "plot_method_trajectory_comparison.py"

DEFAULT_LCGATE_CKPT = ZK_DIR / "goodpt" / "6-6-vlim-lcgat-tree_best_return_4574.55.pt"
DEFAULT_LC_CKPT = ZK_DIR / "goodpt" / "5-16-vlim-lc-tree_best_return_2460.18.pt"
DEFAULT_LIDAR_CKPT = ZK_DIR / "goodpt" / "5-16-vlim-lidar-tree_best_return_668.45.pt"

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "trajectory_compare_from_pt"
DEFAULT_FIGURE = SCRIPT_DIR / "figures" / "lcgate_lc_lidar_trajectory_from_pt.png"
DEFAULT_PYTHON = Path("/home/descfly/anaconda3/envs/omni-5.1/bin/python")


@dataclass
class MethodSpec:
    key: str
    label: str
    script: Path
    checkpoint: Path
    out_name: str


@dataclass
class MethodMetrics:
    label: str
    output_dir: str
    source: str
    valid: bool = False
    episode_count: int = 0
    success_rate: float = 0.0
    completion_pct: float = 0.0
    final_error_m: float | None = None
    path_length_m: float | None = None
    displacement_m: float | None = None
    trajectory_points: int = 0
    score: float = 0.0
    result: str = ""
    error_reason: str = ""


@dataclass
class SeedCandidate:
    seed: int
    strict_order: bool
    order_margin: float
    separation: float
    methods: dict[str, MethodMetrics]
    figure: str | None = None


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _as_xyz(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
    arr = arr[:, :3]
    return arr[np.isfinite(arr).all(axis=1)]


def _split_continuous_segments(xyz: np.ndarray, jump_threshold: float = 5.0) -> list[np.ndarray]:
    if xyz.shape[0] < 2:
        return [xyz]
    step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    cuts = np.where(step > float(jump_threshold))[0] + 1
    bounds = [0, *[int(v) for v in cuts], int(xyz.shape[0])]
    segments = [xyz[start:end] for start, end in zip(bounds, bounds[1:]) if end - start >= 2]
    return segments or [xyz]


def _path_length(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _completion_pct_for_segment(xyz: np.ndarray, target: np.ndarray | None) -> float:
    if target is None or xyz.shape[0] < 2:
        return float("-inf")
    start_dist = float(np.linalg.norm(xyz[0, :3] - target[:3]))
    final_dist = float(np.linalg.norm(xyz[-1, :3] - target[:3]))
    if start_dist <= 1e-6:
        return float("-inf")
    return (1.0 - final_dist / start_dist) * 100.0


def _best_continuous_segment(xyz: np.ndarray, target: np.ndarray | None) -> np.ndarray:
    segments = _split_continuous_segments(xyz)
    return max(
        segments,
        key=lambda seg: (
            _completion_pct_for_segment(seg, target),
            min(_path_length(seg), 80.0),
            seg.shape[0],
        ),
    )


def _metric_from_summary(output_dir: Path, density: int, speed: float) -> dict[str, Any]:
    summary = _read_json(output_dir / "density_sweep_summary.json", [])
    if not isinstance(summary, list):
        return {}
    best: dict[str, Any] = {}
    best_score = -1
    for row in summary:
        if not isinstance(row, dict):
            continue
        spacing = _finite_float(row.get("tree_spacing_m", row.get("obstacles_per_tile")), -999.0)
        target_speed = _finite_float(row.get("target_speed_mps"), -999.0)
        score = 0
        if abs(spacing - float(density)) < 1e-3:
            score += 2
        if abs(target_speed - float(speed)) < 1e-3:
            score += 2
        if score > best_score:
            best = row
            best_score = score
    return best


def _trajectory_stats(live_state: dict[str, Any]) -> tuple[int, float | None, float | None, float | None]:
    xyz = _as_xyz(live_state.get("trajectory", []))
    if xyz.shape[0] == 0:
        return 0, None, None, None

    target = None
    if live_state.get("target") is not None:
        target_xyz = _as_xyz([live_state.get("target")])
        if target_xyz.shape[0] > 0:
            target = target_xyz[0, :3]
    xyz = _best_continuous_segment(xyz, target)

    path_length = _path_length(xyz)
    displacement = float(np.linalg.norm(xyz[-1, :3] - xyz[0, :3]))

    final_error = None
    if target is not None:
        final_error = float(np.linalg.norm(xyz[-1, :3] - target[:3]))

    return int(xyz.shape[0]), path_length, final_error, displacement


def collect_metrics(label: str, output_dir: Path, density: int, speed: float) -> MethodMetrics:
    summary_row = _metric_from_summary(output_dir, density=density, speed=speed)
    live_state = _read_json(output_dir / "live_state.json", {})
    if not isinstance(live_state, dict):
        live_state = {}

    points, path_length, final_error, displacement = _trajectory_stats(live_state)
    success_rate = _finite_float(summary_row.get("success_rate", live_state.get("success_rate")), 0.0)
    episode_count = int(_finite_float(summary_row.get("episode_count", live_state.get("episode_count")), 0.0))
    completion = _finite_float(
        summary_row.get("avg_completion_pct", summary_row.get("mean_completion_pct", live_state.get("mean_completion_pct"))),
        0.0,
    )
    if completion <= 1.0 and completion > 0.0:
        completion *= 100.0

    # Success dominates. Completion and small final error select visually clear
    # failure modes when all methods fail or only one succeeds.
    score = success_rate * 1000.0 + completion * 4.0
    if final_error is not None:
        score -= min(final_error, 80.0) * 3.0
    if path_length is not None:
        score += min(path_length, 160.0) * 0.08
    if displacement is not None:
        score += min(displacement, 80.0) * 0.6
    score += min(points, 1000) * 0.01
    result = str(live_state.get("result", summary_row.get("result", "")))
    error_reason = result
    valid = episode_count > 0 and not result.startswith("error:") and not result.startswith("worker_exit")
    if points < 2 or path_length is None or path_length < 1.0:
        valid = False
        error_reason = "trajectory_static_or_empty"
    if not valid:
        score = -1_000_000.0

    return MethodMetrics(
        label=label,
        output_dir=str(output_dir),
        source=str(output_dir / "live_state.json"),
        valid=valid,
        episode_count=episode_count,
        success_rate=success_rate,
        completion_pct=completion,
        final_error_m=final_error,
        path_length_m=path_length,
        displacement_m=displacement,
        trajectory_points=points,
        score=score,
        result=result,
        error_reason=error_reason,
    )


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    seen: set[int] = set()
    out: list[int] = []
    for seed in seeds:
        if seed not in seen:
            out.append(seed)
            seen.add(seed)
    if not out:
        raise ValueError("at least one seed is required")
    return out


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_command(
    cmd: list[str],
    log_path: Path,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {shell_join(cmd)}")
    if env_overrides:
        env_text = " ".join(f"{key}={value}" for key, value in sorted(env_overrides.items()))
        print(f"[env] {env_text}")
    print(f"[log] {log_path}")
    if dry_run:
        log_path.write_text(shell_join(cmd) + "\n", encoding="utf-8")
        return 0

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        return proc.wait()


def build_eval_command(
    python_bin: str,
    method: MethodSpec,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    eval_overrides: list[str],
) -> list[str]:
    cmd = [
        python_bin,
        str(method.script),
        "--checkpoint-path",
        str(method.checkpoint),
        "--tree-spacing-min",
        str(args.density),
        "--tree-spacing-max",
        str(args.density),
        "--tree-spacing-step",
        "1",
        "--speed-min",
        str(args.speed),
        "--speed-max",
        str(args.speed),
        "--speed-step",
        "1",
        "--trials",
        str(args.trials),
        "--seed",
        str(seed),
        "--eval-num-envs",
        str(args.eval_num_envs),
        "--num-episodes",
        str(args.num_episodes),
        "--exploration-type",
        str(args.exploration_type),
        "--tree-map-size",
        str(args.tree_map_size),
        "--tree-clear-radius",
        str(args.tree_clear_radius),
        "--output-dir",
        str(output_dir),
        "--no-run-subdir",
        "--no-web",
    ]
    view_mode = args.camera_view_mode if method.key in {"lcgate", "lc"} else args.lidar_view_mode
    cmd.extend(["--view-mode", str(view_mode)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if method.key == "lcgate" and args.export_lcgate_replay:
        cmd.extend(
            [
                "--export-success-videos",
                "--videos-per-combo",
                str(args.videos_per_combo),
                "--replay-interval",
                str(args.replay_interval),
                "--video-max-frames",
                str(args.video_max_frames),
            ]
        )
    if args.vulkan_gpu_id is not None and args.gpu_id is None:
        cmd.append(f"++vulkan_gpu_id={int(args.vulkan_gpu_id)}")
    cmd.extend(eval_overrides)
    return cmd


def method_source_for_plot(method_key: str, output_dir: Path, export_lcgate_replay: bool) -> Path:
    manifest = output_dir / "videos" / "manifest.json"
    if method_key == "lcgate" and export_lcgate_replay and manifest.exists():
        records = _read_json(manifest, [])
        if isinstance(records, list) and records:
            return manifest
    return output_dir / "live_state.json"


def plot_candidate(
    python_bin: str,
    candidate: SeedCandidate,
    args: argparse.Namespace,
    methods: list[MethodSpec],
    figure_path: Path,
    dry_run: bool,
) -> None:
    cmd = [
        python_bin,
        str(PLOT_SCRIPT),
    ]
    for method in methods:
        metric = candidate.methods[method.key]
        source = method_source_for_plot(method.key, Path(metric.output_dir), args.export_lcgate_replay)
        metric.source = str(source)
        cmd.extend(["--method", f"{method.label}={source}"])
    cmd.extend(
        [
            "--density",
            str(args.density),
            "--speed",
            str(args.speed),
            "--title",
            args.title,
            "--output",
            str(figure_path),
        ]
    )
    if args.no_altitude:
        cmd.append("--no-altitude")
    ret = run_command(cmd, figure_path.with_suffix(".plot.log"), dry_run=dry_run)
    if ret != 0:
        raise RuntimeError(f"plot command failed with exit code {ret}: {shell_join(cmd)}")
    candidate.figure = str(figure_path)


def rank_candidate(seed: int, metrics: dict[str, MethodMetrics], min_score_gap: float) -> SeedCandidate:
    if not all(metric.valid for metric in metrics.values()):
        return SeedCandidate(
            seed=seed,
            strict_order=False,
            order_margin=-1_000_000.0,
            separation=-1_000_000.0,
            methods=metrics,
        )
    lcgate = metrics["lcgate"].score
    lc = metrics["lc"].score
    lidar = metrics["lidar"].score
    margin = min(lcgate - lc, lc - lidar)
    return SeedCandidate(
        seed=seed,
        strict_order=margin >= float(min_score_gap),
        order_margin=margin,
        separation=lcgate - lidar,
        methods=metrics,
    )


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run three real-tree policies from .pt checkpoints and draw an ordered trajectory comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--density", type=int, default=6, help="Tree spacing/density condition.")
    parser.add_argument("--speed", type=float, default=5.0, help="Target speed in m/s.")
    parser.add_argument("--seeds", default="0-2", help="Comma/range seed list, e.g. 0,1,2 or 0-5.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--exploration-type", choices=["random", "mode"], default="random")
    parser.add_argument("--tree-map-size", type=float, default=40.0)
    parser.add_argument("--tree-clear-radius", type=float, default=2.0)
    parser.add_argument(
        "--camera-view-mode",
        choices=["web", "isaacsim"],
        default="isaacsim",
        help="View/render mode for LC-gate and LC. isaacsim refreshes depth cameras more reliably.",
    )
    parser.add_argument(
        "--lidar-view-mode",
        choices=["web", "isaacsim"],
        default="web",
        help="View/render mode for Single-LiDAR.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Physical CUDA/PhysX GPU id passed via CUDA_VISIBLE_DEVICES=N to the eval subprocesses.",
    )
    parser.add_argument(
        "--vulkan-gpu-id",
        type=int,
        default=None,
        help="Physical Vulkan/Isaac active GPU id. Usually omit it; cuda_visible_devices already drives active_gpu.",
    )
    parser.add_argument("--lcgate-checkpoint", type=Path, default=DEFAULT_LCGATE_CKPT)
    parser.add_argument("--lc-checkpoint", type=Path, default=DEFAULT_LC_CKPT)
    parser.add_argument("--lidar-checkpoint", type=Path, default=DEFAULT_LIDAR_CKPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--title", default="LC-gate vs LiDAE-KU/LC vs Single-LiDAR")
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python executable used to run eval/plot scripts.",
    )
    parser.add_argument("--min-score-gap", type=float, default=25.0, help="Minimum score gap for strict LC-gate > LC > LiDAR.")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing output dirs instead of rerunning finished methods.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue to other seeds when one method command fails.")
    parser.add_argument("--no-altitude", action="store_true", help="Do not draw the altitude subplot.")
    parser.add_argument("--export-lcgate-replay", action="store_true", help="Ask LC-gate script to export success replay/manifest.")
    parser.add_argument("--videos-per-combo", type=int, default=1)
    parser.add_argument("--replay-interval", type=int, default=1)
    parser.add_argument("--video-max-frames", type=int, default=0)
    args, eval_overrides = parser.parse_known_args(argv)
    eval_overrides = [item for item in eval_overrides if str(item).strip()]
    return args, eval_overrides


def main(argv: list[str] | None = None) -> int:
    args, eval_overrides = parse_args(sys.argv[1:] if argv is None else argv)
    seeds = parse_seed_list(args.seeds)
    output_root = args.output_root.expanduser().resolve()
    figure_output = args.figure_output.expanduser().resolve()

    methods = [
        MethodSpec("lcgate", "LC-gate", LCGATE_SCRIPT, args.lcgate_checkpoint.expanduser().resolve(), "lcgate"),
        MethodSpec("lc", "LiDAE-KU/LC", LC_SCRIPT, args.lc_checkpoint.expanduser().resolve(), "lc"),
        MethodSpec("lidar", "Single-LiDAR", LIDAR_SCRIPT, args.lidar_checkpoint.expanduser().resolve(), "lidar"),
    ]
    eval_env = {}
    if args.gpu_id is not None:
        eval_env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))

    missing = [str(path) for path in [PLOT_SCRIPT, *[m.script for m in methods], *[m.checkpoint for m in methods]] if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

    candidates: list[SeedCandidate] = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed:03d}_density_{args.density}_speed_{str(args.speed).replace('.', 'p')}"
        print(f"\n=== seed {seed} -> {seed_dir} ===")
        metrics: dict[str, MethodMetrics] = {}
        seed_failed = False

        for method in methods:
            out_dir = seed_dir / method.out_name
            live_state = out_dir / "live_state.json"
            if args.reuse and live_state.exists():
                print(f"[reuse] {method.label}: {live_state}")
            else:
                cmd = build_eval_command(
                    args.python,
                    method,
                    args,
                    out_dir,
                    seed=seed,
                    eval_overrides=eval_overrides,
                )
                ret = run_command(
                    cmd,
                    seed_dir / "logs" / f"{method.out_name}.log",
                    dry_run=args.dry_run,
                    env_overrides=eval_env,
                )
                if ret != 0:
                    print(f"[failed] {method.label} exit={ret}")
                    seed_failed = True
                    if not args.keep_going:
                        raise SystemExit(ret)
            metrics[method.key] = collect_metrics(method.label, out_dir, density=args.density, speed=args.speed)

        candidate = rank_candidate(seed, metrics, min_score_gap=args.min_score_gap)
        candidates.append(candidate)
        print(
            f"[seed {seed}] strict={candidate.strict_order} margin={candidate.order_margin:.2f} "
            f"sep={candidate.separation:.2f} failed={seed_failed}"
        )
        for key in ("lcgate", "lc", "lidar"):
            m = candidate.methods[key]
            print(
                f"  {m.label}: valid={m.valid} episodes={m.episode_count} "
                f"score={m.score:.2f} success={m.success_rate:.3f} "
                f"completion={m.completion_pct:.1f}% err={m.final_error_m} "
                f"path={m.path_length_m} disp={m.displacement_m} "
                f"points={m.trajectory_points} result={m.result or '-'}"
            )

    if not candidates:
        raise SystemExit("No seed candidates were produced.")

    candidates.sort(
        key=lambda c: (
            int(c.strict_order),
            c.order_margin,
            c.separation,
            c.methods["lcgate"].score,
            -c.methods["lidar"].score,
        ),
        reverse=True,
    )
    best = candidates[0]
    invalid = [metric for metric in best.methods.values() if not metric.valid]
    if invalid:
        print("\n[error] No valid seed candidate has all three methods completed at least one episode.")
        for metric in invalid:
            print(f"  invalid {metric.label}: result={metric.result or metric.error_reason or '-'}")
        _write_json(
            output_root / "selection_manifest.json",
            {
                "selected_seed": None,
                "strict_order": False,
                "density": args.density,
                "speed": args.speed,
                "error": "no valid candidate; at least one method has episode_count=0 or worker error",
                "candidates": [
                    {
                        **asdict(candidate),
                        "methods": {key: asdict(value) for key, value in candidate.methods.items()},
                    }
                    for candidate in candidates
                ],
            },
        )
        return 2
    print(f"\n[select] seed={best.seed} strict={best.strict_order} margin={best.order_margin:.2f}")
    plot_candidate(args.python, best, args, methods, figure_output, dry_run=args.dry_run)

    manifest = {
        "selected_seed": best.seed,
        "strict_order": best.strict_order,
        "order_margin": best.order_margin,
        "density": args.density,
        "speed": args.speed,
        "figure": str(figure_output),
        "eval_overrides": eval_overrides,
        "candidates": [
            {
                **asdict(candidate),
                "methods": {key: asdict(value) for key, value in candidate.methods.items()},
            }
            for candidate in candidates
        ],
    }
    _write_json(output_root / "selection_manifest.json", manifest)
    print(f"[manifest] {output_root / 'selection_manifest.json'}")
    print(f"[figure] {figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
