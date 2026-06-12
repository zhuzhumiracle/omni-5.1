#!/usr/bin/env python3
"""Generate a paper-style trajectory comparison figure for real-tree experiments.

The script overlays trajectories from different methods (for example LC-gate,
single LiDAR, and LiDAE-KU) in one top-down forest map.  Inputs can be:

  * success replay JSON files exported by realtree_sweep_camlidar_gate_video.py
  * manifest.json files that point to replay JSON files
  * result directories containing videos/manifest.json or *_replay.json files
  * live_state.json files with a trajectory field
  * eval_top*_traj_step_*.npz files from the trajectory viewer pipeline

Example:
  python plot_method_trajectory_comparison.py \
    --method LC-gate=results/realtree_sweep_camlidar_gate/lcgate_run/videos/manifest.json \
    --method Single-LiDAR=results/realtree_sweep_camlidar_gate/single_lidar_run/videos/manifest.json \
    --method LiDAE-KU=results/realtree_sweep_camlidar_gate/lidae_ku_run/videos/manifest.json \
    --density 6 --speed 5 --output figures/lc_lidar_lidae_trajectory.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "figures" / "lc_lidar_lidae_trajectory_comparison.png"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "realtree_sweep_camlidar_gate"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

METHOD_COLORS = [
    "#0f766e",  # LC-gate teal
    "#dc2626",  # single LiDAR red
    "#2563eb",  # LiDAE-KU blue
    "#9333ea",
    "#ea580c",
]


@dataclass
class Trace:
    label: str
    xyz: np.ndarray
    source: Path
    start: np.ndarray | None = None
    target: np.ndarray | None = None
    tree_instances: list[dict[str, Any]] | None = None
    obstacles: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def xy(self) -> np.ndarray:
        return self.xyz[:, :2]

    @property
    def path_length(self) -> float:
        if self.xyz.shape[0] < 2:
            return float("nan")
        return float(np.linalg.norm(np.diff(self.xyz, axis=0), axis=1).sum())

    @property
    def final_error(self) -> float:
        if self.target is None or self.xyz.shape[0] == 0:
            return float("nan")
        return float(np.linalg.norm(self.xyz[-1, :3] - self.target[:3]))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite].astype(np.float32)


def _split_continuous_segments(xyz: np.ndarray, jump_threshold: float = 5.0) -> list[tuple[int, int, np.ndarray]]:
    if xyz.shape[0] < 2:
        return [(0, xyz.shape[0], xyz)]
    step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    cuts = np.where(step > float(jump_threshold))[0] + 1
    bounds = [0, *[int(v) for v in cuts], int(xyz.shape[0])]
    segments: list[tuple[int, int, np.ndarray]] = []
    for start, end in zip(bounds, bounds[1:]):
        seg = xyz[start:end]
        if seg.shape[0] >= 2:
            segments.append((start, end, seg))
    return segments or [(0, xyz.shape[0], xyz)]


def _segment_path_length(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _completion_pct(xyz: np.ndarray, target: np.ndarray | None) -> float:
    if target is None or xyz.shape[0] < 2:
        return float("-inf")
    start_dist = float(np.linalg.norm(xyz[0, :3] - target[:3]))
    final_dist = float(np.linalg.norm(xyz[-1, :3] - target[:3]))
    if start_dist <= 1e-6:
        return float("-inf")
    return (1.0 - final_dist / start_dist) * 100.0


def _select_continuous_segment(
    xyz: np.ndarray,
    target: np.ndarray | None,
    *,
    jump_threshold: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    segments = _split_continuous_segments(xyz, jump_threshold=jump_threshold)
    if len(segments) <= 1:
        return xyz, {
            "segment_start": 0,
            "segment_end": int(xyz.shape[0]),
            "segment_count": len(segments),
            "segment_completion_pct": _completion_pct(xyz, target),
        }

    ranked = sorted(
        segments,
        key=lambda item: (
            _completion_pct(item[2], target),
            min(_segment_path_length(item[2]), 80.0),
            item[2].shape[0],
        ),
        reverse=True,
    )
    start, end, seg = ranked[0]
    return seg, {
        "segment_start": int(start),
        "segment_end": int(end),
        "segment_count": len(segments),
        "segment_completion_pct": _completion_pct(seg, target),
    }


def _path_from_maybe_absolute(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _speed_tag(speed: float) -> str:
    text = f"{float(speed):.1f}".replace(".", "p")
    return text


def _candidate_score(data: dict[str, Any], path: Path, density: int | None, speed: float | None) -> tuple[int, float, str]:
    score = 0
    if density is not None:
        spacing = _float_or_none(data.get("tree_spacing_m"))
        if spacing is None:
            spacing = _float_or_none(data.get("tree_config", {}).get("spacing"))
        if spacing is not None and abs(spacing - float(density)) < 1e-3:
            score += 20
        elif re.search(rf"density_{int(density)}(?:_|$)", path.name):
            score += 12
    if speed is not None:
        replay_speed = _float_or_none(data.get("target_speed_mps"))
        if replay_speed is not None and abs(replay_speed - float(speed)) < 1e-3:
            score += 20
        elif f"speed_{_speed_tag(speed)}" in path.name:
            score += 12
    frame_count = len(data.get("frames") or data.get("trajectory") or [])
    return (score, float(frame_count), str(path))


def _iter_manifest_replays(manifest_path: Path) -> Iterable[Path]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list):
        return []
    out: list[Path] = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        replay = item.get("replay_json")
        if replay:
            replay_path = _path_from_maybe_absolute(replay, manifest_path.parent)
            if replay_path.exists():
                out.append(replay_path)
    return out


def resolve_replay_path(path: Path, density: int | None, speed: float | None) -> Path:
    """Resolve a file/manifest/directory into the best matching replay JSON."""
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "manifest.json":
        candidates = list(_iter_manifest_replays(path))
    elif path.is_file():
        return path
    elif path.is_dir():
        candidates = []
        for manifest in sorted(path.glob("**/manifest.json")):
            candidates.extend(_iter_manifest_replays(manifest))
        candidates.extend(sorted(path.glob("**/*_replay.json")))
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    unique: dict[Path, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate in unique:
            continue
        try:
            data = _read_json(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            unique[candidate] = data

    if not unique:
        raise FileNotFoundError(f"No replay JSON found under: {path}")

    ranked = sorted(
        unique.items(),
        key=lambda item: _candidate_score(item[1], item[0], density, speed),
        reverse=True,
    )
    return ranked[0][0]


def load_json_trace(label: str, path: Path, density: int | None, speed: float | None) -> Trace:
    replay_path = resolve_replay_path(path, density=density, speed=speed)
    data = _read_json(replay_path)
    if not isinstance(data, dict):
        raise ValueError(f"JSON input must contain an object: {replay_path}")

    target = _as_xyz([data["target"]])[0] if "target" in data else None

    if isinstance(data.get("frames"), list):
        xyz = _as_xyz([frame.get("pos") for frame in data["frames"] if isinstance(frame, dict)])
    elif "planned_trajectory" in data:
        xyz = _as_xyz(data["planned_trajectory"])
    elif "trajectory" in data:
        xyz = _as_xyz(data["trajectory"])
        xyz, segment_meta = _select_continuous_segment(xyz, target)
    elif "positions" in data:
        xyz = _as_xyz(data["positions"])
    else:
        raise ValueError(f"No trajectory-like field found in: {replay_path}")

    if xyz.shape[0] == 0:
        raise ValueError(f"Trajectory is empty after filtering invalid points: {replay_path}")

    start = _as_xyz([data["start"]])[0] if "start" in data else xyz[0]
    tree_instances = data.get("tree_instances")
    obstacles = data.get("obstacles")
    metadata = {
        "tree_spacing_m": data.get("tree_spacing_m", data.get("tree_config", {}).get("spacing")),
        "target_speed_mps": data.get("target_speed_mps"),
        "seed": data.get("seed"),
        "episode": data.get("episode"),
        "trial": data.get("trial"),
    }
    metadata.update(locals().get("segment_meta", {}))
    return Trace(
        label=label,
        xyz=xyz,
        source=replay_path,
        start=np.asarray(start, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32) if target is not None else None,
        tree_instances=tree_instances if isinstance(tree_instances, list) else None,
        obstacles=obstacles if isinstance(obstacles, list) else None,
        metadata=metadata,
    )


def load_npz_trace(label: str, path: Path, rank: int) -> Trace:
    data = np.load(path, allow_pickle=True)
    xyz_all = np.asarray(data["xyz"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool) if "valid" in data else np.isfinite(xyz_all).all(axis=-1)
    if xyz_all.ndim != 3 or xyz_all.shape[-1] < 3:
        raise ValueError(f"NPZ xyz must have shape [N,T,3+], got {xyz_all.shape}: {path}")

    idx = int(np.clip(rank, 0, xyz_all.shape[0] - 1))
    xyz = _as_xyz(xyz_all[idx][valid[idx]])
    obstacles = None
    if "obstacle_env_points" in data:
        pts = _as_xyz(data["obstacle_env_points"])
        obstacles = [{"x": float(x), "y": float(y), "width": 0.35, "height": 0.35} for x, y in pts[:, :2]]
    elif "obstacle_points" in data:
        pts = _as_xyz(data["obstacle_points"][idx])
        obstacles = [{"x": float(x), "y": float(y), "width": 0.35, "height": 0.35} for x, y in pts[:, :2]]

    return Trace(label=label, xyz=xyz, source=path, start=xyz[0], obstacles=obstacles)


def load_trace(method_spec: str, density: int | None, speed: float | None, npz_rank: int) -> Trace:
    if "=" in method_spec:
        label, raw_path = method_spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
    else:
        raw_path = method_spec.strip()
        label = Path(raw_path).stem
    if not label:
        raise ValueError(f"Method label is empty in spec: {method_spec!r}")
    path = Path(raw_path)
    if path.suffix.lower() == ".npz":
        return load_npz_trace(label, path.expanduser().resolve(), rank=npz_rank)
    return load_json_trace(label, path, density=density, speed=speed)


def _first_nonempty_tree_instances(traces: list[Trace]) -> list[dict[str, Any]]:
    for trace in traces:
        if trace.tree_instances:
            return trace.tree_instances
    return []


def _first_nonempty_obstacles(traces: list[Trace]) -> list[dict[str, Any]]:
    for trace in traces:
        if trace.obstacles:
            return trace.obstacles
    return []


def _axis_limits(traces: list[Trace], margin: float) -> tuple[float, float, float, float]:
    chunks = [trace.xy for trace in traces if trace.xy.size]
    for trace in traces:
        if trace.start is not None:
            chunks.append(trace.start.reshape(1, 3)[:, :2])
        if trace.target is not None:
            chunks.append(trace.target.reshape(1, 3)[:, :2])
    if not chunks:
        return -30.0, 30.0, -30.0, 30.0
    xy = np.concatenate(chunks, axis=0)
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    if abs(xmax - xmin) < 1e-6:
        xmin -= 1.0
        xmax += 1.0
    if abs(ymax - ymin) < 1e-6:
        ymin -= 1.0
        ymax += 1.0
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def _shared_point(points: list[np.ndarray], tolerance: float = 1.0) -> np.ndarray | None:
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float32).reshape(len(points), -1)
    xy = arr[:, :2]
    center = np.nanmean(xy, axis=0)
    if np.nanmax(np.linalg.norm(xy - center[None, :], axis=1)) > float(tolerance):
        return None
    return arr[0]


def _draw_forest(ax: Any, traces: list[Trace], tree_alpha: float) -> None:
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    tree_instances = _first_nonempty_tree_instances(traces)
    if tree_instances:
        trunks = []
        canopies = []
        for item in tree_instances:
            pos = item.get("position", [item.get("x", 0.0), item.get("y", 0.0), 0.0])
            if len(pos) < 2:
                continue
            x, y = float(pos[0]), float(pos[1])
            scale = float(item.get("scale", 1.0))
            trunks.append(patches.Circle((x, y), radius=max(0.08, 0.16 * scale)))
            canopies.append(patches.Circle((x, y), radius=max(0.28, 0.78 * scale)))
        if canopies:
            ax.add_collection(PatchCollection(canopies, facecolor="#8ccf85", edgecolor="none", alpha=0.16 * tree_alpha))
        if trunks:
            ax.add_collection(PatchCollection(trunks, facecolor="#5f4b32", edgecolor="none", alpha=0.58 * tree_alpha))
        return

    obstacles = _first_nonempty_obstacles(traces)
    if obstacles:
        rects = []
        for item in obstacles:
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            w = float(item.get("width", 0.6))
            h = float(item.get("height", 0.6))
            rects.append(patches.Rectangle((x - 0.5 * w, y - 0.5 * h), w, h))
        ax.add_collection(PatchCollection(rects, facecolor="#6b7280", edgecolor="none", alpha=0.35 * tree_alpha))


def plot_figure(
    traces: list[Trace],
    output: Path,
    *,
    title: str,
    margin: float,
    dpi: int,
    tree_alpha: float,
    show_altitude: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if show_altitude:
        fig = plt.figure(figsize=(8.0, 8.8), dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[4.7, 1.15], hspace=0.16)
        ax = fig.add_subplot(gs[0])
        ax_alt = fig.add_subplot(gs[1])
        fig.subplots_adjust(right=0.74)
    else:
        fig, ax = plt.subplots(figsize=(8.0, 7.4), dpi=dpi)
        ax_alt = None
        fig.subplots_adjust(right=0.74)

    _draw_forest(ax, traces, tree_alpha=tree_alpha)
    xmin, xmax, ymin, ymax = _axis_limits(traces, margin=margin)

    legend_handles: list[Any] = []
    for i, trace in enumerate(traces):
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        xy = trace.xy
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.65, alpha=0.96, solid_capstyle="round")
        ax.scatter(xy[0, 0], xy[0, 1], s=36, color=color, marker="o", edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=42, color=color, marker="s", edgecolors="white", linewidths=0.8, zorder=5)
        path_text = f"{trace.path_length:.1f} m" if math.isfinite(trace.path_length) else "-"
        err = trace.final_error
        err_text = f", err {err:.1f} m" if math.isfinite(err) else ""
        legend_handles.append(Line2D([0], [0], color=color, lw=2.65, label=f"{trace.label} ({path_text}{err_text})"))

        if ax_alt is not None:
            dist = np.zeros((trace.xyz.shape[0],), dtype=np.float32)
            if trace.xyz.shape[0] > 1:
                dist[1:] = np.cumsum(np.linalg.norm(np.diff(trace.xyz[:, :2], axis=0), axis=1))
            ax_alt.plot(dist, trace.xyz[:, 2], color=color, linewidth=2.0, label=trace.label)

    shared_start = _shared_point([trace.start for trace in traces if trace.start is not None])
    shared_target = _shared_point([trace.target for trace in traces if trace.target is not None])
    if shared_start is not None:
        s = np.asarray(shared_start, dtype=np.float32)
        ax.scatter(s[0], s[1], s=78, marker="*", color="#111827", edgecolors="white", linewidths=0.9, zorder=6)
        ax.text(s[0], s[1] - 1.2, "Start", ha="center", va="top", fontsize=9, color="#111827")
    if shared_target is not None:
        t = np.asarray(shared_target, dtype=np.float32)
        ax.scatter(t[0], t[1], s=96, marker="X", color="#f59e0b", edgecolors="#111827", linewidths=0.9, zorder=6)
        ax.text(t[0], t[1] + 1.2, "Goal", ha="center", va="bottom", fontsize=9, color="#111827")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, color="#d1d5db", linewidth=0.65, alpha=0.65)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.94,
        fontsize=8.5,
    )

    if ax_alt is not None:
        ax_alt.set_xlabel("horizontal path distance (m)")
        ax_alt.set_ylabel("z (m)")
        ax_alt.grid(True, color="#d1d5db", linewidth=0.65, alpha=0.65)
        ax_alt.margins(x=0.02)

    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _default_method_if_available() -> list[str]:
    if not DEFAULT_RESULTS_ROOT.exists():
        return []
    candidates = sorted(
        DEFAULT_RESULTS_ROOT.glob("**/*_replay.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return []
    return [f"LC-gate={candidates[0]}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay LC-gate, single-LiDAR, and LiDAE-KU trajectories in one forest-map figure."
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Method trace input. PATH may be replay JSON, manifest.json, result directory, live_state.json, or NPZ.",
    )
    parser.add_argument("--density", type=int, default=None, help="Prefer replays with this tree spacing/density.")
    parser.add_argument("--speed", type=float, default=None, help="Prefer replays with this target speed.")
    parser.add_argument("--npz-rank", type=int, default=0, help="Trajectory rank for NPZ inputs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path (.png recommended).")
    parser.add_argument("--title", default="Trajectory comparison in the same real-tree scene")
    parser.add_argument("--margin", type=float, default=4.0, help="Meters of plot margin around trajectories.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--tree-alpha", type=float, default=1.0)
    parser.add_argument("--no-altitude", action="store_true", help="Do not draw the altitude profile panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_specs = args.method or _default_method_if_available()
    if not method_specs:
        raise SystemExit(
            "No method inputs were provided and no default replay was found. "
            "Use --method LC-gate=<path> --method Single-LiDAR=<path> --method LiDAE-KU=<path>."
        )

    traces = [
        load_trace(spec, density=args.density, speed=args.speed, npz_rank=int(args.npz_rank))
        for spec in method_specs
    ]
    plot_figure(
        traces,
        Path(args.output),
        title=str(args.title),
        margin=float(args.margin),
        dpi=int(args.dpi),
        tree_alpha=float(args.tree_alpha),
        show_altitude=not bool(args.no_altitude),
    )

    print(f"saved: {Path(args.output).expanduser().resolve()}")
    for trace in traces:
        meta = trace.metadata or {}
        spacing = meta.get("tree_spacing_m", "-")
        speed = meta.get("target_speed_mps", "-")
        completion = meta.get("segment_completion_pct")
        completion_text = f" completion={float(completion):.1f}%" if completion is not None and math.isfinite(float(completion)) else ""
        print(
            f"  {trace.label}: points={trace.xyz.shape[0]} "
            f"path={trace.path_length:.2f}m{completion_text} spacing={spacing} speed={speed} source={trace.source}"
        )


if __name__ == "__main__":
    main()
