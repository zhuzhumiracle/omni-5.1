#!/usr/bin/env python3
"""Geometry checks for forest_lc_gate camera/LiDAR/yaw alignment.

This script intentionally does not start Isaac Sim. It validates the static
configuration math that must agree before training:
  - visual nose/body forward axis
  - depth camera optical axis
  - LiDAR yaw-bin center used by camera-risk sectors
  - Lee controller yaw offset for non-+X body forward axes
"""

from __future__ import annotations

import ast
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_CFG = REPO_ROOT / "OmniDrones" / "cfg" / "task" / "forest_lc_gate.yaml"


def _strip_comment(line: str) -> str:
    in_quote = False
    quote = ""
    out = []
    for ch in line:
        if ch in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote = ch
            elif quote == ch:
                in_quote = False
        if ch == "#" and not in_quote:
            break
        out.append(ch)
    return "".join(out).strip()


def _read_flat_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw)
        if not line or ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            data[key] = ast.literal_eval(value)
        except Exception:
            lowered = value.lower()
            if lowered in {"true", "false"}:
                data[key] = lowered == "true"
            else:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data


def _vec(data: dict[str, object], key: str, default: list[float]) -> list[float]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != len(default):
        raise ValueError(f"{key} must be a {len(default)}D list, got {value!r}")
    return [float(v) for v in value]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n <= 1e-9:
        raise ValueError(f"Degenerate vector: {v}")
    return [x / n for x in v]


def _yaw(v: list[float]) -> float:
    return math.atan2(v[1], v[0])


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rot_z(v: list[float], yaw: float) -> list[float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def _assert_close(name: str, value: float, limit: float):
    if abs(value) > limit:
        raise AssertionError(f"{name}: {value:.6g} exceeds limit {limit:.6g}")


def main() -> int:
    cfg_path = DEFAULT_CFG
    cfg = _read_flat_yaml(cfg_path)

    body_forward = _normalize(_vec(cfg, "body_forward_axis", [1.0, 0.0, 0.0]))
    cam_pos = _vec(cfg, "depth_camera_pos", [0.22, 0.0, 0.18])
    cam_target = _vec(cfg, "depth_camera_target", [2.0, 0.0, 0.95])
    cam_axis = _normalize([b - a for a, b in zip(cam_pos, cam_target)])

    body_forward_xy = _normalize([body_forward[0], body_forward[1], 0.0])
    cam_axis_xy = _normalize([cam_axis[0], cam_axis[1], 0.0])
    body_yaw = _yaw(body_forward_xy)
    cam_yaw = _yaw(cam_axis_xy)
    yaw_err = _wrap_pi(cam_yaw - body_yaw)
    _assert_close("camera yaw vs body_forward yaw", yaw_err, math.radians(1.0))

    configured_depth_yaw = math.radians(float(cfg.get("depth_yaw_center_deg", math.degrees(cam_yaw))))
    _assert_close("depth_yaw_center_deg vs camera yaw", _wrap_pi(configured_depth_yaw - cam_yaw), math.radians(1.0))

    # USD/OpenGL camera convention: center ray [0, 0, -1] maps to the configured
    # optical axis, so its LiDAR yaw should be the camera yaw center.
    center_ray_body_yaw = cam_yaw
    _assert_close("center ray relative yaw", _wrap_pi(center_ray_body_yaw - cam_yaw), math.radians(0.01))

    num_yaw_bins = int(cfg.get("num_yaw_bins", 80))
    yaw_mod = (cam_yaw + 2.0 * math.pi) % (2.0 * math.pi)
    yaw_bin = int(yaw_mod / (2.0 * math.pi) * num_yaw_bins)
    yaw_bin_center = (yaw_bin + 0.5) / num_yaw_bins * 2.0 * math.pi
    yaw_bin_center = _wrap_pi(yaw_bin_center)
    _assert_close(
        "camera yaw vs nearest LiDAR yaw-bin center",
        _wrap_pi(yaw_bin_center - cam_yaw),
        math.pi / num_yaw_bins + 1e-6,
    )

    # If the desired visual nose yaw is theta, the Lee controller must receive
    # theta - body_forward_yaw because the controller defines yaw for body +X.
    for desired in [0.0, math.pi / 2.0, -math.pi / 2.0, math.pi]:
        controller_yaw = desired - body_yaw
        nose_world = _rot_z(body_forward, controller_yaw)
        actual = _yaw(nose_world)
        _assert_close("controller yaw offset", _wrap_pi(actual - desired), math.radians(0.01))

    focal = float(cfg.get("depth_camera_focal_length", 12.0))
    h_ap = float(cfg.get("depth_camera_horizontal_aperture", 20.955))
    depth_res = _vec(cfg, "depth_resolution", [96.0, 160.0])
    v_ap = h_ap * depth_res[0] / max(1.0, depth_res[1])
    v_fov = 2.0 * math.atan(v_ap / (2.0 * focal))
    cam_pitch = math.atan2(cam_axis[2], math.sqrt(cam_axis[0] * cam_axis[0] + cam_axis[1] * cam_axis[1]))
    lidar_vfov = _vec(cfg, "lidar_vfov", [-7.0, 52.0])
    overlap_min = max(cam_pitch - v_fov / 2.0, math.radians(lidar_vfov[0]))
    overlap_max = min(cam_pitch + v_fov / 2.0, math.radians(lidar_vfov[1]))
    if overlap_max <= overlap_min:
        raise AssertionError(
            "camera vertical FoV does not overlap LiDAR pitch range: "
            f"camera_pitch={math.degrees(cam_pitch):.2f}deg"
        )

    print(f"OK {cfg_path}")
    print(f"  body_forward_axis={body_forward}")
    print(f"  camera_yaw={math.degrees(cam_yaw):.2f}deg lidar_yaw_bin={yaw_bin}/{num_yaw_bins}")
    print(f"  camera_pitch={math.degrees(cam_pitch):.2f}deg overlap=[{math.degrees(overlap_min):.2f}, {math.degrees(overlap_max):.2f}]deg")
    print(f"  controller_yaw_offset={math.degrees(-body_yaw):.2f}deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
