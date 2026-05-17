#!/usr/bin/env python3
import argparse
import contextlib
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ZK_DIR = SCRIPT_DIR.parent
OMNIDRONES_DIR = ZK_DIR.parent.parent
REPO_ROOT = OMNIDRONES_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "realtree_sweep_camlidar_gate"
DEFAULT_TREE_PLY = REPO_ROOT / "YOPO" / "Simulator" / "src" / "pointcloud" / "tree.ply"
DEFAULT_TREE_OBJ = REPO_ROOT / "YOPO" / "Simulator" / "src" / "pointcloud" / "tree_mesh.obj"
DEFAULT_VLIM_CHECKPOINT = "goodpt/5-17-vlim-lcgate-tree_best_return_598.03.pt"
DEFAULT_POLICY_TASK = "forest_lc_gate"


class CanLiDARGateBackbone(torch.nn.Module):
    """Backbone matching train_canlidargate_trees.py exactly for lc-gate checkpoints."""

    def __init__(
        self,
        state_dim,
        lidar_dim=3200,
        camera_risk_dim=13,
        ku_value_max=20.0,
        output_dim=128,
        camera_h_fov_rad=None,
        fov_pitch_range=None,
        num_rows=1,
        num_cols=3,
        features_per_sector=4,
    ):
        super().__init__()
        if lidar_dim != 3200:
            raise ValueError(f"This backbone expects lidar_dim=3200, got {lidar_dim}.")
        if camera_risk_dim <= 0:
            raise ValueError(f"camera_risk_dim must be positive, got {camera_risk_dim}.")

        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.camera_risk_dim = int(camera_risk_dim)
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.features_per_sector = int(features_per_sector)
        if self.num_rows < 1 or self.num_cols < 1:
            raise ValueError(f"num_rows/num_cols must be >= 1, got {self.num_rows}×{self.num_cols}")
        self.ku_value_max = float(ku_value_max)
        if self.ku_value_max <= 0.0:
            raise ValueError(f"ku_value_max must be positive, got {self.ku_value_max}")
        self.ku_unknown_value = self.ku_value_max

        self.ku_h = 40
        self.ku_w = 80
        if camera_h_fov_rad is None:
            camera_h_fov_rad = 2.0 * math.atan(160.0 / (2.0 * 320.0))
        self.camera_h_fov_rad = float(camera_h_fov_rad)
        if fov_pitch_range is None:
            fov_pitch_range = (-math.pi / 2, math.pi / 2)
        self.fov_pitch_min = float(fov_pitch_range[0])
        self.fov_pitch_max = float(fov_pitch_range[1])
        self._build_sector_2d_masks()

        self.ku_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(4, 16),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(8, 32),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(8, 64),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )
        self.ku_global_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
        )
        total_sectors = self.num_rows * self.num_cols
        self.gate_heads = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(self.features_per_sector, 1), torch.nn.Sigmoid())
            for _ in range(total_sectors)
        ])

        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(64 + 128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.ELU(),
        )

    def _build_sector_2d_masks(self):
        """Precompute 2D boolean masks [ku_h, ku_w] for each (row, col) sector."""
        h_fov = self.camera_h_fov_rad
        p_min = self.fov_pitch_min
        p_max = self.fov_pitch_max

        col_yaws = (torch.arange(self.ku_w, dtype=torch.float32) + 0.5) / self.ku_w * 2.0 * math.pi
        col_yaws = torch.remainder(col_yaws + math.pi, 2.0 * math.pi) - math.pi
        row_pitches = (torch.arange(self.ku_h, dtype=torch.float32) + 0.5) / self.ku_h * math.pi - math.pi / 2.0

        yaw_grid = col_yaws.unsqueeze(0).expand(self.ku_h, -1)
        pitch_grid = row_pitches.unsqueeze(1).expand(-1, self.ku_w)

        fov_mask = (
            (yaw_grid >= -h_fov / 2.0) & (yaw_grid <= h_fov / 2.0)
            & (pitch_grid >= p_min) & (pitch_grid <= p_max)
        )

        self._sector_masks = torch.nn.ParameterList()
        for ri in range(self.num_rows):
            p_left = p_min + ri * (p_max - p_min) / self.num_rows
            p_right = p_min + (ri + 1) * (p_max - p_min) / self.num_rows
            for ci in range(self.num_cols):
                y_left = -h_fov / 2.0 + ci * h_fov / self.num_cols
                y_right = -h_fov / 2.0 + (ci + 1) * h_fov / self.num_cols
                mask = (
                    fov_mask
                    & (yaw_grid >= y_left) & (yaw_grid <= y_right)
                    & (pitch_grid >= p_left) & (pitch_grid <= p_right)
                )
                self._sector_masks.append(torch.nn.Parameter(mask.bool(), requires_grad=False))

    def forward(self, obs):
        state = obs[..., :self.state_dim]
        x_ku_flat = obs[..., self.state_dim:self.state_dim + self.lidar_dim]
        camera_risk_start = self.state_dim + self.lidar_dim
        camera_risk_end = camera_risk_start + self.camera_risk_dim
        camera_risk = obs[..., camera_risk_start:camera_risk_end]

        batch_shape = state.shape[:-1]
        b = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1

        x_ku_raw = x_ku_flat.reshape(b, 1, self.ku_h, self.ku_w)
        x_ku_raw = torch.nan_to_num(
            x_ku_raw,
            posinf=self.ku_unknown_value,
            neginf=0.0,
            nan=self.ku_unknown_value,
        )
        x_ku_raw = torch.clamp(x_ku_raw, 0.0, self.ku_unknown_value)

        camera_risk_2d = camera_risk.reshape(b, self.camera_risk_dim)
        camera_risk_2d = torch.nan_to_num(camera_risk_2d, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        total_sectors = self.num_rows * self.num_cols
        sector_feat_dim = total_sectors * self.features_per_sector
        sector_features = camera_risk_2d[:, :sector_feat_dim].reshape(b, total_sectors, self.features_per_sector)

        spatial_gate = torch.ones(b, 1, self.ku_h, self.ku_w, device=x_ku_raw.device, dtype=x_ku_raw.dtype)
        for i in range(total_sectors):
            gate_i = self.gate_heads[i](sector_features[:, i, :])
            spatial_gate[:, :, :, self._sector_masks[i]] = gate_i.view(b, 1, 1, 1)

        lidar_feat = self.ku_encoder((x_ku_raw * spatial_gate) / self.ku_value_max)
        lidar_z = self.ku_global_head(lidar_feat)

        state_2d = state.reshape(b, self.state_dim)
        state_z = self.state_encoder(state_2d)

        fused = torch.cat([state_z, lidar_z], dim=-1)
        out = self.fusion_mlp(fused)
        return out.reshape(*batch_shape, -1)


def _camera_h_fov_rad_from_cfg(cfg):
    focal = float(cfg.task.get("depth_camera_focal_length", 12.0))
    h_aperture = float(cfg.task.get("depth_camera_horizontal_aperture", 20.955))
    if focal <= 0.0 or h_aperture <= 0.0:
        raise ValueError(
            f"Invalid depth camera intrinsics for FoV: focal={focal}, horizontal_aperture={h_aperture}"
        )
    return 2.0 * math.atan(h_aperture / (2.0 * focal))


def _inject_canlidargate_backbone(policy, base_env, env, cfg):
    obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
    lidar_dim = 3200
    _num_rows = int(cfg.task.get("camera_risk_num_rows", 0))
    _num_cols = int(cfg.task.get("camera_risk_num_cols", 0))
    _num_bins = int(cfg.task.get("camera_risk_num_bins", 3))
    if _num_rows <= 0 and _num_cols <= 0:
        num_rows, num_cols = 1, _num_bins
    elif _num_rows <= 0:
        num_rows, num_cols = 1, _num_cols
    elif _num_cols <= 0:
        num_rows, num_cols = _num_rows, 1
    else:
        num_rows, num_cols = _num_rows, _num_cols
    features_per_sector = int(cfg.task.get("camera_risk_features_per_bin", 4))
    total_sectors = num_rows * num_cols
    expected_camera_risk_dim = total_sectors * features_per_sector
    if bool(cfg.task.get("camera_risk_add_stale_ratio", True)):
        expected_camera_risk_dim += 1

    state_dim = int(obs_dim - lidar_dim - expected_camera_risk_dim)
    camera_risk_dim = int(obs_dim - state_dim - lidar_dim)
    if state_dim <= 0 or camera_risk_dim <= 0:
        raise RuntimeError(
            f"Invalid observation split: obs_dim={obs_dim}, state_dim={state_dim}, "
            f"lidar_dim={lidar_dim}, camera_risk_dim={camera_risk_dim}"
        )
    if camera_risk_dim != expected_camera_risk_dim:
        print(
            f"[Warning] camera_risk_dim={camera_risk_dim} derived from observation, "
            f"but config expects {expected_camera_risk_dim}; using derived dimension."
        )

    expected_feature_dim = 128
    ku_value_max = float(cfg.task.get("ku_value_max", 20.0))
    camera_h_fov_rad = _camera_h_fov_rad_from_cfg(cfg)
    # Compute effective pitch range for 2D masks
    focal = float(cfg.task.get("depth_camera_focal_length", 12.0))
    v_aperture = float(cfg.task.get("depth_camera_vertical_aperture", 0.0))
    depth_h_cfg = int(cfg.task.get("depth_resolution", [96, 160])[0])
    if v_aperture <= 0:
        h_aperture = float(cfg.task.get("depth_camera_horizontal_aperture", 20.955))
        v_aperture = h_aperture * depth_h_cfg / max(1, int(cfg.task.get("depth_resolution", [96, 160])[1]))
    camera_v_fov_rad = 2.0 * math.atan(v_aperture / (2.0 * focal))
    _lidar_vfov = cfg.task.get("lidar_vfov", [-7., 52.])
    lidar_pitch_min = math.radians(float(_lidar_vfov[0]))
    lidar_pitch_max = math.radians(float(_lidar_vfov[1]))
    fov_pitch_min = max(-camera_v_fov_rad / 2.0, lidar_pitch_min)
    fov_pitch_max = min(camera_v_fov_rad / 2.0, lidar_pitch_max)

    actor_backbone = CanLiDARGateBackbone(
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        camera_risk_dim=camera_risk_dim,
        ku_value_max=ku_value_max,
        output_dim=expected_feature_dim,
        camera_h_fov_rad=camera_h_fov_rad,
        fov_pitch_range=(fov_pitch_min, fov_pitch_max),
        num_rows=num_rows,
        num_cols=num_cols,
        features_per_sector=features_per_sector,
    ).to(base_env.device)
    critic_backbone = CanLiDARGateBackbone(
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        camera_risk_dim=camera_risk_dim,
        ku_value_max=ku_value_max,
        output_dim=expected_feature_dim,
        camera_h_fov_rad=camera_h_fov_rad,
        fov_pitch_range=(fov_pitch_min, fov_pitch_max),
        num_rows=num_rows,
        num_cols=num_cols,
        features_per_sector=features_per_sector,
    ).to(base_env.device)

    actor_replaced = False
    critic_replaced = False
    if hasattr(policy.actor, "module") and hasattr(policy.actor.module, "module"):
        actor_core = policy.actor.module.module
        if isinstance(actor_core, torch.nn.Sequential) and len(actor_core) > 0:
            actor_core[0] = actor_backbone
            actor_replaced = True
    if not actor_replaced and hasattr(policy.actor, "module") and hasattr(policy.actor.module, "__getitem__"):
        try:
            actor_td_module = policy.actor.module[0]
            if hasattr(actor_td_module, "module") and isinstance(actor_td_module.module, torch.nn.Sequential):
                actor_td_module.module[0] = actor_backbone
                actor_replaced = True
        except Exception:
            pass

    if hasattr(policy.critic, "module") and isinstance(policy.critic.module, torch.nn.Sequential):
        policy.critic.module[0] = critic_backbone
        critic_replaced = True
    if not critic_replaced and hasattr(policy.critic, "module") and hasattr(policy.critic.module, "__getitem__"):
        try:
            critic_td_module = policy.critic.module[0]
            if hasattr(critic_td_module, "module"):
                critic_td_module.module = critic_backbone
                critic_replaced = True
        except Exception:
            pass

    if not actor_replaced or not critic_replaced:
        raise RuntimeError(f"Backbone injection failed: actor={actor_replaced}, critic={critic_replaced}")

    print(
        f"[+] train_canlidargate_trees-compatible backbone injected | "
        f"task={cfg.task.name}, state_dim={state_dim}, lidar_dim={lidar_dim}, "
        f"camera_risk_dim={camera_risk_dim}, grid={num_rows}×{num_cols}, "
        f"camera_h_fov_rad={camera_h_fov_rad:.4f}"
    )
    return actor_backbone, critic_backbone


def make_int_range(min_value, max_value, step):
    min_value = int(min_value)
    max_value = int(max_value)
    step = int(step)
    if step <= 0:
        raise ValueError("obstacle step must be positive")
    if max_value < min_value:
        raise ValueError("obstacle max must be >= min")
    return list(range(min_value, max_value + 1, step))


def make_float_range(min_value, max_value, step):
    min_value = float(min_value)
    max_value = float(max_value)
    step = float(step)
    if step <= 0.0:
        raise ValueError("speed step must be positive")
    if max_value < min_value:
        raise ValueError("speed max must be >= min")
    values = []
    value = min_value
    while value <= max_value + 1e-9:
        values.append(round(value, 3))
        value += step
    return values


def make_speed_values(args):
    if args.speed_min is not None or args.speed_max is not None:
        min_speed = args.speed if args.speed_min is None else args.speed_min
        max_speed = args.speed if args.speed_max is None else args.speed_max
        return make_float_range(min_speed, max_speed, args.speed_step)
    return [round(float(args.speed), 3)]


def _speed_tag(speed):
    return str(round(float(speed), 3)).replace("-", "m").replace(".", "p")


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _finite_float(value, default=float("nan")):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _weighted_success_mean(rows, key):
    total_weight = 0
    weighted_sum = 0.0
    for row in rows:
        value = _finite_float(row.get(key))
        weight = int(row.get("success_count", 0) or 0)
        if weight <= 0 or not np.isfinite(value):
            continue
        weighted_sum += value * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else float("nan")


def read_ply_xyz(path, max_source_points=200_000):
    """Read XYZ vertices from a binary/ascii PLY file without requiring open3d."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        fmt = ""
        vertex_count = None
        vertex_props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append((parts[1], parts[2]))

        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        prop_names = [name for _ptype, name in vertex_props]
        try:
            x_idx, y_idx, z_idx = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
        except ValueError as exc:
            raise ValueError(f"PLY vertex properties must include x/y/z: {path}") from exc

        if fmt == "binary_little_endian":
            dtype_fields = []
            type_map = {
                "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
                "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
                "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
                "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            }
            for i, (ptype, name) in enumerate(vertex_props):
                if ptype not in type_map:
                    raise ValueError(f"Unsupported PLY property type {ptype!r} in {path}")
                dtype_fields.append((name or f"prop_{i}", type_map[ptype]))
            arr = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
            points = np.stack([arr[prop_names[x_idx]], arr[prop_names[y_idx]], arr[prop_names[z_idx]]], axis=1)
        elif fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                parts = f.readline().decode("ascii", errors="replace").split()
                if len(parts) < len(vertex_props):
                    continue
                rows.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            points = np.asarray(rows, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")

    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] > int(max_source_points):
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=int(max_source_points), replace=False)
        points = points[idx]
    return points


def tree_positions_jittered_grid(map_size=60.0, spacing=4.0, seed=0, clear_radius=2.0):
    """Poisson-like jittered grid positions matching YOPO's tree_dist intent."""
    rng = np.random.default_rng(int(seed))
    map_size = float(map_size)
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("tree spacing must be positive")
    half = 0.5 * map_size
    coords = np.arange(-half + 0.5 * spacing, half, spacing, dtype=np.float32)
    positions = []
    jitter = min(0.35 * spacing, 0.5 * max(spacing - 0.8, 0.0))
    clear_points = np.asarray([[0.0, -24.0], [0.0, 24.0], [0.0, 0.0]], dtype=np.float32)
    for x in coords:
        for y in coords:
            px = float(np.clip(x + rng.uniform(-jitter, jitter), -half + 0.3, half - 0.3))
            py = float(np.clip(y + rng.uniform(-jitter, jitter), -half + 0.3, half - 0.3))
            if clear_radius > 0:
                d2 = np.sum((clear_points - np.asarray([px, py], dtype=np.float32)) ** 2, axis=1)
                if np.any(d2 <= clear_radius * clear_radius):
                    continue
            positions.append((px, py))
    return np.asarray(positions, dtype=np.float32)


def _rotation_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _parse_face_index(token, vertex_count):
    raw = token.split("/", 1)[0]
    if not raw:
        raise ValueError(f"Invalid OBJ face token: {token!r}")
    idx = int(raw)
    if idx < 0:
        idx = vertex_count + idx
    else:
        idx = idx - 1
    return idx


def read_obj_mesh(path, max_faces=0):
    """Read vertices and triangulated faces from a simple OBJ mesh."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Tree OBJ file not found: {path}")

    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                if max_faces and len(faces) >= int(max_faces):
                    continue
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                idxs = [_parse_face_index(tok, len(vertices)) for tok in parts]
                for i in range(1, len(idxs) - 1):
                    if max_faces and len(faces) >= int(max_faces):
                        break
                    faces.append((idxs[0], idxs[i], idxs[i + 1]))

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"OBJ mesh contains no usable vertices/faces: {path}")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError(f"OBJ mesh has out-of-range face indices: {path}")
    return vertices, faces


def normalize_tree_mesh(vertices, auto_upright=True):
    """Match train_canlidargate_trees.py: z-up, centered in XY, rooted at z=0."""
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected OBJ vertices with shape [N,3], got {vertices.shape}")

    if auto_upright:
        extents = np.ptp(vertices, axis=0)
        up_axis = int(np.argmax(extents))
        if up_axis == 1:
            vertices = vertices[:, [0, 2, 1]]
        elif up_axis == 0:
            vertices = vertices[:, [1, 2, 0]]

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center_xy = 0.5 * (mins[:2] + maxs[:2])
    vertices[:, 0] -= center_xy[0]
    vertices[:, 1] -= center_xy[1]
    vertices[:, 2] -= vertices[:, 2].min()
    return vertices.astype(np.float32)


def make_combined_tree_forest_mesh(
    tree_obj_path,
    map_size=60.0,
    spacing=4.0,
    seed=0,
    scale_min=0.5,
    scale_max=1.0,
    tilt_deg=10.0,
    clear_radius=2.0,
    max_faces_per_tree=8000,
    auto_upright=True,
):
    """Instantiate one OBJ tree mesh many times and return one combined mesh."""
    base_vertices, base_faces = read_obj_mesh(tree_obj_path, max_faces=int(max_faces_per_tree))
    base_vertices = normalize_tree_mesh(base_vertices, auto_upright=auto_upright)

    positions = tree_positions_jittered_grid(
        map_size=float(map_size),
        spacing=float(spacing),
        seed=int(seed),
        clear_radius=float(clear_radius),
    )
    if positions.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), positions

    rng = np.random.default_rng(int(seed))
    scale_min = float(scale_min)
    scale_max = float(scale_max)
    if scale_max < scale_min:
        scale_min, scale_max = scale_max, scale_min
    max_tilt_rad = math.radians(float(tilt_deg))

    all_vertices = []
    all_faces = []
    vert_offset = 0
    for px, py in positions:
        scale = float(rng.uniform(scale_min, scale_max))
        roll = float(rng.uniform(-max_tilt_rad, max_tilt_rad))
        pitch = float(rng.uniform(-max_tilt_rad, max_tilt_rad))
        yaw = float(rng.uniform(-math.pi, math.pi))
        rot = _rotation_matrix(roll, pitch, yaw)
        verts = (base_vertices @ rot.T) * scale
        verts += np.asarray([float(px), float(py), 0.0], dtype=np.float32)
        verts[:, 2] = np.maximum(verts[:, 2], 0.02)
        all_vertices.append(verts.astype(np.float32))
        all_faces.append((base_faces + vert_offset).astype(np.int32))
        vert_offset += verts.shape[0]

    return (
        np.concatenate(all_vertices, axis=0).astype(np.float32),
        np.concatenate(all_faces, axis=0).astype(np.int32),
        positions,
    )


def make_realtree_forest_points(
    tree_ply,
    map_size=60.0,
    spacing=4.0,
    seed=0,
    points_per_tree=320,
    scale_min=0.5,
    scale_max=1.0,
    tilt_deg=10.0,
    clear_radius=2.0,
):
    base_points = read_ply_xyz(tree_ply)
    base_points = base_points - np.asarray(
        [np.mean(base_points[:, 0]), np.mean(base_points[:, 1]), np.min(base_points[:, 2])],
        dtype=np.float32,
    )
    rng = np.random.default_rng(int(seed))
    if base_points.shape[0] > int(points_per_tree):
        base_points = base_points[rng.choice(base_points.shape[0], size=int(points_per_tree), replace=False)]
    # Random subsampling may miss the lowest source vertices; plant each sampled tree on z=0.
    base_points[:, 2] -= float(np.min(base_points[:, 2]))
    positions = tree_positions_jittered_grid(map_size=map_size, spacing=spacing, seed=seed, clear_radius=clear_radius)
    forest = []
    max_tilt = math.radians(float(tilt_deg))
    for px, py in positions:
        scale = float(rng.uniform(float(scale_min), float(scale_max)))
        roll = float(rng.uniform(-max_tilt, max_tilt))
        pitch = float(rng.uniform(-max_tilt, max_tilt))
        yaw = float(rng.uniform(-math.pi, math.pi))
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        pts = (base_points * scale) @ (rz @ ry @ rx).T
        pts += np.asarray([px, py, 0.0], dtype=np.float32)
        pts[:, 2] = np.maximum(pts[:, 2], 0.02)
        forest.append(pts.astype(np.float32))
    if not forest:
        return np.zeros((0, 3), dtype=np.float32), positions
    return np.concatenate(forest, axis=0), positions


def surfel_cross_mesh(points, surfel_size=0.08):
    """Convert tree points into small crossed triangle surfels for USD/render/raycast."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    s = float(surfel_size) * 0.5
    offsets = np.asarray(
        [
            [-s, 0.0, -s], [s, 0.0, -s], [s, 0.0, s], [-s, 0.0, s],
            [0.0, -s, -s], [0.0, s, -s], [0.0, s, s], [0.0, -s, s],
        ],
        dtype=np.float32,
    )
    vertices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    base = (np.arange(points.shape[0], dtype=np.int32) * 8)[:, None]
    local_faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    faces = (base[:, None, :] + local_faces[None, :, :]).reshape(-1, 3)
    return vertices.astype(np.float32), faces.astype(np.int32)


_DEATH_REASON_KEYS = [
    "death_z_low", "death_z_high", "death_overspeed",
    "death_collision", "death_contact", "death_oob",
    "death_flip", "death_nan",
]
_DEATH_REASON_LABELS = [
    "z_low", "z_high", "overspeed",
    "collision", "contact", "out_of_bounds",
    "flip", "nan",
]


def _extract_death_reason_from_stats(stats_td, env_idx):
    """Extract a single death reason string for one env from the final stats TensorDict."""
    try:
        success_val = float(stats_td["success"][env_idx].item())
    except (KeyError, IndexError):
        success_val = 0.0
    if success_val >= 0.5:
        return "success"
    for key, label in zip(_DEATH_REASON_KEYS, _DEATH_REASON_LABELS):
        try:
            if float(stats_td[key][env_idx].item()) >= 0.5:
                return label
        except (KeyError, IndexError):
            continue
    return "timeout"


def _aggregate_death_reasons(reasons):
    """Aggregate a list of per-episode death reasons into a summary string."""
    from collections import Counter
    counts = Counter(reasons)
    return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def _get_final_stats_from_td(td):
    """Return the most recent stats TensorDict from either next/current scope."""
    for key in (("next", "stats"), "stats"):
        try:
            stats_td = td.get(key)
            if stats_td is not None:
                return stats_td
        except Exception:
            continue
    return None


def write_results(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "density_sweep_results.csv"
    fieldnames = [
        "obstacles_per_tile",
        "tree_spacing_m",
        "target_speed_mps",
        "tree_count",
        "trial",
        "seed",
        "success_rate",
        "success_count",
        "episode_count",
        "mean_return",
        "mean_episode_len",
        "mean_completion_pct",
        "mean_arrival_time_s",
        "mean_path_length_m",
        "mean_speed_mps",
        "death_reason",
        "result",
        "duration_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = []
    groups = sorted(
        {
            (
                round(_finite_float(r.get("target_speed_mps"), float("nan")), 3),
                int(r["obstacles_per_tile"]),
            )
            for r in rows
        },
        key=lambda item: (item[0] if np.isfinite(item[0]) else -1.0, item[1]),
    )
    for target_speed, density in groups:
        combo = [
            r
            for r in rows
            if round(_finite_float(r.get("target_speed_mps"), float("nan")), 3) == target_speed
            and int(r["obstacles_per_tile"]) == density
        ]
        if not combo:
            continue
        total_success = sum(int(r.get("success_count", 0)) for r in combo)
        total_episodes = sum(int(r.get("episode_count", 0)) for r in combo)
        avg_completion = float(np.mean([float(r.get("mean_completion_pct", 0.0)) for r in combo]))
        summary.append(
            {
                "target_speed_mps": target_speed,
                "obstacles_per_tile": density,
                "tree_spacing_m": float(np.mean([_finite_float(r.get("tree_spacing_m"), density) for r in combo])),
                "tree_count": int(round(float(np.mean([_finite_float(r.get("tree_count"), 0) for r in combo])))),
                "trials": len(combo),
                "seeds": sorted({int(r["seed"]) for r in combo}),
                "success_count": total_success,
                "episode_count": total_episodes,
                "success_rate": total_success / max(1, total_episodes),
                "mean_return": float(np.mean([float(r.get("mean_return", 0.0)) for r in combo])),
                "mean_episode_len": float(np.mean([float(r.get("mean_episode_len", 0.0)) for r in combo])),
                "avg_completion_pct": round(avg_completion, 1),
                "mean_arrival_time_s": round(_weighted_success_mean(combo, "mean_arrival_time_s"), 3),
                "mean_path_length_m": round(_weighted_success_mean(combo, "mean_path_length_m"), 3),
                "mean_speed_mps": round(_weighted_success_mean(combo, "mean_speed_mps"), 3),
            }
        )
    write_json(output_dir / "density_sweep_summary.json", summary)
    if not summary:
        return csv_path
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        xs = [x["target_speed_mps"] for x in summary]
        labels = [f"sp={_finite_float(x.get('tree_spacing_m', x.get('obstacles_per_tile'))):g}m" for x in summary]
        ys = [x["success_rate"] * 100.0 for x in summary]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        ax.plot(xs, ys, marker="o", linewidth=2.0, color="#2563eb")
        if len({x.get("obstacles_per_tile") for x in summary}) > 1:
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("Target speed (m/s)")
        ax.set_ylabel("Success rate (%)")
        ax.set_title("OmniDrones Cam+LiDAR Policy Real-Tree Sweep")
        ax.set_ylim(-2, 102)
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_dir / "success_rate.png")
        plt.close(fig)

        # ---- completion percentage chart ----
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        xs2 = [x["target_speed_mps"] for x in summary]
        ys2 = [x["avg_completion_pct"] for x in summary]
        ax2.plot(xs2, ys2, marker="o", linewidth=2.0, color="#16a34a")
        ax2.set_xlabel("Target speed (m/s)")
        ax2.set_ylabel("Avg completion (%)")
        ax2.set_title("OmniDrones Cam+LiDAR Policy Real-Tree Sweep - Completion %")
        ax2.set_ylim(-2, 102)
        ax2.set_xlim(min(xs2) - 0.5, max(xs2) + 0.5)
        ax2.grid(True, alpha=0.35)
        fig2.tight_layout()
        fig2.savefig(output_dir / "completion_pct.png")
        plt.close(fig2)

        metric_specs = [
            ("mean_arrival_time_s", "arrival_time.png", "Avg Arrival Time (success only)", "Avg arrival time (s)", "#dc2626"),
            ("mean_path_length_m", "path_length.png", "Avg Path Length (success only)", "Avg path length (m)", "#9333ea"),
            ("mean_speed_mps", "mean_speed.png", "Avg Speed (success only)", "Avg speed (m/s)", "#ea580c"),
        ]
        for key, filename, title_suffix, ylabel, color in metric_specs:
            metric_summary = [x for x in summary if np.isfinite(_finite_float(x.get(key)))]
            if not metric_summary:
                continue
            fig_m, ax_m = plt.subplots(figsize=(7.2, 4.2), dpi=140)
            xs_m = [x["target_speed_mps"] for x in metric_summary]
            ys_m = [x[key] for x in metric_summary]
            ax_m.plot(xs_m, ys_m, marker="o", linewidth=2.0, color=color)
            ax_m.set_xlabel("Target speed (m/s)")
            ax_m.set_ylabel(ylabel)
            ax_m.set_title(f"OmniDrones Cam+LiDAR Policy Real-Tree Sweep - {title_suffix}")
            ax_m.set_xlim(min(xs_m) - 0.5, max(xs_m) + 0.5)
            ax_m.grid(True, alpha=0.35)
            fig_m.tight_layout()
            fig_m.savefig(output_dir / filename)
            plt.close(fig_m)
    except Exception as exc:
        print(f"[realtree sweep] plot skipped: {exc}")
    return csv_path


def _sanitize_run_label(label):
    cleaned = re.sub(r"[^A-Za-z0-9._+=-]+", "_", str(label or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "run"


def _extract_override_value(hydra_overrides, key):
    prefix = f"{key}="
    for token in reversed(list(hydra_overrides or [])):
        if str(token).startswith(prefix):
            return str(token)[len(prefix):].strip().strip("\"'")
    return ""


def _has_override(hydra_overrides, key):
    prefix = f"{key}="
    return any(str(token).startswith(prefix) for token in hydra_overrides or [])


def _checkpoint_path_from_play_yaml():
    cfg_path = ZK_DIR / "play_camlidar.yaml"
    if not cfg_path.exists():
        return ""
    try:
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("checkpoint_path:"):
                return stripped.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        return ""
    return ""


def infer_run_label(args, hydra_overrides):
    checkpoint_path = str(getattr(args, "checkpoint_path", "") or "").strip()
    if not checkpoint_path:
        checkpoint_path = _extract_override_value(hydra_overrides, "checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = _checkpoint_path_from_play_yaml()
    if not checkpoint_path:
        return ""

    name = Path(checkpoint_path).name or Path(checkpoint_path).stem
    if name.endswith(".pt"):
        name = Path(name).stem
    return _sanitize_run_label(name)


def _resolve_requested_cuda_visible(cfg, hydra_overrides):
    explicit_override = _extract_override_value(hydra_overrides, "cuda_visible_devices")
    if explicit_override:
        return explicit_override
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value:
        return env_value
    return str(cfg.get("cuda_visible_devices", "0")).strip()


def _resolve_precompose_cuda_visible(hydra_overrides):
    explicit_override = _extract_override_value(hydra_overrides, "cuda_visible_devices")
    if explicit_override:
        return explicit_override
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value:
        return env_value
    return "0"


def _first_visible_cuda_device(value):
    first_token = str(value or "").split(",")[0].strip()
    if not first_token:
        return None
    try:
        return int(first_token)
    except ValueError:
        return None


def make_run_output_dir(base_output_dir, run_subdir=True, run_label=""):
    base_output_dir = Path(base_output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(run_subdir):
        return base_output_dir

    if run_label:
        stem = _sanitize_run_label(run_label)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        stem = f"run_{stamp}"

    candidate = base_output_dir / stem
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base_output_dir / f"{stem}_{suffix:02d}"
            suffix += 1


WEB_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OmniDrones Real-Tree Sweep</title>
  <style>
    html, body { margin: 0; height: 100%; background: #101418; color: #e8edf2; font: 14px system-ui, sans-serif; }
    #wrap { display: grid; grid-template-columns: 1fr 320px; height: 100%; }
    canvas { width: 100%; height: 100%; display: block; background: #111820; }
    aside { border-left: 1px solid #2b3540; padding: 14px; background: #151b22; overflow: auto; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    .metric { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid #26313c; }
    .label { color: #95a3b3; }
    .value { font-weight: 650; text-align: right; }
    .ok { color: #65d889; } .bad { color: #ff7878; } .run { color: #7cb7ff; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
    th, td { border-bottom: 1px solid #26313c; padding: 5px 3px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
  </style>
</head>
<body>
<div id="wrap">
  <canvas id="view"></canvas>
  <aside>
    <h1>OmniDrones Real-Tree Sweep</h1>
    <div id="metrics"></div>
    <table><thead><tr><th>spacing</th><th>target</th><th>trees</th><th>success</th><th>actual</th></tr></thead><tbody id="summary"></tbody></table>
  </aside>
</div>
<script>
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const metrics = document.getElementById("metrics");
const summaryEl = document.getElementById("summary");
function resize(){ canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; }
addEventListener("resize", resize); resize();
function worldToCanvas(p){
  const w = canvas.width, h = canvas.height;
  const sx = w / 44, sy = h / 66, s = Math.min(sx, sy);
  return [w/2 + p[0]*s, h/2 - p[1]*s];
}
function drawGrid(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.strokeStyle = "#22303a";
  for(let x=-20; x<=20; x+=4){ const a=worldToCanvas([x,-30]), b=worldToCanvas([x,30]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }
  for(let y=-30; y<=30; y+=4){ const a=worldToCanvas([-20,y]), b=worldToCanvas([20,y]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }
}
function dot(p, r, color){ const q=worldToCanvas(p); ctx.fillStyle=color; ctx.beginPath(); ctx.arc(q[0], q[1], r*devicePixelRatio, 0, Math.PI*2); ctx.fill(); }
function line(points, color, width){
  if(!points || points.length < 2) return;
  ctx.strokeStyle=color; ctx.lineWidth=width*devicePixelRatio; ctx.beginPath();
  points.forEach((p,i)=>{ const q=worldToCanvas(p); if(i===0) ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]); });
  ctx.stroke();
}
function render(state){
  drawGrid();
  const obstacles = state.obstacles || [];
  ctx.fillStyle = "rgba(148, 163, 184, 0.34)";
  ctx.strokeStyle = "rgba(203, 213, 225, 0.55)";
  ctx.lineWidth = 1 * devicePixelRatio;
  for(const o of obstacles){
    const a = worldToCanvas([o.x - o.width/2, o.y - o.height/2]);
    const b = worldToCanvas([o.x + o.width/2, o.y + o.height/2]);
    const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]);
    const w = Math.abs(b[0] - a[0]), h = Math.abs(b[1] - a[1]);
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }
  const lidar = state.lidar_points || [];
  for(const p of lidar) dot(p, 1.2, "rgba(180,190,200,0.28)");
  line(state.trajectory || [], "#67b7ff", 2);
  if(state.target) dot(state.target, 7, "#6ee787");
  if(state.position) dot(state.position, 6, "#ffcf5a");
  const cls = state.phase === "running" ? "run" : (state.result === "success" ? "ok" : (state.result ? "bad" : ""));
  const rows = [
    ["phase", `<span class="${cls}">${state.phase || "waiting"}</span>`],
    ["tree spacing", state.tree_spacing_m ?? state.obstacles_per_tile ?? "-"],
    ["target speed", state.target_speed_mps == null ? "-" : `${Number(state.target_speed_mps).toFixed(2)} m/s`],
    ["tree count", state.tree_count ?? "-"],
    ["trial", `${state.trial ?? "-"} / ${state.trials ?? "-"}`],
    ["seed", state.seed ?? "-"],
    ["step", state.step ?? "-"],
    ["success", state.success_rate == null ? "-" : (state.success_rate*100).toFixed(1)+"%"],
    ["position", state.position ? state.position.map(x=>x.toFixed(2)).join(", ") : "-"],
    ["target", state.target ? state.target.map(x=>x.toFixed(2)).join(", ") : "-"],
    ["result", state.result || "-"]
  ];
  metrics.innerHTML = rows.map(r=>`<div class="metric"><span class="label">${r[0]}</span><span class="value">${r[1]}</span></div>`).join("");
  const summary = state.summary || [];
  summaryEl.innerHTML = summary.map(s=>{
    const speed = Number.isFinite(s.mean_speed_mps) ? s.mean_speed_mps.toFixed(2) : "-";
    const target = Number.isFinite(s.target_speed_mps) ? s.target_speed_mps.toFixed(2) : "-";
    return `<tr><td>${s.tree_spacing_m ?? s.obstacles_per_tile}</td><td>${target}</td><td>${s.tree_count ?? "-"}</td><td>${(s.success_rate*100).toFixed(1)}%</td><td>${speed}</td></tr>`;
  }).join("");
}
async function tick(){
  try { const r = await fetch("/state", {cache:"no-store"}); render(await r.json()); } catch(e) {}
  requestAnimationFrame(()=>setTimeout(tick, 250));
}
tick();
</script>
</body>
</html>"""


def start_web_server(host, port, live_state_path, summary_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                data = WEB_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/state"):
                state = read_json(live_state_path, {})
                if not state.get("obstacles") and int(state.get("obstacles_per_tile", 0) or 0) > 0:
                    state["obstacles"] = make_preview_obstacles(
                        int(state.get("obstacles_per_tile", 0) or 0),
                        int(state.get("seed", 0) or 0),
                    )
                state["summary"] = read_json(summary_path, [])
                data = json.dumps(state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@contextlib.contextmanager
def patched_realtree_forest(args):
    import importlib
    import isaaclab.terrains as terrains
    ray_caster_mod = importlib.import_module("isaaclab.sensors.ray_caster.ray_caster")

    original_obstacle_cfg = terrains.HfDiscreteObstaclesTerrainCfg
    original_terrain_init = terrains.TerrainImporter.__init__
    original_ray_init_meshes = ray_caster_mod.RayCaster._initialize_warp_meshes

    def obstacle_cfg_wrapper(*args, **kwargs):
        kwargs["num_obstacles"] = 0
        kwargs["obstacle_height_mode"] = "fixed"
        return original_obstacle_cfg(*args, **kwargs)

    def install_realtree_mesh():
        import omni.usd  # type: ignore
        from pxr import Sdf, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable while installing real-tree forest")

        tree_obj_path = Path(str(args.tree_ply)).expanduser().resolve()
        if not tree_obj_path.exists():
            raise FileNotFoundError(f"Tree OBJ file not found: {tree_obj_path}")

        vertices, faces, positions = make_combined_tree_forest_mesh(
            tree_obj_path,
            map_size=float(args.tree_map_size),
            spacing=float(args.worker_density),
            seed=int(args.worker_seed),
            scale_min=float(args.tree_scale_min),
            scale_max=float(args.tree_scale_max),
            tilt_deg=float(args.tree_tilt_deg),
            clear_radius=float(args.tree_clear_radius),
            max_faces_per_tree=int(args.tree_max_faces_per_tree),
            auto_upright=bool(args.tree_auto_upright),
        )
        if vertices.size == 0 or faces.size == 0:
            raise RuntimeError("No real-tree mesh was generated. Check tree spacing/map size/clear radius.")

        mesh_path = "/World/ground/realtree_forest_mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr([tuple(map(float, p)) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * int(faces.shape[0]))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        mesh.CreateDoubleSidedAttr(True)

        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("realtree:tree_count", Sdf.ValueTypeNames.Int).Set(int(positions.shape[0]))
        prim.CreateAttribute("realtree:vertices", Sdf.ValueTypeNames.Int).Set(int(vertices.shape[0]))
        prim.CreateAttribute("realtree:faces", Sdf.ValueTypeNames.Int).Set(int(faces.shape[0]))
        prim.CreateAttribute("realtree:spacing_m", Sdf.ValueTypeNames.Double).Set(float(args.worker_density))

        print(
            "[realtree] installed combined tree mesh: "
            f"spacing={float(args.worker_density):.3f}m trees={positions.shape[0]} "
            f"vertices={vertices.shape[0]} faces={faces.shape[0]} "
            f"max_faces_per_tree={int(args.tree_max_faces_per_tree)}"
        )
        return int(positions.shape[0])

    def terrain_init_wrapper(self, cfg):
        original_terrain_init(self, cfg)
        install_realtree_mesh()

    def ray_initialize_all_meshes(self):
        import omni.usd  # type: ignore
        import warp as wp
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils
        from isaaclab.terrains.trimesh.utils import make_plane
        from isaaclab.utils.warp import convert_to_warp_mesh

        if len(self.cfg.mesh_prim_paths) != 1:
            return original_ray_init_meshes(self)

        for mesh_prim_path in self.cfg.mesh_prim_paths:
            plane_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            if plane_prim is not None:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                self.meshes[mesh_prim_path] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                continue

            root = sim_utils.find_first_matching_prim(mesh_prim_path)
            if root is None or not root.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")

            all_points = []
            all_faces = []
            vert_offset = 0
            stack = [root]
            while stack:
                curr = stack.pop()
                stack.extend(list(curr.GetChildren()))
                if curr.GetTypeName() != "Mesh":
                    continue
                mesh = UsdGeom.Mesh(curr)
                pts_attr = mesh.GetPointsAttr().Get()
                idx_attr = mesh.GetFaceVertexIndicesAttr().Get()
                counts_attr = mesh.GetFaceVertexCountsAttr().Get()
                if pts_attr is None or idx_attr is None or counts_attr is None:
                    continue
                pts = np.asarray(pts_attr, dtype=np.float32)
                if pts.size == 0:
                    continue
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh)).T
                pts = np.matmul(pts, transform_matrix[:3, :3].T)
                pts += transform_matrix[:3, 3]
                idx = np.asarray(idx_attr, dtype=np.int32)
                counts = np.asarray(counts_attr, dtype=np.int32)
                cursor = 0
                tris = []
                for count in counts:
                    count = int(count)
                    poly = idx[cursor:cursor + count]
                    cursor += count
                    if count < 3:
                        continue
                    for j in range(1, count - 1):
                        tris.append([poly[0], poly[j], poly[j + 1]])
                if not tris:
                    continue
                all_points.append(pts.astype(np.float32))
                all_faces.append(np.asarray(tris, dtype=np.int32) + vert_offset)
                vert_offset += pts.shape[0]

            if not all_points:
                raise RuntimeError(f"No Mesh children found under raycast path: {mesh_prim_path}")
            points = np.concatenate(all_points, axis=0).astype(np.float32)
            faces = np.concatenate(all_faces, axis=0).astype(np.int32)
            self.meshes[mesh_prim_path] = convert_to_warp_mesh(points, faces, device=self.device)
            wp.synchronize()
            print(
                f"[realtree] RayCaster combined {len(all_points)} meshes under {mesh_prim_path}: "
                f"vertices={points.shape[0]} faces={faces.shape[0]}"
            )

    terrains.HfDiscreteObstaclesTerrainCfg = obstacle_cfg_wrapper
    terrains.TerrainImporter.__init__ = terrain_init_wrapper
    ray_caster_mod.RayCaster._initialize_warp_meshes = ray_initialize_all_meshes
    try:
        yield
    finally:
        terrains.HfDiscreteObstaclesTerrainCfg = original_obstacle_cfg
        terrains.TerrainImporter.__init__ = original_terrain_init
        ray_caster_mod.RayCaster._initialize_warp_meshes = original_ray_init_meshes


def _sample_first_env_lidar_points(base_env, max_points=800):
    try:
        import torch

        hits = base_env.lidar.data.ray_hits_w.reshape(base_env.num_envs, base_env.num_lidar_points, 3)[0]
        hits = hits.detach()
        finite = torch.isfinite(hits).all(dim=-1)
        hits = hits[finite]
        if hits.numel() == 0:
            return []
        if hits.shape[0] > max_points:
            idx = torch.linspace(0, hits.shape[0] - 1, max_points, device=hits.device).long()
            hits = hits[idx]
        return [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for x, y, z in hits.cpu().tolist()]
    except Exception:
        return []


def make_preview_obstacles(tree_spacing_m, seed, map_size=60.0, clear_radius=2.0, max_boxes=1200):
    """Preview real-tree instance positions as small top-down canopy boxes."""
    positions = tree_positions_jittered_grid(
        map_size=map_size,
        spacing=float(tree_spacing_m),
        seed=seed,
        clear_radius=float(clear_radius),
    )
    boxes = []
    for px, py in positions[:max_boxes]:
        boxes.append(
            {
                "x": round(float(px), 3),
                "y": round(float(py), 3),
                "width": 0.8,
                "height": 0.8,
            }
        )
    return boxes


def run_worker(args, hydra_overrides):
    sys.path.insert(0, str(ZK_DIR))
    sys.path.insert(0, str(OMNIDRONES_DIR))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    requested_visible = _resolve_precompose_cuda_visible(hydra_overrides)
    if requested_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_visible
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # PPO is registered through Hydra's ConfigStore by this import; there is no
    # cfg/algo/ppo.yaml on disk in this repo.
    import omni_drones.learning.ppo.ppo  # noqa: F401

    isaacsim_view = str(getattr(args, "view_mode", "web")).lower() == "isaacsim"
    overrides = [f"hydra.searchpath=[file://{OMNIDRONES_DIR / 'cfg'}]"]
    if args.policy_task and not _has_override(hydra_overrides, "task"):
        overrides.append(f"task={args.policy_task}")
    overrides += list(hydra_overrides)
    overrides += [
        f"seed={int(args.worker_seed)}",
        f"eval_num_envs={int(args.eval_num_envs)}",
        f"num_episodes={int(args.num_episodes)}",
        f"++task.vlim={float(args.worker_speed)}",
        f"++task.vlim_train_min={float(args.vlim_train_min)}",
        f"++task.vlim_train_max={float(args.vlim_train_max)}",
        f"++task.observe_vlim={'true' if bool(args.observe_vlim) else 'false'}",
        f"++task.vlim_randomize={'true' if bool(args.vlim_randomize_eval) else 'false'}",
        "final_eval_rounds=1",
        f"headless={'false' if isaacsim_view else 'true'}",
        f"enable_viewport={'true' if isaacsim_view else 'false'}",
        "task.show_depth_preview_window=false",
    ]
    if args.max_steps is not None:
        overrides.append(f"max_steps={int(args.max_steps)}")
    if args.checkpoint_path:
        overrides.append(f"checkpoint_path={args.checkpoint_path}")

    with initialize_config_dir(version_base=None, config_dir=str(ZK_DIR), job_name="density_sweep_worker"):
        cfg = compose(config_name="play_camlidar", overrides=overrides)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    if args.max_steps is None:
        cfg_max_steps = None
        if "env" in cfg:
            cfg_max_steps = cfg.env.get("max_episode_length", None)
        if cfg_max_steps is None:
            cfg_max_steps = cfg.get("max_steps", None)
        args.max_steps = int(cfg_max_steps if cfg_max_steps is not None else 1500)
    else:
        args.max_steps = int(args.max_steps)
    cfg.max_steps = int(args.max_steps)

    requested_visible = _resolve_requested_cuda_visible(cfg, hydra_overrides)

    visible_physical_gpu = _first_visible_cuda_device(requested_visible)
    sim_gpu_index = int(cfg.get("sim_gpu_index", 0))
    logical_cuda_gpu = 0 if requested_visible else sim_gpu_index
    physical_vulkan_gpu = visible_physical_gpu if visible_physical_gpu is not None else sim_gpu_index

    import torch
    from torchrl.data import CompositeSpec
    from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

    from omni_drones import init_simulation_app
    from omni_drones.learning import ALGOS
    from omni_drones.utils.torchrl.transforms import FromDiscreteAction, FromMultiDiscreteAction, ravel_composite

    from play_camlidar import (
        _load_checkpoint_strictish,
        _preflight_runtime_checks,
        _resolve_checkpoint_path,
        _select_first_env_value,
    )

    exploration_name = str(args.exploration_type).lower()
    if exploration_name == "mode":
        exploration_type = ExplorationType.MODE
    elif exploration_name == "random":
        exploration_type = ExplorationType.RANDOM
    else:
        raise ValueError(f"Unsupported exploration type: {args.exploration_type}")

    cfg.task.obstacles_per_tile = 0
    if "env" in cfg:
        cfg.env.num_envs = int(args.eval_num_envs)
    if "task" in cfg and "env" in cfg.task:
        cfg.task.env.num_envs = int(args.eval_num_envs)
    cfg.cuda_visible_devices = requested_visible
    cfg.sim_gpu_index = logical_cuda_gpu
    cfg.sim.device = f"cuda:{logical_cuda_gpu}"
    cfg.sim.active_gpu = physical_vulkan_gpu
    cfg.sim.physics_gpu = logical_cuda_gpu
    cfg.task.vlim = float(args.worker_speed)
    cfg.task.vlim_train_min = float(args.vlim_train_min)
    cfg.task.vlim_train_max = float(args.vlim_train_max)
    cfg.task.observe_vlim = bool(args.observe_vlim)
    cfg.task.vlim_randomize = bool(args.vlim_randomize_eval)
    action_dim = int(cfg.task.get("velocity_action_dim", 4 if str(cfg.task.get("control_mode", "")).lower() == "velocity" else 4))
    if str(cfg.task.get("control_mode", "rotor")).lower() == "velocity":
        cfg.task.state_dim = 10 + action_dim + (1 if bool(cfg.task.get("observe_vlim", False)) else 0)
    cfg.headless = not isaacsim_view
    cfg.enable_viewport = isaacsim_view
    cfg.sim.enable_viewport = isaacsim_view
    cfg.sim.enable_replicator = True

    live_state_path = Path(args.live_state)
    result_path = Path(args.worker_result)
    trajectory = []
    preview_obstacles = make_preview_obstacles(
        float(args.worker_density),
        int(args.worker_seed),
        map_size=float(args.tree_map_size),
        clear_radius=float(args.tree_clear_radius),
    )
    tree_count = len(preview_obstacles)
    start_time = time.time()
    write_json(
        live_state_path,
        {
            "phase": "starting",
            "obstacles_per_tile": int(args.worker_density),
            "tree_spacing_m": float(args.worker_density),
            "target_speed_mps": float(args.worker_speed),
            "tree_count": int(tree_count),
            "trial": int(args.worker_trial),
            "trials": int(args.trials),
            "seed": int(args.worker_seed),
            "trajectory": trajectory,
            "obstacles": preview_obstacles,
        },
    )

    _preflight_runtime_checks(cfg, int(args.eval_num_envs))
    simulation_app = None
    try:
        simulation_app = init_simulation_app(cfg)
        with patched_realtree_forest(args):
            from omni_drones.envs.isaac_env import IsaacEnv
            import importlib

            task_name = str(cfg.task.name)
            try:
                importlib.import_module(f"omni_drones.envs.single.{task_name.lower()}")
            except ModuleNotFoundError:
                pass

            env_class = IsaacEnv.REGISTRY[cfg.task.name]
            base_env = env_class(cfg, headless=cfg.headless)

        transforms = [InitTracker()]
        if cfg.task.get("ravel_obs", False):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
        if cfg.task.get("ravel_obs_central", False):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))
        if (
            cfg.task.get("flatten_intrinsics", True)
            and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
            and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
        ):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

        action_transform = cfg.task.get("action_transform", None)
        if action_transform is not None:
            if action_transform.startswith("multidiscrete"):
                transforms.append(FromMultiDiscreteAction(nbins=int(action_transform.split(":")[1])))
            elif action_transform.startswith("discrete"):
                transforms.append(FromDiscreteAction(nbins=int(action_transform.split(":")[1])))
            else:
                raise NotImplementedError(f"Unknown action transform: {action_transform}")

        env = TransformedEnv(base_env, Compose(*transforms)).eval()
        env.set_seed(int(args.worker_seed))
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
        _inject_canlidargate_backbone(policy, base_env, env, cfg)
        checkpoint_path = _resolve_checkpoint_path(cfg.get("checkpoint_path"))
        _load_checkpoint_strictish(policy, checkpoint_path, base_env.device)
        policy.eval()
        base_env.enable_render(isaacsim_view)
        base_env.eval()
        env.eval()

        def _run_one_trial(trial_idx):
            """Run one trial (num_episodes episodes) and return the result row dict."""
            trial_start = time.time()
            episode_success = []
            episode_returns = []
            episode_lengths = []
            episode_completion_pct = []
            episode_arrival_times = []
            episode_path_lengths = []
            episode_speeds = []
            episode_death_reasons = []
            latest_success_rate = 0.0
            trajectory = []
            with torch.no_grad(), set_exploration_type(exploration_type):
                for ep in range(int(args.num_episodes)):
                    td = env.reset()
                    num_envs_eval = int(base_env.num_envs)
                    ep_death_reasons = [None] * num_envs_eval
                    # Capture start / target positions for completion_pct
                    init_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)
                    target_pos = base_env.target_pos.detach().clone().reshape(num_envs_eval, 3)
                    start_dist = torch.norm(target_pos - init_pos, dim=-1)
                    final_positions = torch.full_like(init_pos, float("nan"))
                    prev_positions = init_pos.clone()
                    path_lengths = torch.zeros(num_envs_eval, dtype=torch.float32, device=base_env.device)
                    arrival_steps = torch.full(
                        (num_envs_eval,),
                        -1,
                        dtype=torch.int32,
                        device=base_env.device,
                    )
                    finish_steps = torch.full(
                        (num_envs_eval,),
                        int(args.max_steps),
                        dtype=torch.int32,
                        device=base_env.device,
                    )
                    finished = torch.zeros(num_envs_eval, dtype=torch.bool, device=base_env.device)
                    ep_returns = torch.zeros(num_envs_eval, dtype=torch.float32, device=base_env.device)
                    ep_success = torch.zeros(num_envs_eval, dtype=torch.int32, device=base_env.device)
                    step_count = 0

                    for step in range(int(args.max_steps)):
                        step_count = step + 1
                        td = policy(td)
                        td = env.step(td)
                        reward = td[("next", "agents", "reward")].reshape(-1).float()
                        done = td[("next", "done")].reshape(-1).to(torch.bool)
                        stats_td = td[("next", "stats")]
                        current_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)

                        active = ~finished
                        if active.any():
                            path_lengths[active] += torch.norm(current_pos[active] - prev_positions[active], dim=-1)
                            prev_positions[active] = current_pos[active]

                        # Capture final positions for envs that just finished
                        just_finished = torch.logical_and(~finished, done)
                        if just_finished.any():
                            final_positions[just_finished] = current_pos[just_finished]
                            finish_steps[just_finished] = step_count
                            for e in torch.nonzero(just_finished, as_tuple=False).flatten().detach().cpu().tolist():
                                try:
                                    ep_death_reasons[int(e)] = _extract_death_reason_from_stats(stats_td, int(e))
                                except Exception:
                                    ep_death_reasons[int(e)] = "unknown"

                        if active.any():
                            ep_returns[active] += reward[active]
                        finished = torch.logical_or(finished, done)
                        if "success" in stats_td.keys():
                            success_now = (stats_td["success"].reshape(-1).float() >= 0.5).to(torch.bool)
                            first_success = torch.logical_and(success_now, arrival_steps < 0)
                            if first_success.any():
                                arrival_steps[first_success] = step_count
                            ep_success = torch.maximum(ep_success, success_now.to(torch.int32))

                        if step % int(args.web_update_interval) == 0 or finished.all():
                            try:
                                pos = _select_first_env_value(base_env.drone.pos).detach().cpu().reshape(-1)[:3].tolist()
                                target = _select_first_env_value(base_env.target_pos).detach().cpu().reshape(-1)[:3].tolist()
                                pos = [round(float(v), 3) for v in pos]
                                target = [round(float(v), 3) for v in target]
                                trajectory.append(pos)
                                trajectory = trajectory[-600:]
                                latest_success_rate = (sum(episode_success) + int(ep_success.sum().item())) / max(
                                    1, len(episode_success) + num_envs_eval
                                )
                                write_json(
                                    live_state_path,
                                    {
                                        "phase": "running",
                                        "obstacles_per_tile": int(args.worker_density),
                                        "tree_spacing_m": float(args.worker_density),
                                        "target_speed_mps": float(args.worker_speed),
                                        "tree_count": int(tree_count),
                                        "trial": trial_idx,
                                        "trials": int(args.trials),
                                        "seed": int(args.worker_seed),
                                        "episode": ep + 1,
                                        "episodes": int(args.num_episodes),
                                        "step": step_count,
                                        "position": pos,
                                        "target": target,
                                        "trajectory": trajectory,
                                        "obstacles": preview_obstacles,
                                        "lidar_points": _sample_first_env_lidar_points(base_env),
                                        "success_rate": latest_success_rate,
                                    },
                                )
                            except Exception:
                                pass

                        if finished.all():
                            break
                        td = step_mdp(td)

                    # For envs that never finished, use current position
                    never_finished = torch.isnan(final_positions).any(dim=-1)
                    if never_finished.any():
                        current_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)
                        final_positions[never_finished] = current_pos[never_finished]
                        for e in torch.nonzero(never_finished, as_tuple=False).flatten().detach().cpu().tolist():
                            if ep_death_reasons[int(e)] is None:
                                ep_death_reasons[int(e)] = "timeout"

                    # Compute per-env completion_pct
                    final_dist = torch.norm(target_pos - final_positions, dim=-1)
                    completion_pct = torch.clamp(
                        (1.0 - final_dist / start_dist.clamp_min(1e-6)) * 100.0, 0.0, 100.0
                    )
                    success_mask = ep_success.bool()
                    sim_dt = float(getattr(base_env, "dt", 0.02))
                    arrival_times = arrival_steps.to(torch.float32) * sim_dt
                    episode_speeds_tensor = path_lengths / arrival_times.clamp_min(1e-6)

                    episode_success.extend(int(v) for v in ep_success.detach().cpu().tolist())
                    episode_returns.extend(float(v) for v in ep_returns.detach().cpu().tolist())
                    episode_lengths.extend(int(v) for v in finish_steps.detach().cpu().tolist())
                    episode_completion_pct.extend(float(v) for v in completion_pct.detach().cpu().tolist())
                    episode_arrival_times.extend(
                        float(v) for v in arrival_times[success_mask].detach().cpu().tolist()
                    )
                    episode_path_lengths.extend(
                        float(v) for v in path_lengths[success_mask].detach().cpu().tolist()
                    )
                    episode_speeds.extend(
                        float(v) for v in episode_speeds_tensor[success_mask].detach().cpu().tolist()
                    )

                    # ---- extract/fill death reasons for this episode ----
                    final_stats = _get_final_stats_from_td(td)
                    for e in range(num_envs_eval):
                        if ep_death_reasons[e] is None:
                            if final_stats is None:
                                ep_death_reasons[e] = "unknown"
                            else:
                                try:
                                    ep_death_reasons[e] = _extract_death_reason_from_stats(final_stats, e)
                                except Exception:
                                    ep_death_reasons[e] = "unknown"
                    episode_death_reasons.extend(ep_death_reasons)

            success_count = int(sum(episode_success))
            episode_count = int(len(episode_success))
            mean_completion = float(np.mean(episode_completion_pct)) if episode_completion_pct else 0.0
            death_reason = _aggregate_death_reasons(episode_death_reasons) if episode_death_reasons else ""
            return {
                "obstacles_per_tile": int(args.worker_density),
                "tree_spacing_m": float(args.worker_density),
                "target_speed_mps": float(args.worker_speed),
                "tree_count": int(tree_count),
                "trial": trial_idx,
                "seed": int(args.worker_seed),
                "success_rate": success_count / max(1, episode_count),
                "success_count": success_count,
                "episode_count": episode_count,
                "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
                "mean_episode_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "mean_completion_pct": round(mean_completion, 1),
                "mean_arrival_time_s": round(float(np.mean(episode_arrival_times)), 3) if episode_arrival_times else float("nan"),
                "mean_path_length_m": round(float(np.mean(episode_path_lengths)), 3) if episode_path_lengths else float("nan"),
                "mean_speed_mps": round(float(np.mean(episode_speeds)), 3) if episode_speeds else float("nan"),
                "death_reason": death_reason,
                "result": "ok",
                "duration_s": round(time.time() - trial_start, 3),
            }

        # ---- run all trials for this density in a single IsaacLab session ----
        all_rows = []
        for trial_idx in range(1, int(args.trials) + 1):
            try:
                row = _run_one_trial(trial_idx)
                all_rows.append(row)
                trial_result_path = Path(str(result_path).replace(f"_trial_{args.worker_trial}", f"_trial_{trial_idx}"))
                write_json(trial_result_path, row)
                print(json.dumps(row, indent=2))
            except Exception as exc:
                trial_tb = traceback.format_exc()
                row = {
                    "obstacles_per_tile": int(args.worker_density),
                    "tree_spacing_m": float(args.worker_density),
                    "target_speed_mps": float(args.worker_speed),
                    "tree_count": int(tree_count),
                    "trial": trial_idx,
                    "seed": int(args.worker_seed),
                    "success_rate": 0.0,
                    "success_count": 0,
                    "episode_count": 0,
                    "mean_return": 0.0,
                    "mean_episode_len": 0.0,
                    "mean_completion_pct": 0.0,
                    "mean_arrival_time_s": float("nan"),
                    "mean_path_length_m": float("nan"),
                    "mean_speed_mps": float("nan"),
                    "death_reason": "",
                    "result": f"error:{type(exc).__name__}:{exc}",
                    "duration_s": 0.0,
                }
                all_rows.append(row)
                trial_result_path = Path(str(result_path).replace(f"_trial_{args.worker_trial}", f"_trial_{trial_idx}"))
                write_json(trial_result_path, row)
                write_json(live_state_path, {**row, "phase": "error", "obstacles": preview_obstacles})
                print(f"[worker] trial {trial_idx} failed: {exc}")
                print(trial_tb)

        # Each trial is written to its own worker_density_*_trial_N.json file above.
        # Do not overwrite trial_1 with the last trial; the controller reads these
        # per-trial files back after this worker exits.
        state = read_json(live_state_path, {})
        last_row = all_rows[-1] if all_rows else {}
        state.update({"phase": "done", "result": "success" if last_row.get("success_rate", 0) > 0 else "no_success",
                       "success_rate": last_row.get("success_rate", 0)})
        state.setdefault("obstacles", preview_obstacles)
        write_json(live_state_path, state)
        print(f"[worker] density={args.worker_density} completed {len(all_rows)}/{args.trials} trials")
        return 0
    except Exception as exc:
        worker_tb = traceback.format_exc()
        row = {
            "obstacles_per_tile": int(args.worker_density),
            "tree_spacing_m": float(args.worker_density),
            "target_speed_mps": float(args.worker_speed),
            "tree_count": int(tree_count),
            "trial": int(args.worker_trial),
            "seed": int(args.worker_seed),
            "success_rate": 0.0,
            "success_count": 0,
            "episode_count": 0,
            "mean_return": 0.0,
            "mean_episode_len": 0.0,
            "mean_completion_pct": 0.0,
            "mean_arrival_time_s": float("nan"),
            "mean_path_length_m": float("nan"),
            "mean_speed_mps": float("nan"),
            "death_reason": "",
            "result": f"error:{type(exc).__name__}:{exc}",
            "duration_s": round(time.time() - start_time, 3),
        }
        write_json(result_path, row)
        write_json(live_state_path, {**row, "phase": "error", "obstacles": preview_obstacles})
        print(worker_tb)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def controller(args, hydra_overrides):
    run_label = infer_run_label(args, hydra_overrides)
    output_dir = make_run_output_dir(args.output_dir, args.run_subdir, run_label=run_label)
    live_state = output_dir / "live_state.json"
    summary_path = output_dir / "density_sweep_summary.json"
    densities = make_int_range(args.obstacles_per_tile_min, args.obstacles_per_tile_max, args.obstacles_per_tile_step)
    speeds = make_speed_values(args)
    total_trials = len(densities) * len(speeds) * int(args.trials)
    rows = []
    write_results(output_dir, rows)
    write_json(live_state, {"phase": "idle", "trajectory": [], "lidar_points": [], "obstacles": []})

    server = None
    use_web_view = str(args.view_mode).lower() == "web"
    if use_web_view and not args.no_web:
        server = start_web_server(args.host, args.port, live_state, summary_path)
        print(f"[realtree sweep] web: http://127.0.0.1:{args.port}")
    elif str(args.view_mode).lower() == "isaacsim":
        if not os.environ.get("DISPLAY", "").strip():
            print("[realtree sweep] warning: DISPLAY is not set; IsaacSim viewport mode needs a display server.")
        print("[realtree sweep] view mode: IsaacSim viewport window")

    print(
        f"[realtree sweep] plan: tree_spacing={densities[0]}..{densities[-1]}m "
        f"step={args.obstacles_per_tile_step}m speeds={speeds}m/s trials={args.trials} total={total_trials} "
        f"seed_mode=density base_seed={args.seed}"
    )
    if run_label:
        print(f"[realtree sweep] run label: {run_label}")
    print(f"[realtree sweep] tree obj: {Path(args.tree_ply).expanduser().resolve()}")
    print(f"[realtree sweep] output: {output_dir}")
    if args.dry_run:
        dry_obstacles = make_preview_obstacles(
            densities[0],
            int(args.seed),
            map_size=float(args.tree_map_size),
            clear_radius=float(args.tree_clear_radius),
        ) if densities else []
        write_json(
            live_state,
            {
                "phase": "dry_run",
                "tree_spacings_m": densities,
                "target_speeds_mps": speeds,
                "tree_count": len(dry_obstacles),
                "trajectory": [],
                "lidar_points": [],
                "obstacles": dry_obstacles,
            },
        )
        if server is not None:
            server.shutdown()
        return 0

    completed = 0
    try:
        for density_index, density in enumerate(densities):
            seed = int(args.seed) + density_index
            for speed_index, speed in enumerate(speeds):
                speed_tag = _speed_tag(speed)
                # Launch ONE worker per density/speed – it runs all trials internally without restarting IsaacLab
                result_path = output_dir / f"worker_density_{density}_speed_{speed_tag}_trial_1.json"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--worker-density",
                    str(density),
                    "--worker-speed",
                    str(speed),
                    "--worker-trial",
                    str(1),
                    "--worker-seed",
                    str(seed),
                    "--trials",
                    str(args.trials),
                    "--eval-num-envs",
                    str(args.eval_num_envs),
                    "--num-episodes",
                    str(args.num_episodes),
                    "--web-update-interval",
                    str(args.web_update_interval),
                    "--view-mode",
                    str(args.view_mode),
                    "--obstacle-height-mode",
                    args.obstacle_height_mode,
                    "--tree-ply",
                    str(args.tree_ply),
                    "--tree-map-size",
                    str(args.tree_map_size),
                    "--tree-points-per-instance",
                    str(args.tree_points_per_instance),
                    "--tree-surfel-size",
                    str(args.tree_surfel_size),
                    "--tree-scale-min",
                    str(args.tree_scale_min),
                    "--tree-scale-max",
                    str(args.tree_scale_max),
                    "--tree-tilt-deg",
                    str(args.tree_tilt_deg),
                    "--tree-clear-radius",
                    str(args.tree_clear_radius),
                    "--tree-max-faces-per-tree",
                    str(args.tree_max_faces_per_tree),
                    "--exploration-type",
                    str(args.exploration_type),
                    "--vlim-train-min",
                    str(args.vlim_train_min),
                    "--vlim-train-max",
                    str(args.vlim_train_max),
                    "--policy-task",
                    str(args.policy_task),
                    "--live-state",
                    str(live_state),
                    "--worker-result",
                    str(result_path),
                ]
                if args.max_steps is not None:
                    cmd += ["--max-steps", str(args.max_steps)]
                if not args.tree_auto_upright:
                    cmd += ["--no-tree-auto-upright"]
                if not args.observe_vlim:
                    cmd += ["--no-observe-vlim"]
                if args.vlim_randomize_eval:
                    cmd += ["--vlim-randomize-eval"]
                if args.checkpoint_path:
                    cmd += ["--checkpoint-path", args.checkpoint_path]
                cmd += hydra_overrides
                print(
                    f"[realtree sweep] spacing={density}m ({density_index+1}/{len(densities)}) "
                    f"speed={speed}m/s ({speed_index+1}/{len(speeds)}) seed={seed} trials={args.trials}"
                )
                proc = subprocess.run(cmd, cwd=str(OMNIDRONES_DIR))

                # Read back all trial results written by the worker
                for trial in range(1, int(args.trials) + 1):
                    completed += 1
                    trial_result_path = output_dir / f"worker_density_{density}_speed_{speed_tag}_trial_{trial}.json"
                    row = read_json(
                        trial_result_path,
                        {
                            "obstacles_per_tile": density,
                            "tree_spacing_m": float(density),
                            "target_speed_mps": float(speed),
                            "tree_count": len(make_preview_obstacles(
                                density,
                                seed,
                                map_size=float(args.tree_map_size),
                                clear_radius=float(args.tree_clear_radius),
                            )),
                            "trial": trial,
                            "seed": seed,
                            "success_rate": 0.0,
                            "success_count": 0,
                            "episode_count": 0,
                            "mean_return": 0.0,
                            "mean_episode_len": 0.0,
                            "mean_completion_pct": 0.0,
                            "mean_arrival_time_s": float("nan"),
                            "mean_path_length_m": float("nan"),
                            "mean_speed_mps": float("nan"),
                            "death_reason": "",
                            "result": f"worker_exit_{proc.returncode}",
                            "duration_s": 0.0,
                        },
                    )
                    if proc.returncode != 0 and str(row.get("result", "ok")) == "ok":
                        row["result"] = f"worker_exit_{proc.returncode}"
                    rows.append(row)
                    csv_path = write_results(output_dir, rows)
                    print(
                        f"[realtree sweep] {completed}/{total_trials}: "
                        f"spacing={density}m speed={speed}m/s trial={trial}/{args.trials} "
                        f"success={float(row['success_rate']) * 100.0:.1f}% "
                        f"arrival={_finite_float(row.get('mean_arrival_time_s')):.2f}s "
                        f"path={_finite_float(row.get('mean_path_length_m')):.2f}m "
                        f"avg_speed={_finite_float(row.get('mean_speed_mps')):.2f}m/s "
                        f"result={row['result']}"
                    )
                if proc.returncode != 0 and bool(args.stop_on_error):
                    write_json(
                        live_state,
                        {
                            "phase": "stopped_on_error",
                            "obstacles_per_tile": density,
                            "tree_spacing_m": float(density),
                            "target_speed_mps": float(speed),
                            "tree_count": len(make_preview_obstacles(
                                density,
                                seed,
                                map_size=float(args.tree_map_size),
                                clear_radius=float(args.tree_clear_radius),
                            )),
                            "trial": 1,
                            "trials": int(args.trials),
                            "seed": seed,
                            "result": f"worker_exit_{proc.returncode}",
                            "trajectory": [],
                            "lidar_points": [],
                            "obstacles": make_preview_obstacles(
                                density,
                                seed,
                                map_size=float(args.tree_map_size),
                                clear_radius=float(args.tree_clear_radius),
                            ),
                        },
                    )
                    print("[realtree sweep] stopped because worker failed; use --keep-going to continue after errors")
                    return proc.returncode
    finally:
        if server is not None:
            server.shutdown()
    print(f"[realtree sweep] done: {output_dir}")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate an existing OmniDrones camera-gated camera+LiDAR policy in a YOPO tree_mesh.obj forest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tree-spacing-min", "--obstacles-per-tile-min", dest="obstacles_per_tile_min", type=int, default=4,
                        help="Compatibility name: minimum tree spacing in meters")
    parser.add_argument("--tree-spacing-max", "--obstacles-per-tile-max", dest="obstacles_per_tile_max", type=int, default=4,
                        help="Compatibility name: maximum tree spacing in meters")
    parser.add_argument("--tree-spacing-step", "--obstacles-per-tile-step", dest="obstacles_per_tile_step", type=int, default=1,
                        help="Compatibility name: tree spacing step in meters")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-num-envs", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum eval loop steps. Defaults to cfg.env.max_episode_length from the selected task.",
    )
    parser.add_argument("--checkpoint-path", default=DEFAULT_VLIM_CHECKPOINT)
    parser.add_argument(
        "--policy-task",
        default=DEFAULT_POLICY_TASK,
        help="Hydra task config used by the checkpoint; train_canlidargate_trees.py uses forest_lc_gate.",
    )
    parser.add_argument("--speed", type=float, default=3.0, help="Fixed eval vlim / target speed value in m/s")
    parser.add_argument("--speed-min", type=float, default=None, help="Minimum eval vlim for a speed sweep")
    parser.add_argument("--speed-max", type=float, default=None, help="Maximum eval vlim for a speed sweep")
    parser.add_argument("--speed-step", type=float, default=1.0, help="Eval vlim sweep step")
    parser.add_argument("--vlim-train-min", type=float, default=1.0,
                        help="Training-time vlim min used to normalize observed vlim")
    parser.add_argument("--vlim-train-max", type=float, default=9.0,
                        help="Training-time vlim max used to normalize observed vlim")
    parser.add_argument("--no-observe-vlim", dest="observe_vlim", action="store_false",
                        help="Disable vlim in observation; keep enabled for latest vlim checkpoints")
    parser.add_argument("--vlim-randomize-eval", action="store_true",
                        help="Randomize vlim during evaluation resets instead of using --speed")
    parser.add_argument(
        "--exploration-type",
        choices=["random", "mode"],
        default="random",
        help="Policy action selection. random matches the stochastic PPO train metric used by best-return checkpoints; mode is deterministic.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-run-subdir",
        dest="run_subdir",
        action="store_false",
        help="Write directly into --output-dir without creating a per-run subdirectory",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument(
        "--view-mode",
        choices=["web", "isaacsim"],
        default="web",
        help="web keeps the current browser monitor; isaacsim opens the real IsaacSim viewport window",
    )
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", dest="stop_on_error", action="store_false",
                        help="Continue the sweep after a worker error")
    parser.add_argument("--web-update-interval", type=int, default=10)
    parser.add_argument("--obstacle-height-mode", default="fixed", choices=["choice", "fixed"],
                        help="Ignored in realtree mode; kept for CLI compatibility")
    parser.add_argument("--tree-ply", default=str(DEFAULT_TREE_OBJ),
                        help="Path to tree OBJ mesh file (default: YOPO tree_mesh.obj)")
    parser.add_argument("--tree-map-size", type=float, default=40.0)
    parser.add_argument("--tree-points-per-instance", type=int, default=320)
    parser.add_argument("--tree-surfel-size", type=float, default=0.08)
    parser.add_argument("--tree-scale-min", type=float, default=0.4)
    parser.add_argument("--tree-scale-max", type=float, default=0.6)
    parser.add_argument("--tree-tilt-deg", type=float, default=10.0)
    parser.add_argument("--tree-clear-radius", type=float, default=2.0)
    parser.add_argument("--tree-max-faces-per-tree", type=int, default=20000)
    parser.add_argument("--no-tree-auto-upright", dest="tree_auto_upright", action="store_false")
    parser.set_defaults(stop_on_error=True, run_subdir=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-density", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--worker-speed", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-trial", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--live-state", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    parser.set_defaults(observe_vlim=True, tree_auto_upright=True)
    args, hydra_overrides = parser.parse_known_args(argv)
    return args, hydra_overrides


def main(argv=None):
    args, hydra_overrides = parse_args(sys.argv[1:] if argv is None else argv)
    if args.worker:
        if not args.live_state or not args.worker_result:
            raise ValueError("--worker requires --live-state and --worker-result")
        return run_worker(args, hydra_overrides)
    return controller(args, hydra_overrides)


if __name__ == "__main__":
    raise SystemExit(main())
