#!/usr/bin/env python3
import argparse
import contextlib
import csv
import json
import math
import os
import re
import shutil
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
DEFAULT_VLIM_CHECKPOINT = "goodpt/6-6-vlim-lcgat-tree_best_return_4574.55.pt"
DEFAULT_POLICY_TASK = "forest_lc_gate"
CAMERA_RISK_FEATURE_NAMES = [
    "coverage",
    "diff",
    "closer",
    "blind",
    "p10_close",
    "mean_close",
    "ttc",
    "depth_trend",
    "risk_trend",
]


def _parse_camera_risk_layout(raw_value):
    text = str(raw_value or "").strip().lower()
    if not text:
        return None
    if "x" in text:
        parts = text.split("x", 1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(
                f"Invalid camera risk layout {raw_value!r}. Use forms like '3' or '3x3'."
            )
        rows = int(parts[0])
        cols = int(parts[1])
        if rows <= 0 or cols <= 0:
            raise ValueError(
                f"Invalid camera risk layout {raw_value!r}. Row/col counts must be positive."
            )
        return {"rows": rows, "cols": cols, "legacy_bins": None, "label": f"{rows}x{cols}"}
    if text.isdigit():
        bins = int(text)
        if bins <= 0:
            raise ValueError(
                f"Invalid camera risk layout {raw_value!r}. Sector count must be positive."
            )
        return {"rows": 1, "cols": bins, "legacy_bins": bins, "label": text}
    raise ValueError(
        f"Invalid camera risk layout {raw_value!r}. Use forms like '3' or '3x3'."
    )


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
        camera_yaw_center_rad=0.0,
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
        self.camera_yaw_center_rad = float(camera_yaw_center_rad)
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
        rel_yaw_grid = torch.remainder(
            yaw_grid - self.camera_yaw_center_rad + math.pi,
            2.0 * math.pi,
        ) - math.pi
        pitch_grid = row_pitches.unsqueeze(1).expand(-1, self.ku_w)

        fov_mask = (
            (rel_yaw_grid >= -h_fov / 2.0) & (rel_yaw_grid <= h_fov / 2.0)
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
                    & (rel_yaw_grid >= y_left) & (rel_yaw_grid <= y_right)
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
        spatial_gate_flat = spatial_gate.reshape(b, 1, self.ku_h * self.ku_w)
        for i in range(total_sectors):
            gate_i = self.gate_heads[i](sector_features[:, i, :])
            sector_mask = self._sector_masks[i].reshape(-1).to(device=x_ku_raw.device, dtype=torch.bool)
            if bool(sector_mask.any()):
                spatial_gate_flat[:, :, sector_mask] = gate_i.view(b, 1, 1)
        spatial_gate = spatial_gate_flat.reshape(b, 1, self.ku_h, self.ku_w)

        lidar_feat = self.ku_encoder((x_ku_raw * spatial_gate) / self.ku_value_max)
        lidar_z = self.ku_global_head(lidar_feat)

        state_2d = state.reshape(b, self.state_dim)
        state_z = self.state_encoder(state_2d)

        fused = torch.cat([state_z, lidar_z], dim=-1)
        out = self.fusion_mlp(fused)
        return out.reshape(*batch_shape, -1)

    @torch.no_grad()
    def camera_gate_debug(self, camera_risk):
        """Return the spatial gate map and per-sector gate values used on LiDAR-KU."""
        if camera_risk is None:
            return None, []
        device = next(self.parameters()).device
        camera_risk_2d = torch.as_tensor(camera_risk, dtype=torch.float32, device=device).reshape(1, -1)
        if camera_risk_2d.shape[-1] < self.camera_risk_dim:
            pad = torch.zeros(1, self.camera_risk_dim - camera_risk_2d.shape[-1], device=device)
            camera_risk_2d = torch.cat([camera_risk_2d, pad], dim=-1)
        camera_risk_2d = camera_risk_2d[..., :self.camera_risk_dim]
        camera_risk_2d = torch.nan_to_num(camera_risk_2d, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        total_sectors = self.num_rows * self.num_cols
        sector_feat_dim = total_sectors * self.features_per_sector
        sector_features = camera_risk_2d[:, :sector_feat_dim].reshape(
            1, total_sectors, self.features_per_sector
        )

        spatial_gate = torch.ones(1, 1, self.ku_h, self.ku_w, device=device, dtype=torch.float32)
        spatial_gate_flat = spatial_gate.reshape(1, 1, self.ku_h * self.ku_w)
        gate_values = []
        for i in range(total_sectors):
            gate_i = self.gate_heads[i](sector_features[:, i, :]).reshape(1)
            gate_values.append(float(gate_i.detach().cpu().item()))
            sector_mask = self._sector_masks[i].reshape(-1).to(device=device, dtype=torch.bool)
            if bool(sector_mask.any()):
                spatial_gate_flat[:, :, sector_mask] = gate_i.view(1, 1, 1)
        return spatial_gate_flat.reshape(self.ku_h, self.ku_w).detach(), gate_values


def _camera_h_fov_rad_from_cfg(cfg):
    focal = float(cfg.task.get("depth_camera_focal_length", 12.0))
    h_aperture = float(cfg.task.get("depth_camera_horizontal_aperture", 20.955))
    if focal <= 0.0 or h_aperture <= 0.0:
        raise ValueError(
            f"Invalid depth camera intrinsics for FoV: focal={focal}, horizontal_aperture={h_aperture}"
        )
    return 2.0 * math.atan(h_aperture / (2.0 * focal))


def _checkpoint_state_dict(checkpoint_path, device="cpu"):
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "policy_state_dict", "policy"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
            for _k, _v in ckpt.items():
                if isinstance(_v, dict) and all(
                    isinstance(vv, torch.Tensor) for vv in _v.values()
                ):
                    state_dict = _v
                    break
    return state_dict


def _probe_checkpoint_gate_spec(checkpoint_path, device="cpu"):
    """Probe LC-gate checkpoint metadata needed before environment creation."""
    state_dict = _checkpoint_state_dict(checkpoint_path, device=device)

    gate_by_stream = {}
    feature_dims = []
    state_dim = None
    for k, v in state_dict.items():
        if not hasattr(v, "shape"):
            continue
        if ".state_encoder.0.weight" in k and len(v.shape) == 2 and state_dim is None:
            state_dim = int(v.shape[1])
        match = re.search(r"^(?P<prefix>.*)\.gate_heads\.(?P<idx>\d+)\.0\.weight$", k)
        if match is None or len(v.shape) != 2:
            continue
        stream = "actor" if str(k).startswith("actor.") else "critic" if str(k).startswith("critic.") else match.group("prefix")
        gate_by_stream.setdefault(stream, set()).add(int(match.group("idx")))
        feature_dims.append(int(v.shape[1]))

    if not feature_dims or not gate_by_stream:
        print("[probe] no gate_heads found in checkpoint — using config values")
        return {"features_per_sector": None, "total_sectors": None, "state_dim": state_dim}

    feature_dim = feature_dims[0]
    if any(dim != feature_dim for dim in feature_dims):
        unique = sorted(set(feature_dims))
        raise RuntimeError(f"Checkpoint has inconsistent gate head input dims: {unique}")

    preferred_stream = "actor" if "actor" in gate_by_stream else next(iter(gate_by_stream))
    gate_indices = gate_by_stream[preferred_stream]
    total_sectors = max(gate_indices) + 1
    if len(gate_indices) != total_sectors:
        print(
            f"[probe] warning: non-contiguous gate head indices for {preferred_stream}: "
            f"count={len(gate_indices)}, max_index={max(gate_indices)}"
        )
        total_sectors = len(gate_indices)

    print(
        f"[probe] detected checkpoint gate spec: sectors={total_sectors}, "
        f"features_per_sector={feature_dim}, state_dim={state_dim}"
    )
    return {
        "features_per_sector": int(feature_dim),
        "total_sectors": int(total_sectors),
        "state_dim": state_dim,
    }


def _layout_from_total_sectors(total_sectors, current_rows, current_cols):
    total_sectors = int(total_sectors)
    current_rows = int(current_rows)
    current_cols = int(current_cols)
    if total_sectors <= 0:
        return None
    if current_rows > 0 and current_cols > 0 and current_rows * current_cols == total_sectors:
        return current_rows, current_cols
    side = int(round(math.sqrt(total_sectors)))
    if side * side == total_sectors:
        return side, side
    if current_rows > 0 and total_sectors % current_rows == 0:
        return current_rows, total_sectors // current_rows
    if current_cols > 0 and total_sectors % current_cols == 0:
        return total_sectors // current_cols, current_cols
    return 1, total_sectors


def _apply_checkpoint_gate_spec_to_cfg(cfg, checkpoint_path, device="cpu"):
    spec = _probe_checkpoint_gate_spec(checkpoint_path, device=device)
    fps = spec.get("features_per_sector")
    if fps is not None:
        config_fps = int(cfg.task.get("camera_risk_features_per_bin", -1))
        if config_fps != int(fps):
            print(
                f"[realtree sweep] overriding camera_risk_features_per_bin before env creation: "
                f"config={config_fps} -> checkpoint={int(fps)}"
            )
            cfg.task.camera_risk_features_per_bin = int(fps)

    total_sectors = spec.get("total_sectors")
    if total_sectors is not None:
        current_rows = int(cfg.task.get("camera_risk_num_rows", 0))
        current_cols = int(cfg.task.get("camera_risk_num_cols", 0))
        layout = _layout_from_total_sectors(total_sectors, current_rows, current_cols)
        if layout is not None:
            rows, cols = layout
            if rows != current_rows or cols != current_cols:
                print(
                    f"[realtree sweep] overriding camera risk layout before env creation: "
                    f"config={current_rows}x{current_cols} -> checkpoint={rows}x{cols}"
                )
                cfg.task.camera_risk_num_rows = int(rows)
                cfg.task.camera_risk_num_cols = int(cols)
                cfg.task.camera_risk_num_bins = int(cols) if int(rows) == 1 else int(rows * cols)
    return spec


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
    cam_pos_cfg = np.asarray(cfg.task.get("depth_camera_pos", [0.22, 0.0, 0.18]), dtype=np.float64)
    cam_target_cfg = np.asarray(cfg.task.get("depth_camera_target", [2.0, 0.0, 0.95]), dtype=np.float64)
    cam_axis = cam_target_cfg - cam_pos_cfg
    cam_xy_norm = float(np.hypot(cam_axis[0], cam_axis[1]))
    if float(np.linalg.norm(cam_axis)) <= 1e-9:
        raise RuntimeError("depth_camera_pos and depth_camera_target must not coincide.")
    camera_yaw_center_rad = math.atan2(float(cam_axis[1]), float(cam_axis[0]))
    cam_pitch_rad = math.atan2(float(cam_axis[2]), cam_xy_norm)
    cam_pitch_min = cam_pitch_rad - camera_v_fov_rad / 2.0
    cam_pitch_max = cam_pitch_rad + camera_v_fov_rad / 2.0
    fov_pitch_min = max(cam_pitch_min, lidar_pitch_min)
    fov_pitch_max = min(cam_pitch_max, lidar_pitch_max)
    if fov_pitch_max <= fov_pitch_min:
        raise RuntimeError(
            "Camera vertical FoV does not overlap LiDAR pitch range: "
            f"camera=[{math.degrees(cam_pitch_min):.2f}, {math.degrees(cam_pitch_max):.2f}]deg, "
            f"lidar=[{math.degrees(lidar_pitch_min):.2f}, {math.degrees(lidar_pitch_max):.2f}]deg."
        )
    print(
        "[lc-gate] masks aligned to camera axis | "
        f"yaw={math.degrees(camera_yaw_center_rad):.2f}deg "
        f"pitch={math.degrees(cam_pitch_rad):.2f}deg "
        f"overlap=[{math.degrees(fov_pitch_min):.2f}, {math.degrees(fov_pitch_max):.2f}]deg"
    )

    actor_backbone = CanLiDARGateBackbone(
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        camera_risk_dim=camera_risk_dim,
        ku_value_max=ku_value_max,
        output_dim=expected_feature_dim,
        camera_h_fov_rad=camera_h_fov_rad,
        fov_pitch_range=(fov_pitch_min, fov_pitch_max),
        camera_yaw_center_rad=camera_yaw_center_rad,
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
        camera_yaw_center_rad=camera_yaw_center_rad,
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


def write_compact_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), separators=(",", ":")), encoding="utf-8")
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


def make_realtree_tree_instances(
    map_size=60.0,
    spacing=4.0,
    seed=0,
    scale_min=0.5,
    scale_max=1.0,
    tilt_deg=10.0,
    clear_radius=2.0,
):
    """Return per-tree transforms using the same RNG sequence as mesh instancing."""
    positions = tree_positions_jittered_grid(
        map_size=float(map_size),
        spacing=float(spacing),
        seed=int(seed),
        clear_radius=float(clear_radius),
    )
    rng = np.random.default_rng(int(seed))
    scale_min = float(scale_min)
    scale_max = float(scale_max)
    if scale_max < scale_min:
        scale_min, scale_max = scale_max, scale_min
    max_tilt_rad = math.radians(float(tilt_deg))

    instances = []
    for px, py in positions:
        instances.append(
            {
                "position": [round(float(px), 6), round(float(py), 6), 0.0],
                "scale": round(float(rng.uniform(scale_min, scale_max)), 6),
                "roll": round(float(rng.uniform(-max_tilt_rad, max_tilt_rad)), 8),
                "pitch": round(float(rng.uniform(-max_tilt_rad, max_tilt_rad)), 8),
                "yaw": round(float(rng.uniform(-math.pi, math.pi)), 8),
            }
        )
    return instances


def make_replay_tree_mesh(tree_obj_path, max_faces=2500, auto_upright=True):
    """Return a compact, remapped source tree mesh for browser replay."""
    vertices, faces = read_obj_mesh(tree_obj_path, max_faces=int(max_faces))
    vertices = normalize_tree_mesh(vertices, auto_upright=bool(auto_upright))
    used = np.unique(faces.reshape(-1))
    old_to_new = -np.ones(vertices.shape[0], dtype=np.int32)
    old_to_new[used] = np.arange(used.shape[0], dtype=np.int32)
    vertices = vertices[used]
    faces = old_to_new[faces]
    return {
        "vertices": [[round(float(x), 5), round(float(y), 5), round(float(z), 5)] for x, y, z in vertices],
        "faces": [[int(a), int(b), int(c)] for a, b, c in faces],
        "max_faces": int(max_faces),
    }


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
    "death_contact", "death_oob",
    "death_flip", "death_nan",
]
_DEATH_REASON_LABELS = [
    "z_low", "z_high", "overspeed",
    "contact", "out_of_bounds",
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
        "success_videos_saved",
        "success_video_manifest",
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

        # Group by obstacle density for per-density coloring
        density_key = "tree_spacing_m"
        density_values = sorted(set(_finite_float(x.get(density_key)) for x in summary if np.isfinite(_finite_float(x.get(density_key)))))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(density_values))]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        all_xs = []
        for di, dens in enumerate(density_values):
            group = [x for x in summary if abs(_finite_float(x.get(density_key)) - dens) < 1e-6]
            group.sort(key=lambda x: x["target_speed_mps"])
            xs = [x["target_speed_mps"] for x in group]
            ys = [x["success_rate"] * 100.0 for x in group]
            label = f"sp={dens:g}m"
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[di], label=label)
            all_xs.extend(xs)
        if len(density_values) > 1:
            ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("Target speed (m/s)")
        ax.set_ylabel("Success rate (%)")
        ax.set_title("OmniDrones Cam+LiDAR Policy Real-Tree Sweep")
        ax.set_ylim(-2, 102)
        if all_xs:
            ax.set_xlim(min(all_xs) - 0.5, max(all_xs) + 0.5)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_dir / "success_rate.png")
        plt.close(fig)

        # ---- completion percentage chart ----
        # Group by obstacle density for per-density coloring
        density_key2 = "tree_spacing_m"
        density_values2 = sorted(set(_finite_float(x.get(density_key2)) for x in summary if np.isfinite(_finite_float(x.get(density_key2)))))
        cmap2 = plt.get_cmap("tab10")
        colors2 = [cmap2(i % 10) for i in range(len(density_values2))]
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        all_xs2 = []
        for di, dens in enumerate(density_values2):
            group = [x for x in summary if abs(_finite_float(x.get(density_key2)) - dens) < 1e-6]
            group.sort(key=lambda x: x["target_speed_mps"])
            xs2 = [x["target_speed_mps"] for x in group]
            ys2 = [x["avg_completion_pct"] for x in group]
            label = f"sp={dens:g}m"
            ax2.plot(xs2, ys2, marker="o", linewidth=2.0, color=colors2[di], label=label)
            all_xs2.extend(xs2)
        if len(density_values2) > 1:
            ax2.legend(fontsize=8, loc="best")
        ax2.set_xlabel("Target speed (m/s)")
        ax2.set_ylabel("Avg completion (%)")
        ax2.set_title("OmniDrones Cam+LiDAR Policy Real-Tree Sweep - Completion %")
        ax2.set_ylim(-2, 102)
        if all_xs2:
            ax2.set_xlim(min(all_xs2) - 0.5, max(all_xs2) + 0.5)
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
            # Group by obstacle density for per-density coloring
            density_key_m = "tree_spacing_m"
            density_values_m = sorted(set(_finite_float(x.get(density_key_m)) for x in metric_summary if np.isfinite(_finite_float(x.get(density_key_m)))))
            cmap_m = plt.get_cmap("tab10")
            colors_m = [cmap_m(i % 10) for i in range(len(density_values_m))]
            fig_m, ax_m = plt.subplots(figsize=(7.2, 4.2), dpi=140)
            all_xs_m = []
            for di, dens in enumerate(density_values_m):
                group = [x for x in metric_summary if abs(_finite_float(x.get(density_key_m)) - dens) < 1e-6]
                group.sort(key=lambda x: x["target_speed_mps"])
                xs_m = [x["target_speed_mps"] for x in group]
                ys_m = [x[key] for x in group]
                label = f"sp={dens:g}m"
                ax_m.plot(xs_m, ys_m, marker="o", linewidth=2.0, color=colors_m[di], label=label)
                all_xs_m.extend(xs_m)
            if len(density_values_m) > 1:
                ax_m.legend(fontsize=8, loc="best")
            ax_m.set_xlabel("Target speed (m/s)")
            ax_m.set_ylabel(ylabel)
            ax_m.set_title(f"OmniDrones Cam+LiDAR Policy Real-Tree Sweep - {title_suffix}")
            if all_xs_m:
                ax_m.set_xlim(min(all_xs_m) - 0.5, max(all_xs_m) + 0.5)
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


def _suppress_noisy_isaac_warnings():
    """Reduce known Isaac Sim warning spam that obscures sweep/video logs."""
    try:
        import omni.log  # type: ignore

        log = omni.log.get_log()
        for channel in (
            "isaacsim.core.simulation_manager.plugin",
        ):
            log.set_channel_level(channel, omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE)
    except Exception:
        pass


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
    .diag { margin-top: 14px; border-top: 1px solid #26313c; padding-top: 12px; }
    .diag h2 { font-size: 14px; margin: 0 0 8px; color: #dbe7ef; }
    .heat { display: grid; grid-template-columns: 1fr; gap: 7px; }
    .heat canvas { width: 100%; height: 82px; image-rendering: pixelated; border: 1px solid #2b3540; border-radius: 4px; background: #091017; }
    .heat label { display: block; color: #95a3b3; font-size: 11px; margin: 0 0 3px; }
    .riskgrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 8px 0 10px; }
    .riskcell { border: 1px solid #2b3540; border-radius: 4px; padding: 5px; background: #111820; font-size: 10px; line-height: 1.25; }
    .riskcell b { color: #e8edf2; font-size: 11px; }
    .riskcell span { color: #9fb1c2; display: inline-block; min-width: 64px; }
    .diag-note { color: #95a3b3; font-size: 11px; margin-top: 6px; }
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
    <div class="diag">
      <h2>CameraRisk / LiDAR-KU</h2>
      <div id="riskgrid" class="riskgrid"></div>
      <div class="heat">
        <div><label>LiDAR-KU original</label><canvas id="kuRaw" width="80" height="40"></canvas></div>
        <div><label>LiDAR-KU after camera gate</label><canvas id="kuFused" width="80" height="40"></canvas></div>
        <div><label>Spatial gate</label><canvas id="kuGate" width="80" height="40"></canvas></div>
      </div>
      <div id="diagNote" class="diag-note">waiting for sensor data</div>
    </div>
    <table><thead><tr><th>spacing</th><th>target</th><th>trees</th><th>success</th><th>actual</th></tr></thead><tbody id="summary"></tbody></table>
  </aside>
</div>
<script>
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const metrics = document.getElementById("metrics");
const summaryEl = document.getElementById("summary");
const riskgrid = document.getElementById("riskgrid");
const diagNote = document.getElementById("diagNote");
function resize(){ canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; }
addEventListener("resize", resize); resize();
function worldToCanvas(p){
  const w = canvas.width, h = canvas.height;
  const sx = w / 44, sy = h / 86, s = Math.min(sx, sy);
  return [w/2 + p[0]*s, h/2 - p[1]*s];
}
function drawGrid(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.strokeStyle = "#22303a";
  for(let x=-20; x<=20; x+=4){ const a=worldToCanvas([x,-40]), b=worldToCanvas([x,40]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }
  for(let y=-40; y<=40; y+=4){ const a=worldToCanvas([-20,y]), b=worldToCanvas([20,y]); ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); }
}
function dot(p, r, color){ const q=worldToCanvas(p); ctx.fillStyle=color; ctx.beginPath(); ctx.arc(q[0], q[1], r*devicePixelRatio, 0, Math.PI*2); ctx.fill(); }
function line(points, color, width){
  if(!points || points.length < 2) return;
  ctx.strokeStyle=color; ctx.lineWidth=width*devicePixelRatio; ctx.beginPath();
  points.forEach((p,i)=>{ const q=worldToCanvas(p); if(i===0) ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]); });
  ctx.stroke();
}
function heatColor(v, mode){
  const t = Math.max(0, Math.min(1, v / 255));
  if(mode === "gate"){
    const c = Math.round(30 + 225 * t);
    return [Math.round(40 + 80*t), c, Math.round(110 + 110*t)];
  }
  const r = Math.round(255 * Math.max(0, 1.4 - 2.2*t));
  const g = Math.round(255 * Math.max(0, 1.3 - Math.abs(t - 0.35) * 2.2));
  const b = Math.round(255 * Math.min(1, 0.25 + 1.2*t));
  return [r, g, b];
}
function drawHeat(id, values, w, h, mode){
  const c = document.getElementById(id);
  const x = c.getContext("2d");
  w = Number(w) || 80; h = Number(h) || 40;
  if(c.width !== w) c.width = w;
  if(c.height !== h) c.height = h;
  const img = x.createImageData(w, h);
  const arr = Array.isArray(values) ? values : [];
  for(let i=0; i<w*h; i++){
    const rgb = heatColor(Number(arr[i] || 0), mode);
    img.data[i*4] = rgb[0]; img.data[i*4+1] = rgb[1]; img.data[i*4+2] = rgb[2]; img.data[i*4+3] = 255;
  }
  x.putImageData(img, 0, 0);
}
function renderSensorDebug(sensor){
  if(!sensor || !sensor.lidar_ku){
    riskgrid.innerHTML = "";
    diagNote.textContent = "waiting for sensor data";
    return;
  }
  const ku = sensor.lidar_ku;
  drawHeat("kuRaw", ku.original_u8, ku.width, ku.height, "ku");
  drawHeat("kuFused", ku.fused_u8, ku.width, ku.height, "ku");
  drawHeat("kuGate", ku.gate_u8, ku.width, ku.height, "gate");
  const cr = sensor.camera_risk || {};
  const sectors = cr.sectors || [];
  const featureNames = cr.feature_names || [];
  riskgrid.style.gridTemplateColumns = `repeat(${Math.max(1, Number(cr.cols) || 3)}, 1fr)`;
  riskgrid.innerHTML = sectors.map(s=>{
    const fs = (s.features || []).map((v,i)=>{
      const name = featureNames[i] || `f${i+1}`;
      return `<span>${name}</span> ${Number(v).toFixed(2)}`;
    }).join("<br>");
    const gate = s.gate == null ? "-" : Number(s.gate).toFixed(2);
    return `<div class="riskcell"><b>r${s.row} c${s.col}</b><br>gate ${gate}<br>${fs}</div>`;
  }).join("");
  const stale = cr.stale_ratio == null ? "-" : Number(cr.stale_ratio).toFixed(3);
  diagNote.textContent = `KU range ${ku.original_min}..${ku.original_max} → ${ku.fused_min}..${ku.fused_max}, stale ${stale}`;
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
  renderSensorDebug(state.sensor_debug);
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


def _resize_rgb_frame(frame, width=0, height=0):
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.size > 0 and np.nanmax(arr) <= 1.0:
            arr = (np.nan_to_num(arr, nan=0.0) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.nan_to_num(arr, nan=0.0).clip(0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 and height <= 0:
        return arr
    try:
        import cv2  # type: ignore

        src_h, src_w = arr.shape[:2]
        if width <= 0:
            width = max(1, int(round(src_w * (height / max(1, src_h)))))
        if height <= 0:
            height = max(1, int(round(src_h * (width / max(1, src_w)))))
        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        return arr


def _as_env_vec3(value, env_idx=0, agent_idx=0):
    return _as_env_vec(value, env_idx=env_idx, agent_idx=agent_idx, length=3)


def _as_env_vec4(value, env_idx=0, agent_idx=0):
    return _as_env_vec(value, env_idx=env_idx, agent_idx=agent_idx, length=4)


def _as_env_vec(value, env_idx=0, agent_idx=0, length=3):
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            tensor = torch.as_tensor(value)
        if tensor.ndim >= 3:
            env_idx = max(0, min(int(env_idx), int(tensor.shape[0]) - 1))
            agent_idx = max(0, min(int(agent_idx), int(tensor.shape[1]) - 1))
            tensor = tensor[env_idx, agent_idx]
        elif tensor.ndim == 2:
            env_idx = max(0, min(int(env_idx), int(tensor.shape[0]) - 1))
            tensor = tensor[env_idx]
        tensor = tensor.reshape(-1)[:length].float()
        if tensor.numel() < length or not torch.isfinite(tensor).all():
            return None
        return tensor
    except Exception:
        return None


def _set_viewport_camera(eye, target, up=None):
    if up is not None:
        try:
            import omni.kit.commands  # type: ignore
            from omni.kit.viewport.utility import get_active_viewport  # type: ignore
            from pxr import Gf, Sdf, Usd, UsdGeom

            viewport_api = get_active_viewport()
            if viewport_api is None:
                raise RuntimeError("No active viewport")

            camera_prim_path = "/OmniverseKit_Persp"
            prim = viewport_api.stage.GetPrimAtPath(camera_prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Viewport camera prim not found: {camera_prim_path}")

            camera = UsdGeom.Camera(prim)
            time_code = Usd.TimeCode.Default()
            world_xform = camera.ComputeLocalToWorldTransform(time_code)
            parent_xform = camera.ComputeParentToWorldTransform(time_code)
            iparent_xform = parent_xform.GetInverse()
            old_local_xform = world_xform * iparent_xform

            eye_gf = Gf.Vec3d(*[float(v) for v in np.asarray(eye).reshape(3)])
            target_gf = Gf.Vec3d(*[float(v) for v in np.asarray(target).reshape(3)])
            up_np = np.asarray(up, dtype=np.float64).reshape(3)
            up_norm = float(np.linalg.norm(up_np))
            if up_norm <= 1e-9:
                up_np = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                up_np = up_np / up_norm
            up_gf = Gf.Vec3d(*[float(v) for v in up_np])

            pos_in_parent = iparent_xform.Transform(eye_gf)
            target_in_parent = iparent_xform.Transform(target_gf)
            try:
                up_in_parent = iparent_xform.TransformDir(up_gf)
            except Exception:
                up_in_parent = up_gf
            new_local_xform = Gf.Matrix4d(1).SetLookAt(pos_in_parent, target_in_parent, up_in_parent).GetInverse()

            coi_attr = prim.GetProperty("omni:kit:centerOfInterest")
            if not coi_attr or not coi_attr.IsValid():
                coi_attr = prim.CreateAttribute(
                    "omni:kit:centerOfInterest",
                    Sdf.ValueTypeNames.Vector3d,
                    True,
                    Sdf.VariabilityUniform,
                )
            new_local_coi = (new_local_xform * parent_xform).GetInverse().Transform(target_gf)
            omni.kit.commands.create(
                "ChangePropertyCommand",
                prop_path=coi_attr.GetPath(),
                value=new_local_coi,
                prev=coi_attr.Get(time_code),
                timecode=time_code,
                usd_context_name=viewport_api.usd_context_name,
                type_to_create_if_not_exist=Sdf.ValueTypeNames.Vector3d,
            ).do()
            omni.kit.commands.create(
                "TransformPrimCommand",
                path=camera_prim_path,
                new_transform_matrix=new_local_xform,
                old_transform_matrix=old_local_xform,
                time_code=time_code,
                usd_context_name=viewport_api.usd_context_name,
            ).do()
            return True
        except Exception:
            pass

    for module_name in ("isaacsim.core.utils.viewports", "omni.isaac.core.utils.viewports"):
        try:
            import importlib

            module = importlib.import_module(module_name)
            module.set_camera_view(eye=eye, target=target)
            return True
        except Exception:
            continue
    return False


def _set_recorded_drone_render_visibility(base_env, env_idx=0, visible=True):
    """Temporarily show/hide one env's drone meshes for follow-camera RGB capture."""
    try:
        import omni.usd  # type: ignore
        from pxr import Usd, UsdGeom
    except Exception:
        return 0

    try:
        stage = omni.usd.get_context().get_stage()
    except Exception:
        stage = None
    if stage is None:
        return 0

    drone = getattr(base_env, "drone", None)
    drone_name = str(getattr(drone, "name", "") or "")
    if not drone_name:
        return 0
    env_idx = max(0, int(env_idx))
    env_prefix = f"/World/envs/env_{env_idx}/"
    root_suffix = f"/{drone_name}_0"
    depth_prim_name = str(getattr(base_env, "depth_prim_name", "DepthCamera"))

    def iter_prims(include_instance_proxies=False):
        if not include_instance_proxies:
            yield from stage.Traverse()
            return
        try:
            yield from Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        except Exception:
            yield from stage.Traverse()

    camera_paths = []
    for prim in iter_prims(False):
        path = str(prim.GetPath())
        if not path.startswith(env_prefix) or root_suffix not in path:
            continue
        try:
            is_camera = prim.IsA(UsdGeom.Camera)
        except Exception:
            is_camera = False
        if f"/{depth_prim_name}" in path or is_camera:
            camera_paths.append(path)

    def is_depth_camera_related(path):
        for camera_path in camera_paths:
            if path == camera_path:
                return True
            if path.startswith(camera_path + "/"):
                return True
            if camera_path.startswith(path + "/"):
                return True
        return False

    def find_writable_imageable_target(prim):
        try:
            if not prim.IsInstanceProxy():
                return prim
        except Exception:
            return prim

        parent = prim.GetParent()
        while parent and parent.IsValid():
            parent_path = str(parent.GetPath())
            if not parent_path.startswith(env_prefix) or root_suffix not in parent_path:
                return None
            if is_depth_camera_related(parent_path):
                return None
            try:
                if not parent.IsInstanceProxy() and parent.IsA(UsdGeom.Imageable):
                    return parent
            except Exception:
                return None
            parent = parent.GetParent()
        return None

    changed_paths = set()
    for prim in iter_prims(True):
        path = str(prim.GetPath())
        if not path.startswith(env_prefix) or root_suffix not in path:
            continue
        if is_depth_camera_related(path):
            continue
        try:
            if not prim.IsA(UsdGeom.Imageable):
                continue
        except Exception:
            continue
        target = find_writable_imageable_target(prim)
        try:
            if target is None or not target.IsValid() or not target.IsA(UsdGeom.Imageable):
                continue
            target_path = str(target.GetPath())
            if is_depth_camera_related(target_path):
                continue
            imageable = UsdGeom.Imageable(target)
            if visible:
                imageable.MakeVisible()
                imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            else:
                imageable.MakeInvisible()
                imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            changed_paths.add(target_path)
        except Exception:
            continue
    return len(changed_paths)


def _body_forward_axis_from_cfg(base_env):
    try:
        raw = base_env.cfg.task.get("body_forward_axis", [1.0, 0.0, 0.0])
    except Exception:
        raw = [1.0, 0.0, 0.0]
    vals = [float(v) for v in raw]
    if len(vals) != 3:
        vals = [1.0, 0.0, 0.0]
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 1e-9:
        vals = [1.0, 0.0, 0.0]
        norm = 1.0
    return [v / norm for v in vals]


def _set_third_person_follow_camera(base_env, env_idx=0, eye_offset=None, lookat_offset=None):
    eye_offset = torch.as_tensor(
        eye_offset if eye_offset is not None else [4.0, 0.0, 1.2],
        dtype=torch.float32,
        device=getattr(base_env, "device", "cpu"),
    )
    lookat_offset = torch.as_tensor(
        lookat_offset if lookat_offset is not None else [1.0, 0.0, 0.15],
        dtype=torch.float32,
        device=getattr(base_env, "device", "cpu"),
    )

    try:
        pos_value = None
        quat_value = None
        if hasattr(base_env.drone, "get_world_poses"):
            pos_value, quat_value = base_env.drone.get_world_poses(clone=True)
        if pos_value is None and hasattr(base_env.drone, "pos"):
            pos_value = base_env.drone.pos
        if quat_value is None and hasattr(base_env.drone, "rot"):
            quat_value = base_env.drone.rot
        drone_pos = _as_env_vec3(pos_value, env_idx=env_idx)
        if drone_pos is None:
            return False
        eye_offset = eye_offset.to(drone_pos.device)
        lookat_offset = lookat_offset.to(drone_pos.device)

        quat = _as_env_vec4(quat_value, env_idx=env_idx)
        if quat is not None:
            from omni_drones.utils.torch import quat_rotate

            quat = quat.to(drone_pos.device)
            quat_b = quat.reshape(1, 4)

            def rotate_body_axis(axis):
                axis_b = torch.tensor(axis, dtype=torch.float32, device=drone_pos.device).reshape(1, 3)
                return quat_rotate(quat_b, axis_b).reshape(3)

            up_axis = rotate_body_axis([0.0, 0.0, 1.0])
            forward = rotate_body_axis(_body_forward_axis_from_cfg(base_env))
            right = torch.cross(forward, up_axis, dim=0)
        else:
            forward = torch.tensor(
                _body_forward_axis_from_cfg(base_env),
                dtype=torch.float32,
                device=drone_pos.device,
            )
            world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=drone_pos.device)
            up_axis = world_up
            right = torch.cross(forward, up_axis, dim=0)

        if float(torch.linalg.norm(forward).detach().cpu().item()) < 1e-5:
            forward = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=drone_pos.device)
        else:
            forward = forward / torch.linalg.norm(forward).clamp_min(1e-6)
        if float(torch.linalg.norm(right).detach().cpu().item()) < 1e-5:
            right = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=drone_pos.device)
        else:
            right = right / torch.linalg.norm(right).clamp_min(1e-6)
        if float(torch.linalg.norm(up_axis).detach().cpu().item()) < 1e-5:
            up_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=drone_pos.device)
        else:
            up_axis = up_axis / torch.linalg.norm(up_axis).clamp_min(1e-6)

        eye = drone_pos - forward * eye_offset[0] + right * eye_offset[1] + up_axis * eye_offset[2]
        target = drone_pos + forward * lookat_offset[0] + right * lookat_offset[1] + up_axis * lookat_offset[2]
        return _set_viewport_camera(
            eye.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            up=up_axis.detach().cpu().numpy(),
        )
    except Exception:
        return False


def _third_person_follow_rgb_frame(
    base_env,
    env_idx=0,
    eye_offset=None,
    lookat_offset=None,
    width=0,
    height=0,
    show_drone=True,
):
    camera_set = _set_third_person_follow_camera(
        base_env,
        env_idx=env_idx,
        eye_offset=eye_offset,
        lookat_offset=lookat_offset,
    )
    visibility_changed = 0
    drone_meshes_are_hidden = bool(getattr(base_env, "hide_drone_meshes_from_depth_camera", True))
    if show_drone and drone_meshes_are_hidden:
        visibility_changed = _set_recorded_drone_render_visibility(base_env, env_idx=env_idx, visible=True)
    try:
        if camera_set and hasattr(base_env, "sim"):
            with contextlib.suppress(Exception):
                base_env.sim.render()
        rgb = base_env.render(mode="rgb_array")
        return _resize_rgb_frame(rgb, width, height)
    finally:
        if visibility_changed:
            _set_recorded_drone_render_visibility(base_env, env_idx=env_idx, visible=False)


def _depth_frame_uint8(base_env, env_idx=0, out_width=0, out_height=0, depth_max=0.0):
    try:
        import torch
        import torch.nn.functional as F

        depth = getattr(base_env, "depth_obs_cache", None)
        if depth is None:
            return None
        depth = depth.detach().float()
        env_idx = max(0, min(int(env_idx), int(depth.shape[0]) - 1))
        depth_h = int(getattr(base_env, "depth_h", 0) or 0)
        depth_w = int(getattr(base_env, "depth_w", 0) or 0)
        if depth_h <= 0 or depth_w <= 0:
            return None
        depth_env = depth[env_idx].reshape(1, 1, depth_h, depth_w)
        out_width = int(out_width or 0)
        out_height = int(out_height or 0)
        if out_width > 0 or out_height > 0:
            if out_width <= 0:
                out_width = max(1, int(round(depth_w * (out_height / max(1, depth_h)))))
            if out_height <= 0:
                out_height = max(1, int(round(depth_h * (out_width / max(1, depth_w)))))
            depth_env = F.interpolate(depth_env, size=(out_height, out_width), mode="nearest")
        depth_env = depth_env[0, 0]
        if depth_max <= 0.0:
            depth_max = float(getattr(base_env, "depth_max_range", 0.0) or 0.0)
        if depth_max <= 0.0:
            depth_max = float(torch.nan_to_num(depth_env, nan=0.0, posinf=0.0).max().item())
        depth_max = max(float(depth_max), 1e-6)
        preview = (1.0 - torch.nan_to_num(depth_env, nan=depth_max, posinf=depth_max, neginf=0.0) / depth_max)
        preview = preview.clamp(0.0, 1.0)
        gray = (preview * 255.0).to(torch.uint8).cpu().numpy()
        return np.repeat(gray[..., None], 3, axis=-1)
    except Exception:
        return None


def _quat_wxyz_to_euler_deg(quat):
    w, x, y, z = [float(v) for v in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [round(math.degrees(v), 5) for v in (roll, pitch, yaw)]


def _as_env_matrix(value, num_envs, width):
    if value is None:
        return None
    try:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.reshape(int(num_envs), -1, int(width))
        tensor = tensor[:, 0, :]
        if not torch.isfinite(tensor).all():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor
    except Exception:
        return None


def _quantize_u8(values, value_max):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    value_max = max(float(value_max), 1e-6)
    tensor = torch.nan_to_num(tensor, nan=value_max, posinf=value_max, neginf=0.0)
    tensor = tensor.clamp(0.0, value_max)
    return torch.round(tensor / value_max * 255.0).to(torch.uint8).detach().cpu().reshape(-1).tolist()


def _round_list(values, digits=4):
    return [round(float(v), int(digits)) for v in values]


@torch.no_grad()
def _capture_sensor_debug(base_env, actor_backbone=None, env_idx=0, num_envs=None):
    """Capture camera-risk values plus original/gated LiDAR-KU for web diagnostics."""
    try:
        if num_envs is None:
            num_envs = int(getattr(base_env, "num_envs", 1))
        env_idx = max(0, min(int(env_idx), int(num_envs) - 1))

        lidar_cache = getattr(base_env, "encoded_lidar_cache", None)
        if lidar_cache is None:
            return None
        lidar = lidar_cache.detach().float().reshape(int(num_envs), -1)[env_idx]
        ku_h = int(getattr(actor_backbone, "ku_h", getattr(base_env, "num_pitch_bins", 40)))
        ku_w = int(getattr(actor_backbone, "ku_w", getattr(base_env, "num_yaw_bins", 80)))
        ku_dim = ku_h * ku_w
        if lidar.numel() < ku_dim:
            return None
        lidar = lidar[:ku_dim]
        ku_value_max = float(getattr(actor_backbone, "ku_value_max", base_env.cfg.task.get("ku_value_max", 20.0)))

        camera_risk_cache = getattr(base_env, "camera_risk_cache", None)
        camera_risk = None
        if camera_risk_cache is not None:
            camera_risk = camera_risk_cache.detach().float().reshape(int(num_envs), -1)[env_idx]

        gate = torch.ones(ku_h, ku_w, dtype=torch.float32, device=lidar.device)
        sector_gates = []
        if actor_backbone is not None and camera_risk is not None:
            gate_debug, sector_gates = actor_backbone.camera_gate_debug(camera_risk)
            if gate_debug is not None:
                gate = gate_debug.to(device=lidar.device, dtype=torch.float32)
        fused = (lidar.reshape(ku_h, ku_w) * gate).reshape(-1)

        camera_risk_values = []
        sectors = []
        stale_ratio = None
        feature_names = []
        if camera_risk is not None:
            camera_risk_values = _round_list(camera_risk.detach().cpu().reshape(-1).tolist(), 4)
            rows = int(getattr(actor_backbone, "num_rows", int(base_env.cfg.task.get("camera_risk_num_rows", 1))))
            cols = int(getattr(actor_backbone, "num_cols", int(base_env.cfg.task.get("camera_risk_num_cols", 3))))
            features_per_sector = int(getattr(actor_backbone, "features_per_sector", int(base_env.cfg.task.get("camera_risk_features_per_bin", 4))))
            feature_names = [
                CAMERA_RISK_FEATURE_NAMES[i] if i < len(CAMERA_RISK_FEATURE_NAMES) else f"f{i + 1}"
                for i in range(max(0, features_per_sector))
            ]
            total_sectors = rows * cols
            flat = camera_risk.detach().cpu().reshape(-1)
            for si in range(total_sectors):
                start = si * features_per_sector
                stop = start + features_per_sector
                features = _round_list(flat[start:stop].tolist(), 4) if stop <= flat.numel() else []
                sectors.append({
                    "index": int(si),
                    "row": int(si // max(1, cols)),
                    "col": int(si % max(1, cols)),
                    "features": features,
                    "gate": round(float(sector_gates[si]), 4) if si < len(sector_gates) else None,
                })
            if flat.numel() > total_sectors * features_per_sector:
                stale_ratio = round(float(flat[total_sectors * features_per_sector].item()), 4)
        else:
            rows = int(getattr(actor_backbone, "num_rows", 1))
            cols = int(getattr(actor_backbone, "num_cols", 3))
            features_per_sector = int(getattr(actor_backbone, "features_per_sector", 4))
            feature_names = [
                CAMERA_RISK_FEATURE_NAMES[i] if i < len(CAMERA_RISK_FEATURE_NAMES) else f"f{i + 1}"
                for i in range(max(0, features_per_sector))
            ]

        return {
            "camera_risk": {
                "rows": int(rows),
                "cols": int(cols),
                "features_per_sector": int(features_per_sector),
                "feature_names": feature_names,
                "values": camera_risk_values,
                "sectors": sectors,
                "stale_ratio": stale_ratio,
            },
            "lidar_ku": {
                "width": int(ku_w),
                "height": int(ku_h),
                "value_max": float(ku_value_max),
                "original_u8": _quantize_u8(lidar, ku_value_max),
                "fused_u8": _quantize_u8(fused, ku_value_max),
                "gate_u8": _quantize_u8(gate.reshape(-1), 1.0),
                "original_min": round(float(torch.nan_to_num(lidar).min().item()), 4),
                "original_max": round(float(torch.nan_to_num(lidar).max().item()), 4),
                "fused_min": round(float(torch.nan_to_num(fused).min().item()), 4),
                "fused_max": round(float(torch.nan_to_num(fused).max().item()), 4),
            },
        }
    except Exception:
        return None


def _capture_replay_pose(base_env, env_idx, num_envs, step, sim_time_s, actor_backbone=None, include_sensors=True):
    pos = _as_env_matrix(getattr(base_env.drone, "pos", None), num_envs, 3)
    quat = _as_env_matrix(getattr(base_env.drone, "rot", None), num_envs, 4)
    if pos is None or quat is None:
        try:
            world_pos, world_quat = base_env.drone.get_world_poses(clone=True)
            if pos is None:
                pos = _as_env_matrix(world_pos, num_envs, 3)
            if quat is None:
                quat = _as_env_matrix(world_quat, num_envs, 4)
        except Exception:
            pass
    if pos is None:
        return None
    env_idx = max(0, min(int(env_idx), int(num_envs) - 1))
    pos_list = [round(float(v), 6) for v in pos[env_idx].detach().cpu().reshape(-1)[:3].tolist()]
    if quat is not None:
        quat_raw = quat[env_idx].detach().cpu().reshape(-1)[:4].tolist()
        norm = math.sqrt(sum(float(v) * float(v) for v in quat_raw))
        if norm > 1e-9:
            quat_list = [round(float(v) / norm, 8) for v in quat_raw]
        else:
            quat_list = [1.0, 0.0, 0.0, 0.0]
    else:
        quat_list = [1.0, 0.0, 0.0, 0.0]

    vel = _as_env_matrix(getattr(base_env.drone, "vel_w", None), num_envs, 6)
    speed = 0.0
    vel_list = [0.0, 0.0, 0.0]
    if vel is not None:
        vel_list = [round(float(v), 6) for v in vel[env_idx].detach().cpu().reshape(-1)[:3].tolist()]
        speed = math.sqrt(sum(float(v) * float(v) for v in vel_list))

    sample = {
        "step": int(step),
        "t": round(float(sim_time_s), 6),
        "pos": pos_list,
        "quat_wxyz": quat_list,
        "euler_deg": _quat_wxyz_to_euler_deg(quat_list),
        "vel": vel_list,
        "speed": round(float(speed), 6),
    }
    if include_sensors:
        sensor_debug = _capture_sensor_debug(base_env, actor_backbone, env_idx=env_idx, num_envs=num_envs)
        if sensor_debug is not None:
            sample["sensor_debug"] = sensor_debug
    return sample


def _write_video_file(path, frames, fps):
    if not frames:
        raise ValueError(f"No frames to write for {path}")
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        [np.ascontiguousarray(f) for f in frames],
        fps=max(1, int(fps)),
        macro_block_size=None,
        quality=9,
    )
    return str(path)


def _ensure_replay_static_assets(out_dir):
    static_dir = Path(out_dir) / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "three.module.js": REPO_ROOT / "EGO-PLANNER" / "tools" / "static" / "three.module.js",
        "OrbitControls.js": REPO_ROOT / "EGO-PLANNER" / "tools" / "static" / "OrbitControls.js",
    }
    for name, src in sources.items():
        dst = static_dir / name
        if src.exists() and (not dst.exists() or src.stat().st_size != dst.stat().st_size):
            shutil.copy2(src, dst)


def _write_replay_html(path, replay_json_name):
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone Success Replay</title>
  <style>
    html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#101315;color:#eef3f4;font-family:system-ui,-apple-system,Segoe UI,sans-serif}}
    canvas{{position:fixed;inset:0}}
    .panel{{position:fixed;left:16px;top:16px;width:min(430px,calc(100vw - 32px));background:rgba(17,22,24,.9);border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:14px;backdrop-filter:blur(10px)}}
    h1{{font-size:15px;margin:0 0 10px 0}}
    .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}}
    .metric{{border:1px solid rgba(255,255,255,.14);border-radius:6px;padding:8px;background:rgba(255,255,255,.05);min-width:0}}
    .label{{font-size:11px;color:#a9b4b7;margin-bottom:3px}} .value{{font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .controls{{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;margin-bottom:10px}}
    button,select{{height:32px;border-radius:6px;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.08);color:#eef3f4;padding:0 10px;font:inherit;font-size:13px}}
    input{{width:100%;accent-color:#55d6d2}} .row{{display:flex;gap:8px;flex-wrap:wrap}}
    .note{{font-size:12px;color:#a9b4b7;line-height:1.45;border-top:1px solid rgba(255,255,255,.14);margin-top:10px;padding-top:10px}}
    .right{{position:fixed;right:16px;bottom:16px;width:min(330px,calc(100vw - 32px));background:rgba(17,22,24,.9);border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:12px;font-size:12px;line-height:1.55}}
    .diag{{position:fixed;right:16px;top:16px;width:min(520px,calc(100vw - 32px));max-height:min(52vh,520px);overflow:auto;background:rgba(17,22,24,.92);border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:12px;font-size:12px;backdrop-filter:blur(10px)}}
    .diag h2{{font-size:13px;margin:0 0 8px;color:#eef3f4}} .diag-note{{color:#a9b4b7;margin-top:7px;line-height:1.35}}
    .riskgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px}} .riskcell{{border:1px solid rgba(255,255,255,.14);border-radius:6px;background:rgba(255,255,255,.05);padding:6px;line-height:1.25;font-size:10px}} .riskcell b{{font-size:11px;color:#eef3f4}} .riskcell span{{color:#9fb1c2;display:inline-block;min-width:64px}}
    .heat{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .heat label{{display:block;color:#a9b4b7;font-size:11px;margin-bottom:4px}} .heat canvas{{position:static;width:100%;height:92px;image-rendering:pixelated;border:1px solid rgba(255,255,255,.14);border-radius:6px;background:#091017}}
    code{{color:#d9e3e5}} @media(max-width:900px){{.right{{display:none}}.diag{{left:16px;right:16px;width:auto;max-height:42vh}}.grid{{grid-template-columns:repeat(2,1fr)}}.heat{{grid-template-columns:1fr}}}}
  </style>
  <script type="importmap">{{"imports":{{"three":"./static/three.module.js"}}}}</script>
</head>
<body>
<canvas id="scene"></canvas>
<section class="panel">
  <h1>Drone Success Replay <span id="status" style="float:right;color:#77d47f;font-size:12px">loading</span></h1>
  <div class="grid">
    <div class="metric"><div class="label">Frame</div><div class="value" id="frame">0/0</div></div>
    <div class="metric"><div class="label">Speed</div><div class="value" id="speed">0 m/s</div></div>
    <div class="metric"><div class="label">Euler</div><div class="value" id="euler">0 0 0</div></div>
    <div class="metric"><div class="label">Trees</div><div class="value" id="trees">0</div></div>
  </div>
  <div class="controls"><button id="play">Pause</button><input id="timeline" type="range" min="0" max="1" value="0" step="1"><select id="rate"><option>.25x</option><option>.5x</option><option selected>1x</option><option>2x</option><option>4x</option></select></div>
  <div class="row"><button id="free">Free</button><button id="follow">Follow</button><button id="top">Top</button><button id="side">Side</button></div>
  <div class="note" id="note"></div>
</section>
<aside class="right">Mouse: left rotate, wheel zoom, right pan.<br><br>Tree mesh is rendered from the same normalized OBJ geometry and per-tree RNG transforms used by the test script. Drone attitude uses recorded quaternion samples.</aside>
<section class="diag">
  <h2>CameraRisk / LiDAR-KU</h2>
  <div id="riskgrid" class="riskgrid"></div>
  <div class="heat">
    <div><label>LiDAR-KU original</label><canvas id="kuRaw" width="80" height="40"></canvas></div>
    <div><label>LiDAR-KU after camera gate</label><canvas id="kuFused" width="80" height="40"></canvas></div>
    <div><label>Spatial gate</label><canvas id="kuGate" width="80" height="40"></canvas></div>
  </div>
  <div id="diagNote" class="diag-note">waiting for sensor data</div>
</section>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from './static/OrbitControls.js';
const canvas=document.getElementById('scene');
const renderer=new THREE.WebGLRenderer({{canvas,antialias:true}}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.shadowMap.enabled=true;
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x101315); scene.fog=new THREE.Fog(0x101315,80,170);
const camera=new THREE.PerspectiveCamera(55,1,.05,300); camera.up.set(0,0,1); camera.position.set(24,-48,28);
const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true; controls.target.set(0,0,2);
scene.add(new THREE.HemisphereLight(0xd8eef8,0x25301e,1.7)); const sun=new THREE.DirectionalLight(0xffffff,2.1); sun.position.set(-25,-35,60); sun.castShadow=true; scene.add(sun);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(100,100),new THREE.MeshStandardMaterial({{color:0x1c241f,roughness:.95}})); ground.receiveShadow=true; scene.add(ground);
const grid=new THREE.GridHelper(100,20,0x596966,0x303936); grid.rotation.x=Math.PI/2; grid.material.transparent=true; grid.material.opacity=.35; scene.add(grid);
const ui={{status:by('status'),frame:by('frame'),speed:by('speed'),euler:by('euler'),trees:by('trees'),timeline:by('timeline'),play:by('play'),rate:by('rate'),note:by('note'),riskgrid:by('riskgrid'),diagNote:by('diagNote')}};
let replay, frames=[], idx=0, playing=true, view='follow', last=performance.now(), drone;
function by(id){{return document.getElementById(id)}}
function heatColor(v,mode){{const t=Math.max(0,Math.min(1,v/255)); if(mode==='gate'){{const c=Math.round(30+225*t); return [Math.round(40+80*t),c,Math.round(110+110*t)]}} const r=Math.round(255*Math.max(0,1.4-2.2*t)); const g=Math.round(255*Math.max(0,1.3-Math.abs(t-.35)*2.2)); const b=Math.round(255*Math.min(1,.25+1.2*t)); return [r,g,b]}}
function drawHeat(id,values,w,h,mode){{const c=by(id); const x=c.getContext('2d'); w=Number(w)||80; h=Number(h)||40; if(c.width!==w)c.width=w; if(c.height!==h)c.height=h; const img=x.createImageData(w,h); const arr=Array.isArray(values)?values:[]; for(let i=0;i<w*h;i++){{const rgb=heatColor(Number(arr[i]||0),mode); img.data[i*4]=rgb[0]; img.data[i*4+1]=rgb[1]; img.data[i*4+2]=rgb[2]; img.data[i*4+3]=255}} x.putImageData(img,0,0)}}
function renderSensorDebug(sensor){{if(!sensor||!sensor.lidar_ku){{ui.riskgrid.innerHTML=''; ui.diagNote.textContent='waiting for sensor data'; return}} const ku=sensor.lidar_ku; drawHeat('kuRaw',ku.original_u8,ku.width,ku.height,'ku'); drawHeat('kuFused',ku.fused_u8,ku.width,ku.height,'ku'); drawHeat('kuGate',ku.gate_u8,ku.width,ku.height,'gate'); const cr=sensor.camera_risk||{{}}; const sectors=cr.sectors||[]; const featureNames=cr.feature_names||[]; ui.riskgrid.style.gridTemplateColumns=`repeat(${{Math.max(1,Number(cr.cols)||3)}},1fr)`; ui.riskgrid.innerHTML=sectors.map(s=>{{const fs=(s.features||[]).map((v,i)=>{{const name=featureNames[i]||`f${{i+1}}`; return `<span>${{name}}</span> ${{Number(v).toFixed(2)}}`}}).join('<br>'); const gate=s.gate==null?'-':Number(s.gate).toFixed(2); return `<div class="riskcell"><b>r${{s.row}} c${{s.col}}</b><br>gate ${{gate}}<br>${{fs}}</div>`}}).join(''); const stale=cr.stale_ratio==null?'-':Number(cr.stale_ratio).toFixed(3); ui.diagNote.textContent=`KU range ${{ku.original_min}}..${{ku.original_max}} → ${{ku.fused_min}}..${{ku.fused_max}}, stale ${{stale}}`}}
function makeTreeGeometry(mesh){{const pos=[]; for(const v of mesh.vertices) pos.push(v[0],v[1],v[2]); const ind=[]; for(const f of mesh.faces) ind.push(f[0],f[1],f[2]); const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3)); g.setIndex(ind); g.computeVertexNormals(); return g}}
function buildTrees(){{const mat=new THREE.MeshStandardMaterial({{color:0x2f8f57,roughness:.82,side:THREE.DoubleSide}}); const geo=makeTreeGeometry(replay.tree_mesh); const inst=new THREE.InstancedMesh(geo,mat,replay.tree_instances.length); inst.castShadow=true; inst.receiveShadow=true; const o=new THREE.Object3D(); replay.tree_instances.forEach((t,i)=>{{o.position.set(t.position[0],t.position[1],t.position[2]); o.rotation.set(t.roll,t.pitch,t.yaw,'XYZ'); o.scale.setScalar(t.scale); o.updateMatrix(); inst.setMatrixAt(i,o.matrix)}}); scene.add(inst); ui.trees.textContent=String(replay.tree_instances.length)}}
function buildDrone(){{const g=new THREE.Group(); const bodyMat=new THREE.MeshStandardMaterial({{color:0xf4b84a,metalness:.25,roughness:.5}}); const body=new THREE.Mesh(new THREE.BoxGeometry(.30,.18,.10),bodyMat); const nose=new THREE.Mesh(new THREE.ConeGeometry(.055,.12,16),bodyMat); nose.rotation.z=-Math.PI/2; nose.position.x=.20; const armMat=new THREE.MeshStandardMaterial({{color:0xd9e3e5,roughness:.55}}); const rotorMat=new THREE.MeshStandardMaterial({{color:0x171b1d,metalness:.35,roughness:.5}}); g.add(body,nose); const armGeo=new THREE.CylinderGeometry(.016,.016,.34,12); const armA=new THREE.Mesh(armGeo,armMat); armA.rotation.z=-Math.PI/4; const armB=new THREE.Mesh(armGeo,armMat); armB.rotation.z=Math.PI/4; g.add(armA,armB); const a=.17/Math.SQRT2; const rotorPositions=[[a,a,.02],[-a,a,.02],[-a,-a,.02],[a,-a,.02]]; g.rotors=[]; for(const p of rotorPositions){{const r=new THREE.Group(); r.position.set(...p); r.add(new THREE.Mesh(new THREE.TorusGeometry(.072,.007,8,32),rotorMat)); const b1=new THREE.Mesh(new THREE.BoxGeometry(.19,.018,.006),rotorMat); const b2=b1.clone(); b2.rotation.z=Math.PI/2; r.add(b1,b2); g.rotors.push(r); g.add(r)}} g.scale.setScalar(2.4); scene.add(g); return g}}
function buildTrajectory(){{const pts=frames.map(f=>new THREE.Vector3(...f.pos)); const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({{color:0x55d6d2}})); scene.add(line); const cur=new THREE.Mesh(new THREE.SphereGeometry(.11,12,8),new THREE.MeshBasicMaterial({{color:0x55d6d2}})); scene.add(cur); replay.cursor=cur}}
function target(){{const p=replay.target||[0,0,2]; const m=new THREE.Mesh(new THREE.SphereGeometry(.45,24,16),new THREE.MeshStandardMaterial({{color:0xff6b5a,emissive:0x260805}})); m.position.set(p[0],p[1],p[2]); scene.add(m)}}
function applyFrame(i){{idx=Math.max(0,Math.min(frames.length-1,Math.round(i))); const f=frames[idx]; drone.position.set(...f.pos); const q=f.quat_wxyz||[1,0,0,0]; drone.quaternion.set(q[1],q[2],q[3],q[0]).normalize(); for(const [ri,r] of drone.rotors.entries()) r.rotation.z += (ri%2?1:-1)*.7; replay.cursor.position.set(...f.pos); ui.timeline.value=idx; ui.frame.textContent=`${{idx+1}}/${{frames.length}}`; ui.speed.textContent=`${{(f.speed||0).toFixed(2)}} m/s`; ui.euler.textContent=(f.euler_deg||[0,0,0]).map(v=>v.toFixed(1)).join(' '); renderSensorDebug(f.sensor_debug); if(view==='follow'){{const p=new THREE.Vector3(...f.pos); const axis=replay.body_forward_axis||[1,0,0]; const fw=new THREE.Vector3(axis[0],axis[1],axis[2]).applyQuaternion(drone.quaternion).normalize(); const up=new THREE.Vector3(0,0,1).applyQuaternion(drone.quaternion).normalize(); const desired=p.clone().add(fw.clone().multiplyScalar(-6)).add(up.clone().multiplyScalar(2.0)); const target=p.clone().add(fw.clone().multiplyScalar(1.2)).add(up.clone().multiplyScalar(.15)); camera.up.lerp(up,.22).normalize(); camera.position.lerp(desired,.18); controls.target.lerp(target,.25)}}}}
function resize(){{renderer.setSize(innerWidth,innerHeight,false); camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix()}} window.addEventListener('resize',resize);
ui.play.onclick=()=>{{playing=!playing; ui.play.textContent=playing?'Pause':'Play'; if(playing&&idx>=frames.length-1)applyFrame(0)}}; ui.timeline.oninput=()=>{{playing=false; ui.play.textContent='Play'; applyFrame(Number(ui.timeline.value))}};
by('free').onclick=()=>view='free'; by('follow').onclick=()=>view='follow'; by('top').onclick=()=>{{view='free'; camera.up.set(0,0,1); const p=new THREE.Vector3(...frames[idx].pos); camera.position.set(p.x,p.y,p.z+70); controls.target.copy(p)}}; by('side').onclick=()=>{{view='free'; camera.up.set(0,0,1); const p=new THREE.Vector3(...frames[idx].pos); camera.position.set(p.x+34,p.y-52,p.z+18); controls.target.copy(p)}};
function animate(now){{requestAnimationFrame(animate); const dt=Math.min(.05,(now-last)/1000); last=now; if(playing&&frames.length){{const rate=parseFloat(ui.rate.value)||1; applyFrame(idx+dt*rate*30); if(idx>=frames.length-1){{playing=false; ui.play.textContent='Play'}}}} controls.update(); renderer.render(scene,camera)}}
fetch('./{replay_json_name}',{{cache:'no-store'}}).then(r=>r.json()).then(data=>{{replay=data; frames=data.frames||[]; ui.timeline.max=Math.max(0,frames.length-1); ui.note.innerHTML=`Drone source: <code>${{data.drone?.usd_path||''}}</code><br>Samples: ${{frames.length}}, replay interval: ${{data.replay_interval_steps}} sim step(s).`; buildTrees(); drone=buildDrone(); buildTrajectory(); target(); ui.status.textContent='ready'; applyFrame(0); resize(); requestAnimationFrame(animate)}}).catch(e=>{{console.error(e); ui.status.textContent='load failed'}});
</script>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


def _make_browser_replay_data(replay_data, max_frames=300):
    """Keep browser replay JSON small enough to parse without freezing the page."""
    frames = replay_data.get("frames", []) if isinstance(replay_data, dict) else []
    if not isinstance(frames, list) or len(frames) <= int(max_frames):
        return replay_data

    stride = max(1, math.ceil(len(frames) / float(max_frames)))
    keep = list(range(0, len(frames), stride))
    if keep[-1] != len(frames) - 1:
        keep.append(len(frames) - 1)

    slim = dict(replay_data)
    slim["frames"] = [frames[i] for i in keep]
    slim["original_frame_count"] = len(frames)
    slim["replay_frame_stride_from_original"] = stride
    slim["browser_replay_note"] = (
        f"Downsampled from {len(frames)} frames to {len(slim['frames'])} "
        "frames so the replay page can load quickly."
    )
    return slim


def _write_success_replay_pair(video_dir, stem, replay_data):
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    _ensure_replay_static_assets(video_dir)
    json_path = video_dir / f"{stem}_replay.json"
    html_path = video_dir / f"{stem}_replay.html"
    write_compact_json(json_path, _make_browser_replay_data(replay_data))
    _write_replay_html(html_path, json_path.name)
    return str(json_path), str(html_path)


def _save_success_video_pair(
    video_dir,
    *,
    density,
    speed,
    trial_idx,
    episode_idx,
    env_idx,
    success_idx,
    rgb_frames,
    depth_frames,
    fps,
    metadata,
    replay_data=None,
):
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    speed_tag = _speed_tag(speed)
    stem = (
        f"density_{int(density)}_speed_{speed_tag}_trial_{int(trial_idx):02d}_"
        f"episode_{int(episode_idx):03d}_env_{int(env_idx):03d}_success_{int(success_idx):03d}"
    )
    saved = {
        "rgb_video": "",
        "depth_video": "",
        "metadata": str(video_dir / f"{stem}.json"),
        "replay_json": "",
        "replay_html": "",
    }
    if rgb_frames:
        saved["rgb_video"] = _write_video_file(video_dir / f"{stem}_follow.mp4", rgb_frames, fps)
    if depth_frames:
        saved["depth_video"] = _write_video_file(video_dir / f"{stem}_depth.mp4", depth_frames, fps)
    if replay_data is not None:
        saved["replay_json"], saved["replay_html"] = _write_success_replay_pair(video_dir, stem, replay_data)
    write_json(
        video_dir / f"{stem}.json",
        {
            **metadata,
            "rgb_video": saved["rgb_video"],
            "depth_video": saved["depth_video"],
            "replay_json": saved["replay_json"],
            "replay_html": saved["replay_html"],
            "frame_count_rgb": len(rgb_frames),
            "frame_count_depth": len(depth_frames),
            "fps": int(fps),
        },
    )
    return saved


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

    export_success_videos = bool(getattr(args, "export_success_videos", False))
    isaacsim_view = str(getattr(args, "view_mode", "web")).lower() == "isaacsim" or export_success_videos
    overrides = [f"hydra.searchpath=[file://{OMNIDRONES_DIR / 'cfg'}]"]
    if args.policy_task and not _has_override(hydra_overrides, "task"):
        overrides.append(f"task={args.policy_task}")
    overrides += list(hydra_overrides)
    layout_override = _parse_camera_risk_layout(getattr(args, "camera_risk_layout", ""))
    if layout_override is not None:
        overrides += [
            f"++task.camera_risk_num_rows={int(layout_override['rows'])}",
            f"++task.camera_risk_num_cols={int(layout_override['cols'])}",
        ]
        if layout_override["legacy_bins"] is not None:
            overrides.append(f"++task.camera_risk_num_bins={int(layout_override['legacy_bins'])}")
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
    if export_success_videos:
        eye_offset = "[" + ",".join(str(float(v)) for v in args.video_follow_eye_offset) + "]"
        lookat_offset = "[" + ",".join(str(float(v)) for v in args.video_follow_lookat_offset) + "]"
        overrides += [
            "++task.follow_camera=true",
            f"++task.hide_drone_meshes_from_depth_camera={'true' if bool(args.video_hide_drone) else 'false'}",
            f"++task.follow_camera_env_index={int(args.video_env_index)}",
            f"++task.follow_camera_eye_offset={eye_offset}",
            f"++task.follow_camera_lookat_offset={lookat_offset}",
        ]
    if args.max_steps is not None:
        overrides.append(f"++max_steps={int(args.max_steps)}")
    if args.checkpoint_path:
        overrides.append(f"checkpoint_path={args.checkpoint_path}")

    # 评估模式：只用 IsaacSim 物理接触、OOB、timeout 判定失败；
    # 禁用训练时的 LiDAR/距离阈值 collision 判定。
    overrides += [
        "++task.collision_dist=-1.0",
        "++task.terminate_z_min=-9999",
        "++task.terminate_z_max=9999",
        "++task.terminate_v_norm=99999",
        "++task.flip_tilt_deg=180",
        "++task.flip_consecutive_steps=99999",
    ]

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
    if "env" in cfg:
        cfg.env.max_episode_length = int(args.max_steps)
    if "task" in cfg:
        with contextlib.suppress(Exception):
            cfg.task.max_episode_length = int(args.max_steps)

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
    if layout_override is not None:
        cfg.task.camera_risk_num_rows = int(layout_override["rows"])
        cfg.task.camera_risk_num_cols = int(layout_override["cols"])
        if layout_override["legacy_bins"] is not None:
            cfg.task.camera_risk_num_bins = int(layout_override["legacy_bins"])
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
    checkpoint_path = _resolve_checkpoint_path(cfg.get("checkpoint_path"))
    _apply_checkpoint_gate_spec_to_cfg(cfg, checkpoint_path, device="cpu")
    camera_risk_layout_label = (
        f"{int(cfg.task.get('camera_risk_num_rows', 1))}x"
        f"{int(cfg.task.get('camera_risk_num_cols', int(cfg.task.get('camera_risk_num_bins', 3))))}"
    )
    if export_success_videos:
        cfg.task.follow_camera = True
        cfg.task.hide_drone_meshes_from_depth_camera = bool(args.video_hide_drone)
        cfg.task.follow_camera_env_index = int(args.video_env_index)
        cfg.task.follow_camera_eye_offset = list(args.video_follow_eye_offset)
        cfg.task.follow_camera_lookat_offset = list(args.video_follow_lookat_offset)

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
            "camera_risk_layout": layout_override["label"] if layout_override is not None else camera_risk_layout_label,
            "trajectory": trajectory,
            "obstacles": preview_obstacles,
        },
    )

    _preflight_runtime_checks(cfg, int(args.eval_num_envs))
    if layout_override is not None:
        print(f"[realtree sweep] camera risk layout override: {layout_override['label']}")
    simulation_app = None
    try:
        simulation_app = init_simulation_app(cfg)
        _suppress_noisy_isaac_warnings()
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
        actor_backbone, _critic_backbone = _inject_canlidargate_backbone(policy, base_env, env, cfg)
        _load_checkpoint_strictish(policy, checkpoint_path, base_env.device)
        policy.eval()
        base_env.enable_render(isaacsim_view)
        base_env.eval()
        env.eval()

        video_dir = Path(getattr(args, "video_output_dir", "") or (Path(args.worker_result).parent / "videos"))
        video_success_target = max(0, int(getattr(args, "videos_per_combo", 0) or 0))
        video_success_saved = 0
        video_records = read_json(video_dir / "manifest.json", [])
        if not isinstance(video_records, list):
            video_records = []
        video_env_index = max(0, int(getattr(args, "video_env_index", 0) or 0))
        video_auto_env = bool(export_success_videos and not getattr(args, "video_fixed_env", False))
        video_interval = max(1, int(getattr(args, "video_interval", 1) or 1))
        video_fps = int(getattr(args, "video_fps", 0) or 0)
        if video_fps <= 0:
            video_fps = max(10, int(round(1.0 / max(float(getattr(base_env, "dt", 0.02)) * video_interval, 1e-6))))
        video_rgb_width = max(0, int(getattr(args, "video_width", 0) or 0))
        video_rgb_height = max(0, int(getattr(args, "video_height", 0) or 0))
        video_depth_width = max(0, int(getattr(args, "video_depth_width", 0) or 0))
        video_depth_height = max(0, int(getattr(args, "video_depth_height", 0) or 0))
        if video_depth_width <= 0:
            video_depth_width = video_rgb_width
        if video_depth_height <= 0:
            video_depth_height = video_rgb_height
        video_depth_max = max(0.0, float(getattr(args, "video_depth_max", 0.0) or 0.0))
        video_max_frames = max(0, int(getattr(args, "video_max_frames", 0) or 0))
        video_show_drone = bool(export_success_videos and not getattr(args, "video_hide_drone", False))
        replay_interval = max(1, int(getattr(args, "replay_interval", 1) or 1))
        replay_tree_mesh = None
        replay_tree_instances = []
        if export_success_videos:
            replay_tree_instances = make_realtree_tree_instances(
                map_size=float(args.tree_map_size),
                spacing=float(args.worker_density),
                seed=int(args.worker_seed),
                scale_min=float(args.tree_scale_min),
                scale_max=float(args.tree_scale_max),
                tilt_deg=float(args.tree_tilt_deg),
                clear_radius=float(args.tree_clear_radius),
            )
            replay_tree_mesh = make_replay_tree_mesh(
                Path(str(args.tree_ply)).expanduser().resolve(),
                max_faces=int(getattr(args, "replay_tree_max_faces", 2500) or 2500),
                auto_upright=bool(args.tree_auto_upright),
            )
        if export_success_videos:
            video_env_label = "auto(any-success)" if video_auto_env else str(video_env_index)
            print(
                "[video] success export enabled | "
                f"target={video_success_target} env={video_env_label} interval={video_interval} "
                f"fps={video_fps} replay_interval={replay_interval} show_drone={video_show_drone} dir={video_dir}"
            )
        video_capture_warned = False

        def _run_one_trial(trial_idx):
            """Run one trial (num_episodes episodes) and return the result row dict."""
            nonlocal video_success_saved, video_capture_warned
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
                    capture_this_episode = (
                        export_success_videos
                        and video_success_saved < video_success_target
                    )
                    capture_env_indices = []
                    if capture_this_episode:
                        if video_auto_env:
                            capture_env_indices = list(range(num_envs_eval))
                        elif video_env_index < num_envs_eval:
                            capture_env_indices = [video_env_index]
                        else:
                            capture_this_episode = False
                    rgb_frames_by_env = {int(e): [] for e in capture_env_indices}
                    depth_frames_by_env = {int(e): [] for e in capture_env_indices}
                    replay_frames_by_env = {int(e): [] for e in capture_env_indices}

                    for step in range(int(args.max_steps)):
                        step_count = step + 1
                        td = policy(td)
                        td = env.step(td)
                        reward = td[("next", "agents", "reward")].reshape(-1).float()
                        done = td[("next", "done")].reshape(-1).to(torch.bool)
                        stats_td = td[("next", "stats")]
                        current_pos = base_env.drone.pos.detach().clone().reshape(num_envs_eval, 3)
                        if capture_this_episode and (step % replay_interval == 0):
                            sim_time_s = float(step_count) * float(getattr(base_env, "dt", 0.02))
                            for capture_env_idx in capture_env_indices:
                                capture_env_idx = int(capture_env_idx)
                                if capture_env_idx >= num_envs_eval:
                                    continue
                                pose_sample = _capture_replay_pose(
                                    base_env,
                                    capture_env_idx,
                                    num_envs_eval,
                                    step_count,
                                    sim_time_s,
                                    actor_backbone=actor_backbone,
                                    include_sensors=True,
                                )
                                if pose_sample is not None:
                                    replay_frames_by_env.setdefault(capture_env_idx, []).append(pose_sample)

                        active = ~finished
                        if active.any():
                            path_lengths[active] += torch.norm(current_pos[active] - prev_positions[active], dim=-1)
                            prev_positions[active] = current_pos[active]
                            ep_returns[active] += reward[active]

                        success_now = torch.zeros(num_envs_eval, dtype=torch.bool, device=base_env.device)
                        first_success = torch.zeros(num_envs_eval, dtype=torch.bool, device=base_env.device)
                        if "success" in stats_td.keys():
                            success_now = (stats_td["success"].reshape(-1).float() >= 0.5).to(torch.bool)
                            first_success = torch.logical_and(torch.logical_and(success_now, active), arrival_steps < 0)
                            if first_success.any():
                                arrival_steps[first_success] = step_count
                            ep_success = torch.maximum(ep_success, first_success.to(torch.int32))

                        if capture_this_episode and (step % video_interval == 0):
                            for capture_env_idx in capture_env_indices:
                                capture_env_idx = int(capture_env_idx)
                                if capture_env_idx >= num_envs_eval:
                                    continue
                                if bool(finished[capture_env_idx].item()) and int(arrival_steps[capture_env_idx].item()) < 0:
                                    continue
                                rgb_frames = rgb_frames_by_env.setdefault(capture_env_idx, [])
                                depth_frames = depth_frames_by_env.setdefault(capture_env_idx, [])
                                if video_max_frames > 0 and len(rgb_frames) >= video_max_frames:
                                    continue
                                try:
                                    rgb = _third_person_follow_rgb_frame(
                                        base_env,
                                        env_idx=capture_env_idx,
                                        eye_offset=args.video_follow_eye_offset,
                                        lookat_offset=args.video_follow_lookat_offset,
                                        width=video_rgb_width,
                                        height=video_rgb_height,
                                        show_drone=video_show_drone,
                                    )
                                    if rgb is not None:
                                        rgb_frames.append(rgb)
                                except Exception as exc:
                                    if not video_capture_warned:
                                        print(f"[video] third-person RGB capture disabled after error: {exc}")
                                        video_capture_warned = True
                                depth_rgb = _depth_frame_uint8(
                                    base_env,
                                    env_idx=capture_env_idx,
                                    out_width=video_depth_width,
                                    out_height=video_depth_height,
                                    depth_max=video_depth_max,
                                )
                                if depth_rgb is not None:
                                    depth_frames.append(depth_rgb)

                        # Capture final positions for envs that just finished.  Treat reaching the
                        # goal as terminal for evaluation, even if the underlying env would keep
                        # simulating and later report an out-of-bounds/contact death.
                        just_finished = torch.logical_and(active, torch.logical_or(done, first_success))
                        if just_finished.any():
                            final_positions[just_finished] = current_pos[just_finished]
                            finish_steps[just_finished] = step_count
                            for e in torch.nonzero(just_finished, as_tuple=False).flatten().detach().cpu().tolist():
                                if bool(first_success[int(e)].item()) or bool(success_now[int(e)].item()):
                                    ep_death_reasons[int(e)] = "success"
                                else:
                                    try:
                                        ep_death_reasons[int(e)] = _extract_death_reason_from_stats(stats_td, int(e))
                                    except Exception:
                                        ep_death_reasons[int(e)] = "unknown"

                        finished = torch.logical_or(finished, torch.logical_or(done, first_success))

                        if (
                            capture_this_episode
                            and video_success_saved < video_success_target
                        ):
                            success_env_indices = [
                                int(e)
                                for e in torch.nonzero(first_success, as_tuple=False).flatten().detach().cpu().tolist()
                                if int(e) in rgb_frames_by_env
                            ]
                            if not video_auto_env and video_env_index not in success_env_indices:
                                success_env_indices = []
                            for success_env_idx in success_env_indices:
                                if video_success_saved >= video_success_target:
                                    break
                                try:
                                    last_replay_step = -1
                                    if replay_frames_by_env.get(success_env_idx):
                                        last_replay_step = int(replay_frames_by_env[success_env_idx][-1].get("step", -1))
                                    if last_replay_step < step_count:
                                        pose_sample = _capture_replay_pose(
                                            base_env,
                                            success_env_idx,
                                            num_envs_eval,
                                            step_count,
                                            float(step_count) * float(getattr(base_env, "dt", 0.02)),
                                            actor_backbone=actor_backbone,
                                            include_sensors=True,
                                        )
                                        if pose_sample is not None:
                                            replay_frames_by_env.setdefault(success_env_idx, []).append(pose_sample)
                                    replay_data = {
                                        "schema": "omnidrones_success_replay_v1",
                                        "record_mode": "auto_any_success" if video_auto_env else "fixed_env",
                                        "env_index": int(success_env_idx),
                                        "arrival_step": int(step_count),
                                        "sim_dt": float(getattr(base_env, "dt", 0.02)),
                                        "replay_interval_steps": int(replay_interval),
                                        "tree_spacing_m": float(args.worker_density),
                                        "target_speed_mps": float(args.worker_speed),
                                        "seed": int(args.worker_seed),
                                        "episode": int(ep + 1),
                                        "trial": int(trial_idx),
                                        "body_forward_axis": _body_forward_axis_from_cfg(base_env),
                                        "target": [
                                            round(float(v), 6)
                                            for v in target_pos[success_env_idx].detach().cpu().reshape(-1)[:3].tolist()
                                        ],
                                        "start": [
                                            round(float(v), 6)
                                            for v in init_pos[success_env_idx].detach().cpu().reshape(-1)[:3].tolist()
                                        ],
                                        "drone": {
                                            "model": str(getattr(base_env.drone, "name", "Hummingbird")),
                                            "usd_path": str(getattr(base_env.drone, "usd_path", "")),
                                            "param_path": str(getattr(base_env.drone, "param_path", "")),
                                        },
                                        "tree_config": {
                                            "obj_path": str(Path(str(args.tree_ply)).expanduser().resolve()),
                                            "map_size": float(args.tree_map_size),
                                            "spacing": float(args.worker_density),
                                            "scale_min": float(args.tree_scale_min),
                                            "scale_max": float(args.tree_scale_max),
                                            "tilt_deg": float(args.tree_tilt_deg),
                                            "clear_radius": float(args.tree_clear_radius),
                                            "auto_upright": bool(args.tree_auto_upright),
                                        },
                                        "tree_mesh": replay_tree_mesh,
                                        "tree_instances": replay_tree_instances,
                                        "sensor_debug": {
                                            "enabled": True,
                                            "ku_width": int(getattr(actor_backbone, "ku_w", 80)),
                                            "ku_height": int(getattr(actor_backbone, "ku_h", 40)),
                                            "ku_value_max": float(getattr(actor_backbone, "ku_value_max", 20.0)),
                                            "camera_risk_rows": int(getattr(actor_backbone, "num_rows", 1)),
                                            "camera_risk_cols": int(getattr(actor_backbone, "num_cols", 3)),
                                            "features_per_sector": int(getattr(actor_backbone, "features_per_sector", 4)),
                                        },
                                        "frames": replay_frames_by_env.get(success_env_idx, []),
                                    }
                                    video_success_saved += 1
                                    saved = _save_success_video_pair(
                                        video_dir,
                                        density=int(args.worker_density),
                                        speed=float(args.worker_speed),
                                        trial_idx=trial_idx,
                                        episode_idx=ep + 1,
                                        env_idx=success_env_idx,
                                        success_idx=video_success_saved,
                                        rgb_frames=rgb_frames_by_env.get(success_env_idx, []),
                                        depth_frames=depth_frames_by_env.get(success_env_idx, []),
                                        fps=video_fps,
                                        metadata={
                                            "obstacles_per_tile": int(args.worker_density),
                                            "tree_spacing_m": float(args.worker_density),
                                            "target_speed_mps": float(args.worker_speed),
                                            "trial": int(trial_idx),
                                            "episode": int(ep + 1),
                                            "env_index": int(success_env_idx),
                                            "record_mode": "auto_any_success" if video_auto_env else "fixed_env",
                                            "seed": int(args.worker_seed),
                                            "arrival_step": int(step_count),
                                            "sim_dt": float(getattr(base_env, "dt", 0.02)),
                                            "capture_interval": int(video_interval),
                                            "camera": "third_person_follow",
                                            "show_drone": bool(video_show_drone),
                                            "follow_eye_offset": [float(v) for v in args.video_follow_eye_offset],
                                            "follow_lookat_offset": [float(v) for v in args.video_follow_lookat_offset],
                                        },
                                        replay_data=replay_data,
                                    )
                                    video_records.append(saved)
                                    write_json(video_dir / "manifest.json", video_records)
                                    print(
                                        f"[video] saved success {video_success_saved}/{video_success_target} "
                                        f"env={success_env_idx}: "
                                        f"{saved.get('rgb_video') or '-'} | {saved.get('depth_video') or '-'}"
                                    )
                                except Exception as exc:
                                    print(f"[video] failed to save success video for env={success_env_idx}: {exc}")
                            if video_success_saved >= video_success_target:
                                capture_this_episode = False
                            else:
                                capture_env_indices = [
                                    int(e)
                                    for e in capture_env_indices
                                    if int(e) not in success_env_indices
                                    and not bool(finished[int(e)].item())
                                ]
                                if not capture_env_indices:
                                    capture_this_episode = False

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
                                        "sensor_debug": _capture_sensor_debug(
                                            base_env,
                                            actor_backbone=actor_backbone,
                                            env_idx=0,
                                            num_envs=num_envs_eval,
                                        ),
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
                    success_mask = arrival_steps >= 0
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
                "success_videos_saved": int(video_success_saved) if export_success_videos else 0,
                "success_video_manifest": str(video_dir / "manifest.json") if export_success_videos else "",
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
        if export_success_videos:
            state["success_videos_saved"] = int(video_success_saved)
            state["success_video_manifest"] = str(video_dir / "manifest.json")
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
    if bool(getattr(args, "export_success_videos", False)):
        if not os.environ.get("DISPLAY", "").strip():
            print("[realtree sweep] warning: video export needs viewport rendering; DISPLAY is not set.")
        video_env_label = str(int(args.video_env_index)) if bool(getattr(args, "video_fixed_env", False)) else "auto(any-success)"
        print(
            f"[realtree sweep] success video export: target={int(args.videos_per_combo)} per spacing/speed, "
            f"env={video_env_label}, interval={int(args.video_interval)}"
        )

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
                if args.export_success_videos:
                    cmd += [
                        "--export-success-videos",
                        "--videos-per-combo",
                        str(args.videos_per_combo),
                        "--video-output-dir",
                        str(output_dir / "videos"),
                        "--video-env-index",
                        str(args.video_env_index),
                        "--video-interval",
                        str(args.video_interval),
                        "--video-fps",
                        str(args.video_fps),
                        "--video-width",
                        str(args.video_width),
                        "--video-height",
                        str(args.video_height),
                        "--video-depth-width",
                        str(args.video_depth_width),
                        "--video-depth-height",
                        str(args.video_depth_height),
                        "--video-depth-max",
                        str(args.video_depth_max),
                        "--video-max-frames",
                        str(args.video_max_frames),
                        "--replay-interval",
                        str(args.replay_interval),
                        "--replay-tree-max-faces",
                        str(args.replay_tree_max_faces),
                        "--video-follow-eye-offset",
                        *[str(v) for v in args.video_follow_eye_offset],
                        "--video-follow-lookat-offset",
                        *[str(v) for v in args.video_follow_lookat_offset],
                    ]
                    if args.video_fixed_env:
                        cmd += ["--video-fixed-env"]
                    if args.video_hide_drone:
                        cmd += ["--video-hide-drone"]
                if str(getattr(args, "camera_risk_layout", "")).strip():
                    cmd += ["--camera-risk-layout", str(args.camera_risk_layout)]
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
    parser.add_argument(
        "--camera-risk-layout",
        default="",
        help="Optional camera risk layout override. Leave empty to infer from the checkpoint/task config; use '3' for legacy 3-sector or '3x3' for a grid.",
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
    parser.add_argument(
        "--export-success-videos",
        action="store_true",
        help="Save third-person follow RGB and onboard depth videos for successful episodes.",
    )
    parser.add_argument(
        "--videos-per-combo",
        type=int,
        default=0,
        help="Number of successful videos to save for each tree-spacing/speed combination.",
    )
    parser.add_argument(
        "--video-output-dir",
        default="",
        help="Directory for exported videos. Controller defaults to <run_output>/videos.",
    )
    parser.add_argument(
        "--video-env-index",
        type=int,
        default=0,
        help="Parallel env index to record when --video-fixed-env is set.",
    )
    parser.add_argument(
        "--video-fixed-env",
        action="store_true",
        help="Only record --video-env-index; default records whichever parallel env succeeds.",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=2,
        help="Capture one video frame every N simulation steps.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=0,
        help="Output video FPS. <=0 derives FPS from sim dt and --video-interval.",
    )
    parser.add_argument("--video-width", type=int, default=0, help="Optional RGB video width; 0 keeps viewport width.")
    parser.add_argument("--video-height", type=int, default=0, help="Optional RGB video height; 0 keeps aspect/height.")
    parser.add_argument("--video-depth-width", type=int, default=0, help="Optional depth video width; 0 follows RGB width.")
    parser.add_argument("--video-depth-height", type=int, default=0, help="Optional depth video height; 0 follows RGB height.")
    parser.add_argument("--video-depth-max", type=float, default=0.0,
                        help="Depth visualization max range in meters; <=0 uses task depth_max_range.")
    parser.add_argument("--video-max-frames", type=int, default=0,
                        help="Optional cap on frames kept per episode; 0 keeps the full successful episode.")
    parser.add_argument("--replay-interval", type=int, default=1,
                        help="Record one success-replay pose sample every N simulation steps.")
    parser.add_argument("--replay-tree-max-faces", type=int, default=2500,
                        help="Maximum source tree OBJ faces embedded once in each success replay JSON.")
    parser.add_argument(
        "--video-hide-drone",
        action="store_true",
        help="Keep the recorded drone hidden in third-person follow RGB videos.",
    )
    parser.add_argument(
        "--video-follow-eye-offset",
        type=float,
        nargs=3,
        default=[4.0, 0.0, 1.2],
        metavar=("BACK", "RIGHT", "BODY_UP"),
        help="Third-person follow-camera eye offset in the drone body frame.",
    )
    parser.add_argument(
        "--video-follow-lookat-offset",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.15],
        metavar=("FWD", "RIGHT", "BODY_UP"),
        help="Third-person follow-camera look-at offset in the drone body frame.",
    )
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
    if args.export_success_videos and int(args.videos_per_combo) <= 0:
        args.videos_per_combo = 1
    args.video_interval = max(1, int(args.video_interval))
    args.replay_interval = max(1, int(args.replay_interval))
    args.replay_tree_max_faces = max(100, int(args.replay_tree_max_faces))
    args.video_env_index = max(0, int(args.video_env_index))
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
