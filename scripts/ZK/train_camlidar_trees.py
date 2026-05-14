#!/usr/bin/env python3
"""
Training script for Cam+LiDAR policy using YOPO tree_mesh.obj as obstacles.

Key fixes compared with the previous version:
  1) Do not import omni_drones learning/torchrl utility modules before SimulationApp starts.
     Only import init_simulation_app immediately before creating SimulationApp.
  2) Do not reference OBJ directly as a USD reference. Instead, read the OBJ mesh in Python,
     instance it into one combined USD Mesh, and apply CollisionAPI to that mesh.
  3) Patch RayCaster mesh initialization so LiDAR/RayCaster can see the injected real-tree mesh.
  4) Add safer checkpoint loading and cleanup.

Important system note:
  If Isaac Sim prints "Failed to create change watch ... errno=28/No space left on device",
  first fix the Linux inotify watch limit or disk/inode usage. Code alone cannot fully solve that.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from pathlib import Path
from typing import Iterable, Tuple

# Must be set before torch / isaac imports.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Isaac Sim GPU foundation is usually unhappy with CUDA_VISIBLE_DEVICES remapping.
# Select devices through Hydra overrides instead, e.g. ++gpu_id=1 ++vulkan_gpu_id=1.
if "CUDA_VISIBLE_DEVICES" in os.environ:
    logging.warning(
        "CUDA_VISIBLE_DEVICES is set. Clearing it because Isaac Sim GPU foundation may break. "
        "Use ++gpu_id=N and ++vulkan_gpu_id=N instead."
    )
    del os.environ["CUDA_VISIBLE_DEVICES"]

import hydra
import importlib
import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path
from setproctitle import setproctitle


SCRIPT_DIR = Path(__file__).resolve().parent
OMNIDRONES_DIR = SCRIPT_DIR.parent.parent
REPO_ROOT = OMNIDRONES_DIR.parent
DEFAULT_TREE_OBJ = REPO_ROOT / "YOPO" / "Simulator" / "src" / "pointcloud" / "tree_mesh.obj"


# ============================================================
# Real-tree mesh utilities
# ============================================================
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

    # Keep start, goal, and map center relatively clean.
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


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _parse_face_index(token: str, vertex_count: int) -> int:
    """Parse OBJ face token like '12', '12/3/4', or '-1/2/3' into 0-based index."""
    raw = token.split("/", 1)[0]
    if not raw:
        raise ValueError(f"Invalid OBJ face token: {token!r}")
    idx = int(raw)
    if idx < 0:
        idx = vertex_count + idx
    else:
        idx = idx - 1
    return idx


def read_obj_mesh(path: str | Path, *, max_faces: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                poly = [_parse_face_index(tok, len(vertices)) for tok in parts]
                # Fan triangulation for polygons.
                for j in range(1, len(poly) - 1):
                    faces.append([poly[0], poly[j], poly[j + 1]])

    if not vertices or not faces:
        raise RuntimeError(f"OBJ has no readable vertices/faces: {path}")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    finite = np.isfinite(vertices).all(axis=1)
    if not finite.all():
        # Keep only finite vertices and remap faces.
        old_to_new = -np.ones(vertices.shape[0], dtype=np.int32)
        old_to_new[np.where(finite)[0]] = np.arange(int(finite.sum()), dtype=np.int32)
        face_mask = finite[faces].all(axis=1)
        faces = old_to_new[faces[face_mask]]
        vertices = vertices[finite]

    if int(max_faces) > 0 and faces.shape[0] > int(max_faces):
        # Deterministic uniform face sampling. This is not a geometric decimator,
        # but it is useful for making training scenes light enough.
        idx = np.linspace(0, faces.shape[0] - 1, int(max_faces), dtype=np.int64)
        faces = faces[idx]

    return vertices.astype(np.float32), faces.astype(np.int32)


def normalize_tree_mesh(
    vertices: np.ndarray,
    *,
    auto_upright: bool = True,
    recenter_xy: bool = True,
    ground_z: bool = True,
) -> np.ndarray:
    """Make tree mesh z-up, centered in XY, and rooted at z=0."""
    v = np.asarray(vertices, dtype=np.float32).copy()
    if v.size == 0:
        return v

    if auto_upright:
        extent = v.max(axis=0) - v.min(axis=0)
        height_axis = int(np.argmax(extent))
        if height_axis == 0:
            # old x -> new z
            v = v[:, [1, 2, 0]]
        elif height_axis == 1:
            # old y -> new z
            v = v[:, [0, 2, 1]]
        # if height_axis == 2, already z-up

    if recenter_xy:
        v[:, 0] -= float(np.mean(v[:, 0]))
        v[:, 1] -= float(np.mean(v[:, 1]))
    if ground_z:
        v[:, 2] -= float(np.min(v[:, 2]))
    return v.astype(np.float32)


def make_combined_tree_forest_mesh(
    tree_obj_path: str | Path,
    *,
    map_size: float,
    spacing: float,
    seed: int,
    scale_min: float,
    scale_max: float,
    tilt_deg: float,
    clear_radius: float,
    max_faces_per_tree: int = 8000,
    auto_upright: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instantiate one OBJ tree mesh many times and return one combined mesh."""
    base_vertices, base_faces = read_obj_mesh(tree_obj_path, max_faces=max_faces_per_tree)
    base_vertices = normalize_tree_mesh(base_vertices, auto_upright=auto_upright)

    positions = tree_positions_jittered_grid(
        map_size=map_size,
        spacing=spacing,
        seed=seed,
        clear_radius=clear_radius,
    )
    if positions.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), positions

    rng = np.random.default_rng(int(seed))
    scale_min_val = float(scale_min)
    scale_max_val = float(scale_max)
    if scale_max_val < scale_min_val:
        scale_min_val, scale_max_val = scale_max_val, scale_min_val
    max_tilt_rad = math.radians(float(tilt_deg))

    all_vertices = []
    all_faces = []
    vert_offset = 0

    for px, py in positions:
        scale = float(rng.uniform(scale_min_val, scale_max_val))
        roll = float(rng.uniform(-max_tilt_rad, max_tilt_rad))
        pitch = float(rng.uniform(-max_tilt_rad, max_tilt_rad))
        yaw = float(rng.uniform(-math.pi, math.pi))
        rot = _rotation_matrix(roll, pitch, yaw)
        pts = (base_vertices @ rot.T) * scale
        pts += np.asarray([float(px), float(py), 0.0], dtype=np.float32)
        pts[:, 2] = np.maximum(pts[:, 2], 0.02)
        all_vertices.append(pts.astype(np.float32))
        all_faces.append(base_faces.astype(np.int32) + int(vert_offset))
        vert_offset += pts.shape[0]

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)
    return vertices, faces, positions


@contextlib.contextmanager
def patched_realtree_forest(
    tree_obj_path: str | Path,
    map_size: float = 60.0,
    spacing: float = 4.0,
    seed: int = 0,
    scale_min: float = 0.5,
    scale_max: float = 1.0,
    tilt_deg: float = 10.0,
    clear_radius: float = 2.0,
    max_faces_per_tree: int = 8000,
    auto_upright: bool = True,
):
    """Monkey-patch terrain importer and RayCaster to install a real-tree OBJ forest."""
    import importlib as _importlib
    import isaaclab.terrains as terrains

    ray_caster_mod = _importlib.import_module("isaaclab.sensors.ray_caster.ray_caster")

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

        vertices, faces, positions = make_combined_tree_forest_mesh(
            tree_obj_path,
            map_size=float(map_size),
            spacing=float(spacing),
            seed=int(seed),
            scale_min=float(scale_min),
            scale_max=float(scale_max),
            tilt_deg=float(tilt_deg),
            clear_radius=float(clear_radius),
            max_faces_per_tree=int(max_faces_per_tree),
            auto_upright=bool(auto_upright),
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
        prim.CreateAttribute("realtree:spacing_m", Sdf.ValueTypeNames.Double).Set(float(spacing))

        print(
            "[realtree] installed combined tree mesh: "
            f"spacing={float(spacing):.3f}m trees={positions.shape[0]} "
            f"vertices={vertices.shape[0]} faces={faces.shape[0]} "
            f"max_faces_per_tree={int(max_faces_per_tree)}"
        )
        return int(positions.shape[0])

    def terrain_init_wrapper(self, cfg):
        original_terrain_init(self, cfg)
        install_realtree_mesh()

    def ray_initialize_all_meshes(self):
        """A more tolerant RayCaster mesh loader that combines Mesh children under the configured root."""
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
                plane = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                self.meshes[mesh_prim_path] = convert_to_warp_mesh(plane.vertices, plane.faces, device=self.device)
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
                    poly = idx[cursor:cursor + int(count)]
                    cursor += int(count)
                    if count < 3:
                        continue
                    for j in range(1, int(count) - 1):
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
                f"[realtree] RayCaster combined {len(all_points)} mesh(es) under {mesh_prim_path}: "
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


# ============================================================
# DualStreamBackbone
# ============================================================
class DualStreamBackbone(torch.nn.Module):
    """
    三输入 backbone: state + LiDAR-KU + camera risk.
    LiDAR 仍是主导航模态，camera risk 只通过轻量 gate 调制 LiDAR 特征。
    """

    def __init__(
        self,
        state_dim,
        lidar_dim=3200,
        camera_risk_dim=21,
        ku_value_max=20.0,
        output_dim=128,
        camera_risk_gate_alpha=0.2,
        camera_risk_fusion_mode="gate",
    ):
        super().__init__()
        if lidar_dim != 3200:
            raise ValueError(f"This backbone expects lidar_dim=3200, got {lidar_dim}.")
        if camera_risk_dim <= 0:
            raise ValueError(f"camera_risk_dim must be positive, got {camera_risk_dim}.")

        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.camera_risk_dim = int(camera_risk_dim)
        self.camera_risk_gate_alpha = float(camera_risk_gate_alpha)
        self.camera_risk_fusion_mode = str(camera_risk_fusion_mode).lower()

        self.ku_value_max = float(ku_value_max)
        self.ku_unknown_value = self.ku_value_max
        self.ku_h = 40
        self.ku_w = 80

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
        self.camera_risk_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.camera_risk_dim, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 64),
            torch.nn.ELU(),
        )
        self.camera_gate = torch.nn.Sequential(
            torch.nn.Linear(128 + 64, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.Sigmoid(),
        )

        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(64 + 128 + 64, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.ELU(),
        )

    def get_probe_params(self):
        return {
            "ku_encoder": self.ku_encoder[0].weight,
            "camera_risk": self.camera_risk_encoder[0].weight,
            "fusion": self.fusion_mlp[0].weight,
        }

    def forward(self, obs):
        state = obs[..., :self.state_dim]
        x_ku_flat = obs[..., self.state_dim:self.state_dim + self.lidar_dim]
        camera_risk_start = self.state_dim + self.lidar_dim
        camera_risk_end = camera_risk_start + self.camera_risk_dim
        camera_risk = obs[..., camera_risk_start:camera_risk_end]

        batch_shape = state.shape[:-1]
        b = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1

        x_ku_raw = x_ku_flat.reshape(b, 1, self.ku_h, self.ku_w)
        x_ku_raw = torch.nan_to_num(x_ku_raw, posinf=self.ku_unknown_value, neginf=0.0, nan=self.ku_unknown_value)
        x_ku_raw = torch.clamp(x_ku_raw, 0.0, self.ku_unknown_value)
        lidar_feat = self.ku_encoder(x_ku_raw / self.ku_value_max)
        lidar_z = self.ku_global_head(lidar_feat)

        state_2d = state.reshape(b, self.state_dim)
        camera_risk_2d = camera_risk.reshape(b, self.camera_risk_dim)
        camera_risk_2d = torch.nan_to_num(camera_risk_2d, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        state_z = self.state_encoder(state_2d)
        camera_z = self.camera_risk_encoder(camera_risk_2d)

        if self.camera_risk_fusion_mode == "gate":
            gate = self.camera_gate(torch.cat([lidar_z, camera_z], dim=-1))
            lidar_z = lidar_z * (1.0 - self.camera_risk_gate_alpha * gate)

        fused = torch.cat([state_z, lidar_z, camera_z], dim=-1)
        out = self.fusion_mlp(fused)
        return out.reshape(*batch_shape, -1)


def _load_policy_checkpoint_compatible(model, ckpt_path, map_location):
    raw_state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(raw_state, dict):
        if "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
            raw_state = raw_state["state_dict"]
        elif "model_state_dict" in raw_state and isinstance(raw_state["model_state_dict"], dict):
            raw_state = raw_state["model_state_dict"]
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint format at {ckpt_path}: {type(raw_state)}")

    model_state = model.state_dict()
    filtered_state = {}
    skipped_mismatch = []
    for key, value in raw_state.items():
        if key not in model_state:
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value

    load_info = model.load_state_dict(filtered_state, strict=False)
    logging.info(
        "Checkpoint loaded: %d params matched, %d shape mismatches, missing=%d, unexpected=%d",
        len(filtered_state), len(skipped_mismatch), len(load_info.missing_keys), len(load_info.unexpected_keys)
    )
    if skipped_mismatch:
        logging.warning("Skipped shape mismatches: %s", skipped_mismatch[:10])
    return load_info


def _resolve_goodpt_path(goodpt_path_cfg: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_path = goodpt_path_cfg
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(script_dir, candidate_path)
    candidate_path = os.path.abspath(candidate_path)
    if not os.path.isdir(candidate_path):
        return candidate_path

    final_ckpt = os.path.join(candidate_path, "checkpoint_final.pt")
    if os.path.exists(final_ckpt):
        return final_ckpt
    pt_files = [os.path.join(candidate_path, f) for f in os.listdir(candidate_path) if f.endswith(".pt")]
    if pt_files:
        return max(pt_files, key=os.path.getmtime)
    return candidate_path


# ============================================================
# Evaluation helper
# ============================================================
@torch.no_grad()
def _evaluate(env, policy, base_env, cfg):
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

    num_eval = min(int(cfg.get("eval_num_envs", 10)), int(base_env.num_envs))
    env.eval()

    td = env.reset()
    returns = torch.zeros(num_eval, device=base_env.device)
    successes = torch.zeros(num_eval, device=base_env.device)
    done_all = torch.zeros(num_eval, dtype=torch.bool, device=base_env.device)

    with set_exploration_type(ExplorationType.MODE):
        for _ in range(int(cfg.get("max_steps", 1500))):
            td = policy(td)
            td = env.step(td)
            returns += td[("next", "agents", "reward")].reshape(num_eval).float()
            done = td[("next", "done")].reshape(num_eval).bool()
            stats = td[("next", "stats")]
            if "success" in stats.keys():
                successes = torch.maximum(successes, (stats["success"].reshape(num_eval) >= 0.5).to(torch.float32))
            done_all = done_all | done
            if done_all.all():
                break
            td = step_mdp(td)

    env.train()
    return {
        "eval/success_rate": successes.mean().item(),
        "eval/mean_return": returns.mean().item(),
    }


def _suppress_noisy_isaac_warnings():
    """Reduce known Isaac Sim warning spam that obscures training logs."""
    try:
        import omni.log

        log = omni.log.get_log()
        for channel in (
            "isaacsim.core.simulation_manager.plugin",
        ):
            log.set_channel_level(channel, omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE)
    except Exception:
        # Best-effort only. Training should proceed even if log controls are unavailable.
        pass


# ============================================================
# Main training
# ============================================================
@hydra.main(version_base=None, config_path="", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # GPU config: do not use CUDA_VISIBLE_DEVICES. Use cfg keys.
    gpu_id = int(cfg.get("gpu_id", cfg.get("sim_gpu_index", 0)))
    vulkan_gpu_id = int(cfg.get("vulkan_gpu_id", cfg.get("active_gpu", 1)))
    cfg.sim.device = f"cuda:{gpu_id}"
    cfg.sim.active_gpu = vulkan_gpu_id
    cfg.sim.physics_gpu = gpu_id

    # Tree config.
    tree_cfg = cfg.get("tree", {})
    tree_obj_path = str(tree_cfg.get("obj_path", str(DEFAULT_TREE_OBJ)))
    tree_map_size = float(tree_cfg.get("map_size", 60.0))
    tree_spacing = float(tree_cfg.get("spacing", 6.0))
    tree_scale_min = float(tree_cfg.get("scale_min", 0.35))
    tree_scale_max = float(tree_cfg.get("scale_max", 0.55))
    tree_tilt_deg = float(tree_cfg.get("tilt_deg", 5.0))
    tree_clear_radius = float(tree_cfg.get("clear_radius", 6.0))
    tree_seed = int(tree_cfg.get("seed", cfg.seed))
    tree_max_faces_per_tree = int(tree_cfg.get("max_faces_per_tree", 20000))
    tree_auto_upright = bool(tree_cfg.get("auto_upright", True))

    needs_depth_camera = bool(
        cfg.task.get("use_depth_ku_observation", False)
        or cfg.task.get("use_camera_risk_observation", False)
    )

    display_env = os.environ.get("DISPLAY", "").strip()
    has_display = len(display_env) > 0
    force_headless_no_display = bool(cfg.get("force_headless_no_display", True))
    if force_headless_no_display and not has_display:
        logging.warning("DISPLAY is not set; forcing headless mode.")
        cfg.headless = True
        cfg.sim.enable_viewport = False
        if "task" in cfg and cfg.task.get("show_depth_preview_window", False):
            cfg.task.show_depth_preview_window = False

    cfg.sim.enable_viewport = (not bool(cfg.headless)) and has_display
    cfg.sim.enable_replicator = True
    if needs_depth_camera and not cfg.sim.enable_replicator:
        raise RuntimeError("Camera/depth observations enabled but replicator is off.")

    simulation_app = None
    run = None
    try:
        # Import only init_simulation_app before starting SimulationApp.
        from omni_drones import init_simulation_app
        simulation_app = init_simulation_app(cfg)
        _suppress_noisy_isaac_warnings()

        # Delayed imports: these should happen after SimulationApp is alive.
        from omni_drones.envs.isaac_env import IsaacEnv
        from omni_drones.learning import ALGOS
        from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
        from omni_drones.utils.torchrl.transforms import FromMultiDiscreteAction, FromDiscreteAction, ravel_composite
        from omni_drones.utils.wandb import init_wandb
        from torchrl.data import CompositeSpec
        from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

        run = init_wandb(cfg)
        if run is not None:
            setproctitle(run.name)
        print(OmegaConf.to_yaml(cfg))

        # Create environment with real-tree forest.
        with patched_realtree_forest(
            tree_obj_path=tree_obj_path,
            map_size=tree_map_size,
            spacing=tree_spacing,
            seed=tree_seed,
            scale_min=tree_scale_min,
            scale_max=tree_scale_max,
            tilt_deg=tree_tilt_deg,
            clear_radius=tree_clear_radius,
            max_faces_per_tree=tree_max_faces_per_tree,
            auto_upright=tree_auto_upright,
        ):
            task_name = str(cfg.task.name)
            try:
                importlib.import_module(f"omni_drones.envs.single.{task_name.lower()}")
            except ModuleNotFoundError:
                pass
            env_class = IsaacEnv.REGISTRY[cfg.task.name]
            base_env = env_class(cfg, headless=cfg.headless)

        # Transforms.
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
            action_transform = str(action_transform)
            if action_transform.startswith("multidiscrete"):
                transforms.append(FromMultiDiscreteAction(nbins=int(action_transform.split(":")[1])))
            elif action_transform.startswith("discrete"):
                transforms.append(FromDiscreteAction(nbins=int(action_transform.split(":")[1])))
            else:
                raise NotImplementedError(f"Unknown action transform: {action_transform}")

        env = TransformedEnv(base_env, Compose(*transforms)).train()
        env.set_seed(cfg.seed)

        # Policy.
        try:
            policy = ALGOS[cfg.algo.name.lower()](
                cfg.algo, env.observation_spec, env.action_spec, env.reward_spec,
                device=base_env.device,
            )
        except KeyError as exc:
            raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}") from exc

        # Inject DualStreamBackbone.
        actor_lr = policy.actor_opt.param_groups[0]["lr"]
        critic_lr = policy.critic_opt.param_groups[0]["lr"]

        obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
        lidar_dim = 3200
        ku_value_max = float(cfg.task.get("ku_value_max", 20.0))
        expected_camera_risk_dim = int(cfg.task.get("camera_risk_num_bins", 5)) * int(
            cfg.task.get("camera_risk_features_per_bin", 4)
        )
        if bool(cfg.task.get("camera_risk_add_stale_ratio", True)):
            expected_camera_risk_dim += 1
        state_dim = int(obs_dim - lidar_dim - expected_camera_risk_dim)
        camera_risk_dim = int(obs_dim - state_dim - lidar_dim)

        if state_dim <= 0 or camera_risk_dim <= 0:
            raise RuntimeError(
                f"Invalid observation split: obs_dim={obs_dim}, state_dim={state_dim}, "
                f"lidar_dim={lidar_dim}, camera_risk_dim={camera_risk_dim}"
            )

        camera_risk_gate_alpha = float(cfg.task.get("camera_risk_gate_alpha", 0.2))
        camera_risk_fusion_mode = str(cfg.task.get("camera_risk_fusion_mode", "gate"))
        expected_feature_dim = 128

        actor_backbone = DualStreamBackbone(
            state_dim=state_dim, lidar_dim=lidar_dim, camera_risk_dim=camera_risk_dim,
            ku_value_max=ku_value_max, output_dim=expected_feature_dim,
            camera_risk_gate_alpha=camera_risk_gate_alpha,
            camera_risk_fusion_mode=camera_risk_fusion_mode,
        ).to(base_env.device)
        critic_backbone = DualStreamBackbone(
            state_dim=state_dim, lidar_dim=lidar_dim, camera_risk_dim=camera_risk_dim,
            ku_value_max=ku_value_max, output_dim=expected_feature_dim,
            camera_risk_gate_alpha=camera_risk_gate_alpha,
            camera_risk_fusion_mode=camera_risk_fusion_mode,
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

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        actor_backbone.apply(init_weights)
        critic_backbone.apply(init_weights)
        policy.actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=actor_lr)
        policy.critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)

        print(
            f"✅ DualStreamBackbone injected: state_dim={state_dim}, lidar_dim={lidar_dim}, "
            f"camera_risk_dim={camera_risk_dim}, fusion={camera_risk_fusion_mode}"
        )

        # Optional warm start.
        init_mode = str(cfg.get("init_mode", "scratch")).lower()
        goodpt_path_cfg = str(cfg.get("goodpt_path", "")).strip()
        if init_mode == "goodpt":
            if not goodpt_path_cfg:
                raise ValueError("init_mode=goodpt but goodpt_path is empty.")
            resolved_ckpt_path = _resolve_goodpt_path(goodpt_path_cfg)
            if os.path.exists(resolved_ckpt_path):
                _load_policy_checkpoint_compatible(policy, resolved_ckpt_path, map_location=base_env.device)
                logging.info("Loaded checkpoint: %s", resolved_ckpt_path)
            else:
                raise FileNotFoundError(f"goodpt checkpoint not found: {resolved_ckpt_path}")

        # Training loop.
        frames_per_batch = env.num_envs * int(cfg.algo.train_every)
        total_frames = cfg.get("total_frames", -1)
        if int(total_frames) > 0:
            total_frames = int(total_frames) // frames_per_batch * frames_per_batch
        max_iters = int(cfg.get("max_iters", -1))
        eval_interval = int(cfg.get("eval_interval", -1))
        save_interval = int(cfg.get("save_interval", -1))

        stats_keys = [k for k in base_env.observation_spec.keys(True, True) if isinstance(k, tuple) and k[0] == "stats"]
        episode_stats = EpisodeStats(stats_keys)

        collector = SyncDataCollector(
            env, policy=policy, frames_per_batch=frames_per_batch,
            total_frames=total_frames, device=cfg.sim.device, return_same_td=True,
        )

        pbar_iters = max_iters if max_iters > 0 else (total_frames // frames_per_batch if total_frames > 0 else None)
        from tqdm import tqdm
        pbar = tqdm(collector, total=pbar_iters, dynamic_ncols=True)

        env.train()
        for i, data in enumerate(pbar):
            if max_iters > 0 and i >= max_iters:
                break

            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            episode_stats.add(data.to_tensordict())

            for _ in range(int(cfg.algo.ppo_epochs)):
                policy.train_op(data.to_tensordict())

            stats_dict = episode_stats.pop()
            if stats_dict is not None:
                for key, val in stats_dict.items():
                    try:
                        info[f"train/{key}"] = val.detach().float().mean().item()
                    except (AttributeError, RuntimeError):
                        pass

            if eval_interval > 0 and (i + 1) % eval_interval == 0:
                info.update(_evaluate(env, policy, base_env, cfg))

            run_dir = run.dir if run is not None else os.getcwd()
            if save_interval > 0 and (i + 1) % save_interval == 0:
                ckpt_path = os.path.join(run_dir, f"checkpoint_step_{collector._frames}.pt")
                torch.save({"model_state_dict": policy.state_dict()}, ckpt_path)
                info["checkpoint_path"] = ckpt_path

            if run is not None:
                run.log(info)

            info_str = f"frames={collector._frames}"
            if "train/return" in info:
                info_str += f" return={info['train/return']:.2f}"
            if "eval/success_rate" in info:
                info_str += f" eval_success={info['eval/success_rate']:.2f}"
            pbar.set_description(info_str)

        run_dir = run.dir if run is not None else os.getcwd()
        final_ckpt = os.path.join(run_dir, "checkpoint_final.pt")
        torch.save({"model_state_dict": policy.state_dict()}, final_ckpt)
        print(f"Training done. Final checkpoint: {final_ckpt}")

    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
