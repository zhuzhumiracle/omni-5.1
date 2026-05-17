#!/usr/bin/env python3
"""
Training script for camera-gated Cam+LiDAR policy using YOPO tree_mesh.obj as obstacles.

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
import sys
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
# Camera geometry + spatial-gate DualStreamBackbone
# ============================================================
def _normalize_np(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def _expected_row_camera_rotation_from_view(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    up_axis: np.ndarray | None = None,
) -> np.ndarray:
    if up_axis is None:
        up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = _normalize_np(target_pos - camera_pos)
    up_axis = _normalize_np(up_axis)
    right = np.cross(forward, up_axis)
    if np.linalg.norm(right) <= 1e-9:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
        if np.linalg.norm(right) <= 1e-9:
            raise ValueError("Depth camera forward axis is degenerate.")
    right = _normalize_np(right)
    up = _normalize_np(np.cross(right, forward))
    return np.stack([right, up, -forward], axis=0)


def _rotation_error_deg(candidate: np.ndarray, reference: np.ndarray) -> float:
    relative_rot = candidate.T @ reference
    trace_val = float(np.trace(relative_rot))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace_val - 1.0)))
    return float(np.degrees(np.arccos(cos_angle)))


def _select_camera_rotation_convention(
    rot_raw: np.ndarray,
    expected_row_rot: np.ndarray,
) -> tuple[np.ndarray, str, float, float]:
    row_err_deg = _rotation_error_deg(rot_raw, expected_row_rot)
    col_err_deg = _rotation_error_deg(rot_raw.T, expected_row_rot)
    if col_err_deg + 1e-9 < row_err_deg:
        return rot_raw.T.copy(), "column-vector-transposed", row_err_deg, col_err_deg
    return rot_raw.copy(), "row-vector", row_err_deg, col_err_deg


def _read_depth_camera_geometry_from_stage(base_env):
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom
    except Exception as exc:
        raise RuntimeError("Failed to import Omniverse USD modules for camera geometry.") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available; cannot read depth camera geometry.")

    depth_prim_path = f"/World/envs/env_0/{base_env.drone.name}_0/base_link/{base_env.depth_prim_name}"
    base_prim_path = f"/World/envs/env_0/{base_env.drone.name}_0/base_link"
    prim = stage.GetPrimAtPath(depth_prim_path)
    base_prim = stage.GetPrimAtPath(base_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Depth camera prim not found or invalid: {depth_prim_path}")
    if not base_prim or not base_prim.IsValid():
        raise RuntimeError(f"Base link prim not found or invalid: {base_prim_path}")

    camera = UsdGeom.Camera(prim)
    focal_length = camera.GetFocalLengthAttr().Get()
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    vertical_aperture = camera.GetVerticalApertureAttr().Get()
    horizontal_aperture_offset = camera.GetHorizontalApertureOffsetAttr().Get() or 0.0
    vertical_aperture_offset = camera.GetVerticalApertureOffsetAttr().Get() or 0.0
    clipping_range = camera.GetClippingRangeAttr().Get()
    if focal_length is None or horizontal_aperture is None:
        raise RuntimeError(f"Depth camera {depth_prim_path} is missing focal/aperture attributes.")
    if vertical_aperture is None:
        vertical_aperture = float(horizontal_aperture) * float(base_env.depth_h) / float(max(1, base_env.depth_w))

    def _extract_pose_from_matrix(matrix_gf: "Gf.Matrix4d"):
        translation = matrix_gf.ExtractTranslation()
        quat = matrix_gf.ExtractRotationQuat()
        rot = np.array(Gf.Matrix3d(quat), dtype=np.float64)
        pos = np.array([translation[0], translation[1], translation[2]], dtype=np.float64)
        return pos, rot

    def _orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
        u, _, vh = np.linalg.svd(rot)
        rot_ortho = u @ vh
        if np.linalg.det(rot_ortho) < 0.0:
            u[:, -1] *= -1.0
            rot_ortho = u @ vh
        return rot_ortho

    cam_in_base_gf = omni.usd.get_local_transform_matrix(prim)
    if not isinstance(cam_in_base_gf, Gf.Matrix4d):
        cam_in_base_gf = Gf.Matrix4d(cam_in_base_gf)
    cam_pos_np, cam_rot_np = _extract_pose_from_matrix(cam_in_base_gf)
    cam_rot_np = _orthonormalize_rotation(cam_rot_np)
    relative_transform_source = "omni.usd.get_local_transform_matrix"

    try:
        from isaacsim.core.includes.pose import getRelativeTransform  # type: ignore

        official = getRelativeTransform(stage, None, prim.GetPath(), base_prim.GetPath())
        if not isinstance(official, Gf.Matrix4d):
            official = Gf.Matrix4d(official)
        official_pos, official_rot = _extract_pose_from_matrix(official)
        official_rot = _orthonormalize_rotation(official_rot)
        pos_err = float(np.linalg.norm(official_pos - cam_pos_np))
        rot_err = _rotation_error_deg(official_rot, cam_rot_np)
        if pos_err <= 1e-5 and rot_err <= 1e-3:
            cam_pos_np = official_pos
            cam_rot_np = official_rot
            relative_transform_source = "isaacsim.core.includes.pose.getRelativeTransform"
        else:
            logging.warning(
                "Depth camera relative transform mismatch; using local transform fallback. "
                "pos_err=%.6e rot_err_deg=%.6e",
                pos_err,
                rot_err,
            )
    except ImportError:
        pass

    fx_px = float(base_env.depth_w) * float(focal_length) / float(horizontal_aperture)
    fy_px = float(base_env.depth_h) * float(focal_length) / float(vertical_aperture)
    cx_px = 0.5 * float(base_env.depth_w) + float(horizontal_aperture_offset) * fx_px
    cy_px = 0.5 * float(base_env.depth_h) + float(vertical_aperture_offset) * fy_px
    intrinsic_matrix = [
        [fx_px, 0.0, cx_px],
        [0.0, fy_px, cy_px],
        [0.0, 0.0, 1.0],
    ]

    depth_cam_pos_cfg = base_env.cfg.task.get("depth_camera_pos", [0.12, 0.0, 0.03])
    depth_cam_target_cfg = base_env.cfg.task.get("depth_camera_target", [2.0, 0.0, 0.03])
    expected_rot = _expected_row_camera_rotation_from_view(
        np.array(depth_cam_pos_cfg, dtype=np.float64),
        np.array(depth_cam_target_cfg, dtype=np.float64),
    )
    cam_rot_np, rotation_convention, row_err_deg, col_err_deg = _select_camera_rotation_convention(
        cam_rot_np.astype(np.float64),
        expected_rot,
    )

    optical_axis_camera = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    optical_axis_lidar = optical_axis_camera @ cam_rot_np
    optical_axis_lidar = optical_axis_lidar / max(np.linalg.norm(optical_axis_lidar), 1e-12)

    logging.info(
        "Depth camera geometry %s via %s | fx=%.3f fy=%.3f fov_x=%.3f deg convention=%s row_err=%.3f col_err=%.3f axis=%s",
        depth_prim_path,
        relative_transform_source,
        fx_px,
        fy_px,
        math.degrees(2.0 * math.atan(float(base_env.depth_w) / (2.0 * fx_px))),
        rotation_convention,
        row_err_deg,
        col_err_deg,
        tuple(float(v) for v in optical_axis_lidar.tolist()),
    )

    near_clip = None
    far_clip = None
    if clipping_range is not None and len(clipping_range) >= 2:
        near_clip = float(clipping_range[0])
        far_clip = float(clipping_range[1])

    return {
        "prim_path": depth_prim_path,
        "relative_transform_source": relative_transform_source,
        "intrinsic_matrix": intrinsic_matrix,
        "near_clip": near_clip,
        "far_clip": far_clip,
    }


class DualStreamBackbone(torch.nn.Module):
    """
    三输入 backbone：state + LiDAR-KU + camera risk。
    相机前方 FoV 被均分为 num_rows (pitch) × num_cols (yaw) 个 2D 扇区，
    每扇区独立计算 gate，在像素级调制对应 LiDAR KU 区域。
    扇区布局由 camera_risk_num_rows / camera_risk_num_cols 单一控制。
    """

    def __init__(
        self,
        state_dim,
        lidar_dim=3200,
        camera_risk_dim=13,
        ku_value_max=20.0,
        output_dim=128,
        camera_h_fov_rad=None,
        fov_pitch_range=None,
        num_rows=3,
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

        # 相机水平 FoV（从环境配置传入）
        if camera_h_fov_rad is None:
            camera_h_fov_rad = 2.0 * math.atan(160.0 / (2.0 * 320.0))
        self.camera_h_fov_rad = float(camera_h_fov_rad)
        if fov_pitch_range is None:
            fov_pitch_range = (-math.pi / 2, math.pi / 2)
        self.fov_pitch_min = float(fov_pitch_range[0])
        self.fov_pitch_max = float(fov_pitch_range[1])

        # 预计算 2D 扇区掩码
        self._build_sector_2d_masks()

        # ================= 1. LiDAR-KU 编码分支 =================
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

        # ================= 2. 自身状态编码 =================
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
        )

        # ================= 3. 逐扇区空间 Gate 头 =================
        total_sectors = self.num_rows * self.num_cols
        self.gate_heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.features_per_sector, 1),
                torch.nn.Sigmoid(),
            )
            for _ in range(total_sectors)
        ])

        # ================= 4. 最终融合 MLP =================
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

    def get_probe_params(self):
        """Expose representative parameters for optimizer/gradient debug checks."""
        return {
            "ku_encoder": self.ku_encoder[0].weight,
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
            sector_mask = self._sector_masks[i]
            if bool(sector_mask.any()):
                spatial_gate[:, :, sector_mask] = gate_i.view(b, 1, 1)

        x_ku_gated = x_ku_raw * spatial_gate

        lidar_feat = self.ku_encoder(x_ku_gated / self.ku_value_max)
        lidar_z = self.ku_global_head(lidar_feat)

        state_2d = state.reshape(b, self.state_dim)
        state_z = self.state_encoder(state_2d)

        fused = torch.cat([state_z, lidar_z], dim=-1)
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
def _evaluate(env, policy, base_env, cfg, *, run_dir=None, step_frames=None):
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

    num_eval = min(int(cfg.get("eval_num_envs", 10)), int(base_env.num_envs))
    should_render = (int(base_env.num_envs) <= int(cfg.get("eval_render_max_envs", 200))) and bool(cfg.sim.enable_viewport)
    should_record = bool(cfg.get("eval_record_video", True)) and (
        should_render or bool(cfg.sim.get("enable_replicator", False))
    )
    eval_video_interval = max(1, int(cfg.get("eval_video_interval", 1)))
    render_callback = None

    if should_record:
        try:
            from omni_drones.utils.torchrl import RenderCallback

            base_env.enable_render(True)
            render_callback = RenderCallback(interval=eval_video_interval)
            print(f"🎥 [评估阶段] 已开启 wandb 视频/图像记录。采样间隔={eval_video_interval}")
        except Exception as exc:
            logging.warning("Failed to enable eval rendering: %s", exc)
            render_callback = None
    else:
        try:
            base_env.enable_render(False)
        except Exception:
            pass

    env.eval()
    base_env.eval()
    info = {}
    try:
        td = env.reset()
        returns = torch.zeros(num_eval, device=base_env.device)
        successes = torch.zeros(num_eval, device=base_env.device)
        done_all = torch.zeros(num_eval, dtype=torch.bool, device=base_env.device)
        first_episode_stats = {}
        latest_stats = None

        with set_exploration_type(ExplorationType.MODE):
            for _ in range(int(cfg.get("max_steps", 1500))):
                td = policy(td)
                td = env.step(td)

                if render_callback is not None:
                    try:
                        render_callback(base_env)
                    except Exception as exc:
                        logging.warning("Eval render callback failed: %s", exc)
                        render_callback = None

                rewards = td[("next", "agents", "reward")].reshape(int(base_env.num_envs)).float()[:num_eval]
                returns += rewards
                done = td[("next", "done")].reshape(int(base_env.num_envs)).bool()[:num_eval]
                stats = td[("next", "stats")]
                latest_stats = stats
                if "success" in stats.keys():
                    success_now = stats["success"].reshape(int(base_env.num_envs))[:num_eval]
                    successes = torch.maximum(successes, (success_now >= 0.5).to(torch.float32))

                just_finished = done & (~done_all)
                if just_finished.any():
                    for key, val in stats.items():
                        if key not in first_episode_stats:
                            first_episode_stats[key] = torch.zeros(
                                num_eval,
                                device=base_env.device,
                                dtype=val.dtype,
                            )
                        val_eval = val.reshape(int(base_env.num_envs), -1)[:num_eval, 0]
                        first_episode_stats[key][just_finished] = val_eval[just_finished]
                done_all = done_all | done
                if done_all.all():
                    break
                td = step_mdp(td)

        if latest_stats is not None and len(first_episode_stats) > 0 and (~done_all).any():
            for key, val in latest_stats.items():
                if key not in first_episode_stats:
                    first_episode_stats[key] = torch.zeros(
                        num_eval,
                        device=base_env.device,
                        dtype=val.dtype,
                    )
                val_eval = val.reshape(int(base_env.num_envs), -1)[:num_eval, 0]
                first_episode_stats[key][~done_all] = val_eval[~done_all]

        info.update({
            "eval/success_rate": successes.mean().item(),
            "eval/mean_return": returns.mean().item(),
        })
        for key, val in first_episode_stats.items():
            try:
                info[f"eval/stats.{key}"] = val.detach().float().mean().item()
            except (AttributeError, RuntimeError):
                pass

        if render_callback is not None and len(render_callback.frames) > 0:
            import wandb

            video_array = render_callback.get_video_array(axes="t c h w")
            fps_val = max(10, int(1.0 / (cfg.sim.dt * cfg.sim.substeps * eval_video_interval)))
            info["recording"] = wandb.Video(video_array, fps=fps_val, format="mp4")

            video_np = np.asarray(video_array)
            last_frame = np.transpose(video_np[-1], (1, 2, 0))
            info["eval/final_frame"] = wandb.Image(
                last_frame,
                caption=f"eval_step_{step_frames if step_frames is not None else 0}",
            )

            if run_dir is not None:
                try:
                    import imageio

                    video_local = np.transpose(video_np, (0, 2, 3, 1))
                    if video_local.dtype != np.uint8:
                        if video_local.max() <= 1.0:
                            video_local = (video_local * 255).astype(np.uint8)
                        else:
                            video_local = video_local.astype(np.uint8)
                    local_video_name = os.path.join(
                        run_dir,
                        f"eval_video_step_{step_frames if step_frames is not None else 0}.mp4",
                    )
                    imageio.mimwrite(
                        local_video_name,
                        video_local,
                        fps=fps_val,
                        macro_block_size=None,
                        quality=9,
                    )
                    info["eval/video_path"] = local_video_name
                    print(f"🎥 [录像已保存] 本地视频路径: {local_video_name}")
                except ImportError:
                    logging.warning("imageio is not installed; skipped local eval video save.")
                except Exception as exc:
                    logging.warning("Failed to save local eval video: %s", exc)

        return info
    finally:
        try:
            base_env.enable_render((not bool(cfg.headless)) and bool(cfg.sim.enable_viewport))
        except Exception:
            pass
        env.train()
        base_env.train()


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

    if str(cfg.task.name) != "forest_lc_gate":
        logging.warning(
            "train_canlidargate_trees.py is intended for task=forest_lc_gate, got task=%s.",
            cfg.task.name,
        )
    if (
        str(cfg.task.get("control_mode", "")).lower() == "velocity"
        and str(cfg.task.get("target_yaw_mode", "")).lower() == "action"
        and int(cfg.task.get("velocity_action_dim", 5)) != 5
    ):
        logging.warning(
            "Using decoupled velocity action [dir_x, dir_y, dir_z, speed_ratio, yaw]; "
            "overriding velocity_action_dim=%s to 5.",
            cfg.task.get("velocity_action_dim"),
        )
        cfg.task.velocity_action_dim = 5

    vlim_override = cfg.get("vlim", None)
    if vlim_override is not None:
        cfg.task.vlim = float(vlim_override)
    cfg.task.vlim = float(cfg.task.get("vlim", cfg.task.get("v_max", 8.0)))
    if bool(cfg.get("sync_vlim_to_v_max", True)):
        cfg.task.v_max = float(cfg.task.vlim)
    if bool(cfg.task.get("observe_vlim", False)) and not bool(cfg.task.get("vlim_randomize", False)):
        cfg.task.vlim_train_min = float(cfg.task.vlim)
        cfg.task.vlim_train_max = float(cfg.task.vlim)

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
    tree_scale_max = float(tree_cfg.get("scale_max", 0.6))
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
            # 2D grid layout: rows=pitch, cols=yaw.  Backward compat: old camera_risk_num_bins -> 1 row × N cols.
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
                logging.warning(
                    "camera_risk_dim=%d derived from observation, expected %d from config. "
                    "Using derived dimension to match the environment.",
                    camera_risk_dim,
                    expected_camera_risk_dim,
                )

            camera_geom = _read_depth_camera_geometry_from_stage(base_env)
            fx = camera_geom["intrinsic_matrix"][0][0]
            fy = camera_geom["intrinsic_matrix"][1][1]
            depth_h_cfg = int(cfg.task.get("depth_resolution", [96, 160])[0])
            depth_w_cfg = int(cfg.task.get("depth_resolution", [96, 160])[1])
            camera_h_fov_rad = 2.0 * math.atan(depth_w_cfg / (2.0 * fx))
            camera_v_fov_rad = 2.0 * math.atan(depth_h_cfg / (2.0 * fy))
            # Effective pitch range = camera vFoV ∩ LiDAR vFoV
            _lidar_vfov = cfg.task.get("lidar_vfov", [-7., 52.])
            lidar_pitch_min = math.radians(float(_lidar_vfov[0]))
            lidar_pitch_max = math.radians(float(_lidar_vfov[1]))
            fov_pitch_min = max(-camera_v_fov_rad / 2.0, lidar_pitch_min)
            fov_pitch_max = min(camera_v_fov_rad / 2.0, lidar_pitch_max)
            expected_feature_dim = 128

            actor_backbone = DualStreamBackbone(
                state_dim=state_dim, lidar_dim=lidar_dim, camera_risk_dim=camera_risk_dim,
                ku_value_max=ku_value_max, output_dim=expected_feature_dim,
                camera_h_fov_rad=camera_h_fov_rad,
                fov_pitch_range=(fov_pitch_min, fov_pitch_max),
                num_rows=num_rows, num_cols=num_cols, features_per_sector=features_per_sector,
            ).to(base_env.device)
            critic_backbone = DualStreamBackbone(
                state_dim=state_dim, lidar_dim=lidar_dim, camera_risk_dim=camera_risk_dim,
                ku_value_max=ku_value_max, output_dim=expected_feature_dim,
                camera_h_fov_rad=camera_h_fov_rad,
                fov_pitch_range=(fov_pitch_min, fov_pitch_max),
                num_rows=num_rows, num_cols=num_cols, features_per_sector=features_per_sector,
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
                f"✅ Spatial-gate DualStreamBackbone injected: state_dim={state_dim}, lidar_dim={lidar_dim}, "
                f"camera_risk_dim={camera_risk_dim}, grid={num_rows}×{num_cols}, "
                f"camera_h_fov_rad={camera_h_fov_rad:.4f}, "
                f"vlim={float(cfg.task.vlim):.3f}"
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
            max_return = -float("inf")
            last_best_return_ckpt_path = None

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
            success_ema = None
            return_ema = None
            ema_alpha = float(cfg.get("train_metric_ema_alpha", 0.1))
            for i, data in enumerate(pbar):
                if max_iters > 0 and i >= max_iters:
                    break

                info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
                run_dir = run.dir if run is not None else os.getcwd()
                episode_stats.add(data.to_tensordict())

                if len(episode_stats) > 0:
                    for key, val in episode_stats.pop().items(True, True):
                        try:
                            key_name = ".".join(key) if isinstance(key, tuple) else key
                            info[f"train/{key_name}"] = val.detach().float().mean().item()
                        except (AttributeError, RuntimeError):
                            pass

                with torch.no_grad():
                    actor_probe_before = {
                        k: v.detach().clone() for k, v in actor_backbone.get_probe_params().items()
                    }
                    critic_probe_before = {
                        k: v.detach().clone() for k, v in critic_backbone.get_probe_params().items()
                    }

                train_info = policy.train_op(data.to_tensordict())
                info.update(train_info)

                with torch.no_grad():
                    actor_probe_after = actor_backbone.get_probe_params()
                    critic_probe_after = critic_backbone.get_probe_params()
                    actor_deltas = {
                        k: (actor_probe_after[k].detach() - actor_probe_before[k]).abs().mean().item()
                        for k in actor_probe_before
                    }
                    critic_deltas = {
                        k: (critic_probe_after[k].detach() - critic_probe_before[k]).abs().mean().item()
                        for k in critic_probe_before
                    }
                    for key, val in actor_deltas.items():
                        info[f"debug/actor_delta_{key}"] = val
                    for key, val in critic_deltas.items():
                        info[f"debug/critic_delta_{key}"] = val
                    info["debug/actor_backbone_delta"] = max(actor_deltas.values())
                    info["debug/critic_backbone_delta"] = max(critic_deltas.values())

                if "train/stats.success" in info:
                    current_success = float(info["train/stats.success"])
                    success_ema = current_success if success_ema is None else (
                        ema_alpha * current_success + (1.0 - ema_alpha) * success_ema
                    )
                    info["train/stats.success_ema"] = success_ema
                if "train/stats.return" in info:
                    current_return_for_ema = float(info["train/stats.return"])
                    return_ema = current_return_for_ema if return_ema is None else (
                        ema_alpha * current_return_for_ema + (1.0 - ema_alpha) * return_ema
                    )
                    info["train/stats.return_ema"] = return_ema

                if "train/stats.return" in info:
                    current_return = float(info["train/stats.return"])
                    if current_return > max_return:
                        max_return = current_return
                        try:
                            ckpt_path = os.path.join(run_dir, f"checkpoint_best_return_{max_return:.2f}.pt")
                            torch.save({"model_state_dict": policy.state_dict()}, ckpt_path)
                            info["best_return"] = max_return
                            info["best_return_checkpoint_path"] = ckpt_path
                            logging.info("Saved best-return checkpoint to %s", ckpt_path)
                            if last_best_return_ckpt_path is not None and last_best_return_ckpt_path != ckpt_path:
                                try:
                                    if os.path.exists(last_best_return_ckpt_path):
                                        os.remove(last_best_return_ckpt_path)
                                except OSError:
                                    pass
                            last_best_return_ckpt_path = ckpt_path
                        except AttributeError:
                            logging.warning("Policy %s does not implement `.state_dict()`", policy)

                if eval_interval > 0 and (i + 1) % eval_interval == 0:
                    info.update(
                        _evaluate(
                            env,
                            policy,
                            base_env,
                            cfg,
                            run_dir=run.dir if run is not None else None,
                            step_frames=collector._frames,
                        )
                    )

                if save_interval > 0 and (i + 1) % save_interval == 0:
                    ckpt_path = os.path.join(run_dir, f"checkpoint_step_{collector._frames}.pt")
                    torch.save({"model_state_dict": policy.state_dict()}, ckpt_path)
                    info["checkpoint_path"] = ckpt_path

                if run is not None:
                    run.log(info)

                info_str = f"frames={collector._frames}"
                if "train/stats.return" in info:
                    info_str += f" return={info['train/stats.return']:.2f}"
                if "train/stats.success" in info:
                    info_str += f" success={info['train/stats.success']:.2f}"
                if "eval/success_rate" in info:
                    info_str += f" eval_success={info['eval/success_rate']:.2f}"
                pbar.set_description(info_str)

            run_dir = run.dir if run is not None else os.getcwd()
            final_eval_enabled = bool(cfg.get("final_eval", True))
            if run is not None and final_eval_enabled:
                try:
                    logging.info("Final Eval at %s steps.", collector._frames)
                    final_info = {"env_frames": collector._frames}
                    final_info.update(
                        _evaluate(
                            env,
                            policy,
                            base_env,
                            cfg,
                            run_dir=run_dir,
                            step_frames=collector._frames,
                        )
                    )
                    run.log(final_info)
                except Exception as exc:
                    logging.warning("Final evaluation skipped: %s", exc)

            final_ckpt = os.path.join(run_dir, "checkpoint_final.pt")
            torch.save({"model_state_dict": policy.state_dict()}, final_ckpt)
            if run is not None:
                try:
                    import wandb

                    artifact_name = f"{cfg.task.name}-{cfg.algo.name.lower()}"
                    model_artifact = wandb.Artifact(
                        artifact_name,
                        type="model",
                        description=artifact_name,
                        metadata=OmegaConf.to_container(cfg, resolve=True),
                    )
                    model_artifact.add_file(final_ckpt)
                    wandb.save(final_ckpt)
                    run.log_artifact(model_artifact)
                except Exception as exc:
                    logging.warning("Failed to upload checkpoint artifact to wandb: %s", exc)
            print(f"Training done. Final checkpoint: {final_ckpt}")

    finally:
        if run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        if simulation_app is not None:
            simulation_app.close()


def _ensure_default_task_override() -> None:
    has_task_override = any(
        arg == "task"
        or arg.startswith("task=")
        or arg.startswith("+task=")
        or arg.startswith("++task=")
        for arg in sys.argv[1:]
    )
    if not has_task_override:
        sys.argv.append("task=forest_lc_gate")


if __name__ == "__main__":
    _ensure_default_task_override()
    main()
