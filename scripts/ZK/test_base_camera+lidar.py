import logging
import os
# 🌟 必须放在 import torch 和其他库的最前面！
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 固定当前脚本只使用物理 0 号 GPU，避免继承到外部的多卡/错卡配置。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import importlib
import math
import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

class DualStreamBackbone(torch.nn.Module):
    """
    轻量三输入 backbone：state + LiDAR-KU + camera risk。
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
        if self.camera_risk_fusion_mode not in {"gate", "concat"}:
            raise ValueError(
                f"camera_risk_fusion_mode must be 'gate' or 'concat', got {camera_risk_fusion_mode}"
            )
        self.ku_value_max = float(ku_value_max)
        if self.ku_value_max <= 0.0:
            raise ValueError(f"ku_value_max must be positive, got {self.ku_value_max}")
        self.ku_unknown_value = self.ku_value_max

        self.ku_h = 40
        self.ku_w = 80

        # LiDAR-KU 编码分支: lidar_ku_norm [B,1,40,80] -> lidar_feat:[B,128]
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
        """Expose representative parameters for optimizer/gradient debug checks."""
        return {
            "ku_encoder": self.ku_encoder[0].weight,
            "camera_risk": self.camera_risk_encoder[0].weight,
            "fusion": self.fusion_mlp[0].weight,
        }

    def forward(self, obs):
        # obs: [*, state_dim + lidar_ku(3200) + camera_risk(camera_risk_dim)]
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
            raise ValueError(
                "Depth camera forward axis is degenerate with both default and fallback up axes."
            )
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
        raise RuntimeError(
            "Failed to import Omniverse USD modules while reading depth camera geometry."
        ) from exc

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
    horizontal_aperture_offset = camera.GetHorizontalApertureOffsetAttr().Get()
    vertical_aperture_offset = camera.GetVerticalApertureOffsetAttr().Get()
    clipping_range = camera.GetClippingRangeAttr().Get()

    if focal_length is None or horizontal_aperture is None:
        raise RuntimeError(
            f"Depth camera prim {depth_prim_path} is missing focalLength/horizontalAperture attributes: "
            f"focal={focal_length}, horizontal_aperture={horizontal_aperture}"
        )

    if vertical_aperture is None:
        vertical_aperture = float(horizontal_aperture) * float(base_env.depth_h) / float(max(1, base_env.depth_w))
        logging.warning(
            "Depth camera prim %s has no verticalAperture; inferred %.6f mm from aspect ratio.",
            depth_prim_path,
            float(vertical_aperture),
        )
    if horizontal_aperture_offset is None:
        horizontal_aperture_offset = 0.0
    if vertical_aperture_offset is None:
        vertical_aperture_offset = 0.0

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

    fallback_cam_in_base_gf = omni.usd.get_local_transform_matrix(prim)
    if not isinstance(fallback_cam_in_base_gf, Gf.Matrix4d):
        fallback_cam_in_base_gf = Gf.Matrix4d(fallback_cam_in_base_gf)
    fallback_cam_pos_lidar_np, fallback_cam_rot_lidar_from_camera_np = _extract_pose_from_matrix(
        fallback_cam_in_base_gf
    )
    fallback_cam_rot_lidar_from_camera_np = _orthonormalize_rotation(fallback_cam_rot_lidar_from_camera_np)

    cam_pos_lidar_np = fallback_cam_pos_lidar_np
    cam_rot_lidar_from_camera_np = fallback_cam_rot_lidar_from_camera_np
    relative_transform_source = "omni.usd.get_local_transform_matrix"
    transform_consistency_pos_err = 0.0
    transform_consistency_rot_deg = 0.0
    try:
        from isaacsim.core.includes.pose import getRelativeTransform  # type: ignore

        cam_in_base_gf = getRelativeTransform(
            stage,
            None,
            prim.GetPath(),
            base_prim.GetPath(),
        )
        if not isinstance(cam_in_base_gf, Gf.Matrix4d):
            cam_in_base_gf = Gf.Matrix4d(cam_in_base_gf)
        official_cam_pos_lidar_np, official_cam_rot_lidar_from_camera_np = _extract_pose_from_matrix(cam_in_base_gf)
        official_cam_rot_lidar_from_camera_np = _orthonormalize_rotation(official_cam_rot_lidar_from_camera_np)
        transform_consistency_pos_err = float(
            np.linalg.norm(official_cam_pos_lidar_np - fallback_cam_pos_lidar_np)
        )
        relative_rot = official_cam_rot_lidar_from_camera_np.T @ fallback_cam_rot_lidar_from_camera_np
        trace_val = float(np.trace(relative_rot))
        cos_angle = max(-1.0, min(1.0, 0.5 * (trace_val - 1.0)))
        transform_consistency_rot_deg = float(np.degrees(np.arccos(cos_angle)))
        if transform_consistency_pos_err > 1e-5 or transform_consistency_rot_deg > 1e-3:
            raise RuntimeError(
                "Depth camera relative transform mismatch between getRelativeTransform and "
                "omni.usd.get_local_transform_matrix fallback. "
                f"prim={depth_prim_path}, pos_err={transform_consistency_pos_err:.6e} m, "
                f"rot_err_deg={transform_consistency_rot_deg:.6e}. "
                "This indicates an inconsistent transform interpretation and training is aborted to avoid "
                "using incorrect camera extrinsics."
            )
        cam_pos_lidar_np = official_cam_pos_lidar_np
        cam_rot_lidar_from_camera_np = official_cam_rot_lidar_from_camera_np
        relative_transform_source = "isaacsim.core.includes.pose.getRelativeTransform"
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

    depth_cam_pos_cfg = base_env.cfg.task.get("depth_camera_pos", [0.22, 0.0, 0.18])
    depth_cam_target_cfg = base_env.cfg.task.get("depth_camera_target", [2.0, 0.0, 0.18])
    expected_cam_pos = np.array(depth_cam_pos_cfg, dtype=np.float64)
    cam_pos_error = float(np.linalg.norm(cam_pos_lidar_np.astype(np.float64) - expected_cam_pos))
    if cam_pos_error > 1e-4:
        logging.warning(
            "Depth camera position differs from YAML config — using stage-derived position. "
            f"prim={depth_prim_path}, position_error={cam_pos_error:.6e}, "
            f"camera_pos_lidar={tuple(float(v) for v in cam_pos_lidar_np.tolist())}, "
            f"expected_pos_lidar={tuple(float(v) for v in expected_cam_pos.tolist())}. "
            "This suggests depth_camera_pos in the YAML is not applied when spawning cameras; "
            "the actual stage geometry will be used."
        )
    expected_optical_axis = np.array(depth_cam_target_cfg, dtype=np.float64) - np.array(depth_cam_pos_cfg, dtype=np.float64)
    expected_axis_norm = np.linalg.norm(expected_optical_axis)
    if expected_axis_norm <= 1e-9:
        raise RuntimeError(
            f"Invalid depth camera config: target and position coincide for prim {depth_prim_path}."
        )
    expected_optical_axis = expected_optical_axis / expected_axis_norm
    expected_rot_lidar_from_camera_np = _expected_row_camera_rotation_from_view(
        np.array(depth_cam_pos_cfg, dtype=np.float64),
        np.array(depth_cam_target_cfg, dtype=np.float64),
    )
    (
        cam_rot_lidar_from_camera_np,
        rotation_convention,
        row_err_deg,
        col_err_deg,
    ) = _select_camera_rotation_convention(
        cam_rot_lidar_from_camera_np.astype(np.float64),
        expected_rot_lidar_from_camera_np,
    )

    optical_axis_camera = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    optical_axis_row = optical_axis_camera @ cam_rot_lidar_from_camera_np
    optical_axis_row = optical_axis_row / max(np.linalg.norm(optical_axis_row), 1e-12)
    optical_axis_col = cam_rot_lidar_from_camera_np @ optical_axis_camera
    optical_axis_col = optical_axis_col / max(np.linalg.norm(optical_axis_col), 1e-12)
    row_axis_alignment = float(np.dot(optical_axis_row, expected_optical_axis))
    col_axis_alignment = float(np.dot(optical_axis_col, expected_optical_axis))
    optical_axis_lidar = optical_axis_row

    cam_pos_lidar = torch.tensor(cam_pos_lidar_np, dtype=torch.float32)
    cam_rot_lidar_from_camera = torch.tensor(cam_rot_lidar_from_camera_np, dtype=torch.float32)
    axis_alignment = float(np.dot(optical_axis_lidar, expected_optical_axis))

    if axis_alignment < 0.99:
        logging.warning(
            "Depth camera optical axis differs from YAML config — using stage-derived axis. "
            f"prim={depth_prim_path}, alignment={axis_alignment:.6f}, "
            f"row_alignment={row_axis_alignment:.6f}, column_alignment={col_axis_alignment:.6f}, "
            f"row_err_deg={row_err_deg:.6f}, col_err_deg={col_err_deg:.6f}, "
            f"rotation_convention={rotation_convention}, "
            f"optical_axis_lidar={tuple(float(v) for v in optical_axis_lidar.tolist())}, "
            f"expected_axis_lidar={tuple(float(v) for v in expected_optical_axis.tolist())}. "
            "This may indicate depth_camera_target is not applied as expected, "
            "or a camera-axis convention mismatch."
        )

    near_clip = None
    far_clip = None
    if clipping_range is not None and len(clipping_range) >= 2:
        near_clip = float(clipping_range[0])
        far_clip = float(clipping_range[1])

    logging.info(
        "Depth camera geometry from prim %s relative to %s via %s | pos=(%.6f, %.6f, %.6f)m "
        "optical_axis_lidar=(%.6f, %.6f, %.6f) align=%.6f "
        "row_align=%.6f column_align=%.6f rotation_convention=%s row_err_deg=%.6f col_err_deg=%.6f pos_err=%.6e "
        "xform_consistency_pos_err=%.6e xform_consistency_rot_deg=%.6e "
        "focal=%.6fmm, h_ap=%.6fmm, v_ap=%.6fmm, h_off=%.6f, v_off=%.6f, "
        "fx=%.6fpx, fy=%.6fpx, cx=%.6fpx, cy=%.6fpx, clip=(%s, %s)",
        depth_prim_path,
        base_prim_path,
        relative_transform_source,
        float(cam_pos_lidar[0].item()),
        float(cam_pos_lidar[1].item()),
        float(cam_pos_lidar[2].item()),
        float(optical_axis_lidar[0]),
        float(optical_axis_lidar[1]),
        float(optical_axis_lidar[2]),
        axis_alignment,
        row_axis_alignment,
        col_axis_alignment,
        rotation_convention,
        row_err_deg,
        col_err_deg,
        cam_pos_error,
        transform_consistency_pos_err,
        transform_consistency_rot_deg,
        float(focal_length),
        float(horizontal_aperture),
        float(vertical_aperture),
        float(horizontal_aperture_offset),
        float(vertical_aperture_offset),
        fx_px,
        fy_px,
        cx_px,
        cy_px,
        "None" if near_clip is None else f"{near_clip:.6f}",
        "None" if far_clip is None else f"{far_clip:.6f}",
    )

    return {
        "prim_path": depth_prim_path,
        "relative_transform_source": relative_transform_source,
        "focal_length_mm": float(focal_length),
        "horizontal_aperture_mm": float(horizontal_aperture),
        "vertical_aperture_mm": float(vertical_aperture),
        "horizontal_aperture_offset": float(horizontal_aperture_offset),
        "vertical_aperture_offset": float(vertical_aperture_offset),
        "intrinsic_matrix": intrinsic_matrix,
        "position_lidar_m": tuple(float(v.item()) for v in cam_pos_lidar),
        "rot_lidar_from_camera": cam_rot_lidar_from_camera.tolist(),
        "optical_axis_lidar": tuple(float(v) for v in optical_axis_lidar.tolist()),
        "rotation_convention": rotation_convention,
        "row_axis_alignment": row_axis_alignment,
        "column_axis_alignment": col_axis_alignment,
        "near_clip": near_clip,
        "far_clip": far_clip,
    }


@hydra.main(version_base=None, config_path="", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # 双保险：脚本内统一固定到 0 号卡。
    cfg.sim.device = "cuda:0"
    cfg.sim.active_gpu = 0
    cfg.sim.physics_gpu = 0

    # 按规模自动控制渲染，避免大批量并行时触发 syntheticdata 崩溃（exit 139）
    num_envs = int(cfg.env.num_envs)
    user_viewport = bool(cfg.sim.get("enable_viewport", False))
    user_replicator = bool(cfg.sim.get("enable_replicator", True))
    needs_depth_camera = bool(
        cfg.task.get("use_depth_ku_observation", False)
        or cfg.task.get("use_camera_risk_observation", False)
    )
    max_replicator_envs = int(cfg.get("max_camera_replicator_envs", 64))

    # 无显示设备时强制 headless，避免 Vulkan swapchain / present 初始化失败导致崩溃。
    display_env = os.environ.get("DISPLAY", "").strip()
    has_display = len(display_env) > 0
    force_headless_no_display = bool(cfg.get("force_headless_no_display", True))
    if force_headless_no_display and (not has_display):
        logging.warning(
            "DISPLAY is not set; forcing headless mode and disabling viewport. "
            "Replicator remains enabled when depth camera observations are requested."
        )
        cfg.headless = True
        cfg.sim.enable_viewport = False
        if "task" in cfg and cfg.task.get("show_depth_preview_window", False):
            cfg.task.show_depth_preview_window = False

    # 仅在需要评估可视化且并行规模可控时开启
    allow_render = (not cfg.headless) and (num_envs <= 200)
    allow_replicator = num_envs <= max_replicator_envs if needs_depth_camera else (
        (cfg.get("eval_interval", -1) > 0) and (num_envs <= 200)
    )

    # cfg.sim.enable_viewport = user_viewport or allow_render
    cfg.sim.enable_viewport = (not cfg.headless) and (user_viewport or allow_render) and has_display

    cfg.sim.enable_replicator = user_replicator and allow_replicator
    if needs_depth_camera and not cfg.sim.enable_replicator:
        raise RuntimeError(
            "Camera/depth observations are enabled, but cfg.sim.enable_replicator is false. "
            f"num_envs={num_envs}, max_camera_replicator_envs={max_replicator_envs}. "
            "Lower env.num_envs or increase max_camera_replicator_envs after confirming GPU memory."
        )

    # 安全阈值：高并行下强制关闭复制器与视口
    if num_envs > 200 and not needs_depth_camera:
        cfg.sim.enable_viewport = False
        cfg.sim.enable_replicator = False

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    task_name = str(cfg.task.name)
    try:
        importlib.import_module(f"omni_drones.envs.single.{task_name.lower()}")
    except ModuleNotFoundError:
        pass

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    #将任务观测中的雷达数据和状态数据进行拼接，并且可以选择性地将它们展平为一维向量，适配不同算法的输入需求。
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
        and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
    ):
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1)
        )
    #将离散化的动作空间转换为连续空间，适配不同算法的输出需求。支持从 MultiDiscrete 或 Discrete 两种常见离散空间类型转换。
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    def _load_policy_checkpoint_compatible(model, ckpt_path, map_location):
        raw_state = torch.load(ckpt_path, map_location=map_location)
        if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
            raw_state = raw_state["state_dict"]
        if not isinstance(raw_state, dict):
            raise TypeError(f"Unsupported checkpoint format at {ckpt_path}: {type(raw_state)}")

        model_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        skipped_missing = []

        for key, value in raw_state.items():
            if key not in model_state:
                skipped_missing.append(key)
                continue
            if model_state[key].shape != value.shape:
                skipped_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = value

        load_info = model.load_state_dict(filtered_state, strict=False)
        logging.info(
            "Compatible checkpoint load: loaded=%d, missing_in_ckpt=%d, unexpected_in_ckpt=%d, shape_mismatch=%d",
            len(filtered_state),
            len(load_info.missing_keys),
            len(skipped_missing),
            len(skipped_mismatch),
        )
        if skipped_mismatch:
            preview = ", ".join(
                f"{k}: ckpt{src_shape}->model{dst_shape}" for k, src_shape, dst_shape in skipped_mismatch[:5]
            )
            logging.warning(f"Checkpoint shape mismatches (first 5): {preview}")
        if skipped_missing:
            logging.warning(f"Checkpoint unexpected keys skipped (first 5): {skipped_missing[:5]}")
        if load_info.missing_keys:
            logging.warning(f"Model keys not loaded from checkpoint (first 5): {load_info.missing_keys[:5]}")

        return load_info

    # Keep optimizer hyper-parameters from PPO implementation before replacing modules.
    actor_lr = policy.actor_opt.param_groups[0]["lr"]
    critic_lr = policy.critic_opt.param_groups[0]["lr"]

    obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
    lidar_dim = 3200
    ku_value_max = float(cfg.task.get("ku_value_max", 20.0))

    # Compute expected camera_risk_dim from config, supporting both:
    # - forest_lc:       camera_risk_num_bins (backward compat, 1 row × N cols)
    # - forest_lc_gate:  camera_risk_num_rows × camera_risk_num_cols (2D grid)
    _num_rows = int(cfg.task.get("camera_risk_num_rows", 0))
    _num_cols = int(cfg.task.get("camera_risk_num_cols", 0))
    _num_bins = int(cfg.task.get("camera_risk_num_bins", 3))
    if _num_rows <= 0 and _num_cols <= 0:
        _grid_rows, _grid_cols = 1, _num_bins
    elif _num_rows <= 0:
        _grid_rows, _grid_cols = 1, _num_cols
    elif _num_cols <= 0:
        _grid_rows, _grid_cols = _num_rows, 1
    else:
        _grid_rows, _grid_cols = _num_rows, _num_cols
    _features_per_sector = int(cfg.task.get("camera_risk_features_per_bin", 4))
    expected_camera_risk_dim = _grid_rows * _grid_cols * _features_per_sector
    if bool(cfg.task.get("camera_risk_add_stale_ratio", True)):
        expected_camera_risk_dim += 1

    # Derive state_dim from observation layout, NOT from hardcoded defaults.
    state_dim = int(obs_dim - lidar_dim - expected_camera_risk_dim)
    camera_risk_dim = int(obs_dim - state_dim - lidar_dim)

    if state_dim <= 0 or camera_risk_dim <= 0:
        task_name_cfg = str(cfg.task.get("name", "<unknown>"))
        raise ValueError(
            f"Invalid observation layout: obs_dim={obs_dim}, state_dim={state_dim}, "
            f"lidar_dim={lidar_dim}, camera_risk_dim={camera_risk_dim}, task={task_name_cfg}. "
            "This script expects observation=[state, lidar_ku(3200), camera_risk]. "
            "Ensure use_camera_risk_observation=true and use_depth_ku_observation=false."
        )

    logging.info(
        "Observation layout verified: state_dim=%d, lidar_dim=%d, camera_risk_dim=%d (expected=%d from %dx%d grid).",
        state_dim, lidar_dim, camera_risk_dim, expected_camera_risk_dim, _grid_rows, _grid_cols,
    )
    expected_feature_dim = 128
    camera_risk_gate_alpha = float(cfg.task.get("camera_risk_gate_alpha", 0.2))
    camera_risk_fusion_mode = str(cfg.task.get("camera_risk_fusion_mode", "gate"))
    _read_depth_camera_geometry_from_stage(base_env)

    actor_backbone = DualStreamBackbone(
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        camera_risk_dim=camera_risk_dim,
        ku_value_max=ku_value_max,
        output_dim=expected_feature_dim,
        camera_risk_gate_alpha=camera_risk_gate_alpha,
        camera_risk_fusion_mode=camera_risk_fusion_mode,
    ).to(base_env.device)

    critic_backbone = DualStreamBackbone(
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        camera_risk_dim=camera_risk_dim,
        ku_value_max=ku_value_max,
        output_dim=expected_feature_dim,
        camera_risk_gate_alpha=camera_risk_gate_alpha,
        camera_risk_fusion_mode=camera_risk_fusion_mode,
    ).to(base_env.device)

    print("type(policy.actor) =", type(policy.actor))
    print("type(policy.actor.module) =", type(policy.actor.module))
    print("policy.actor.module =", policy.actor.module)
    print("type(policy.actor.module[0]) =", type(policy.actor.module[0]))
    print("policy.actor.module[0] =", policy.actor.module[0])

    if hasattr(policy.actor.module[0], "module"):
        print("type(policy.actor.module[0].module) =", type(policy.actor.module[0].module))
        print("policy.actor.module[0].module =", policy.actor.module[0].module)

    print("type(policy.critic.module) =", type(policy.critic.module))
    print("policy.critic.module =", policy.critic.module)

    # 按模块类型安全替换，兼容 ppo / ppo_priv_critic 两种结构。
    actor_replaced = False
    critic_replaced = False

    # actor 常见结构: ProbabilisticActor.module -> TensorDictModule(module=nn.Sequential(...))
    if hasattr(policy.actor, "module") and hasattr(policy.actor.module, "module"):
        actor_core = policy.actor.module.module
        if isinstance(actor_core, torch.nn.Sequential) and len(actor_core) > 0:
            actor_core[0] = actor_backbone
            actor_replaced = True

    # 备选结构: ProbabilisticActor.module[0].module -> nn.Sequential(...)
    if (not actor_replaced) and hasattr(policy.actor, "module") and hasattr(policy.actor.module, "__getitem__"):
        try:
            actor_td_module = policy.actor.module[0]
            if hasattr(actor_td_module, "module") and isinstance(actor_td_module.module, torch.nn.Sequential):
                actor_td_module.module[0] = actor_backbone
                actor_replaced = True
        except Exception:
            pass

    # critic 结构 1: TensorDictModule(module=nn.Sequential(...))
    if hasattr(policy.critic, "module") and isinstance(policy.critic.module, torch.nn.Sequential):
        policy.critic.module[0] = critic_backbone
        critic_replaced = True

    # critic 结构 2 (ppo_priv_critic): TensorDictSequential([TensorDictModule(...), ...])
    # 注意这里必须替换第一个 TensorDictModule 的 .module，不能直接替换 module[0]。
    if (not critic_replaced) and hasattr(policy.critic, "module") and hasattr(policy.critic.module, "__getitem__"):
        try:
            critic_td_module = policy.critic.module[0]
            if hasattr(critic_td_module, "module"):
                critic_td_module.module = critic_backbone
                critic_replaced = True
        except Exception:
            pass

    if not actor_replaced or not critic_replaced:
        raise RuntimeError(
            f"Backbone injection failed: actor_replaced={actor_replaced}, "
            f"critic_replaced={critic_replaced}. "
            f"actor.module={type(policy.actor.module)}, critic.module={type(policy.critic.module)}"
        )
# ================= 🌟 修复 1：补充正交初始化 =================
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, torch.nn.Conv2d):
            # Kaiming normal for LeakyReLU (negative_slope=0.1)
            torch.nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    actor_backbone.apply(init_weights)
    critic_backbone.apply(init_weights)

    if hasattr(policy.actor.module, "module"):
        print("actor after replace:", policy.actor.module.module)
    elif hasattr(policy.actor.module, "__getitem__") and hasattr(policy.actor.module[0], "module"):
        print("actor after replace:", policy.actor.module[0].module)
    print("critic after replace:", policy.critic.module)
    print(
        f"✅ 成功注入 camera-risk gating 骨干网络！state_dim={state_dim}, "
        f"lidar_dim={lidar_dim}, camera_risk_dim={camera_risk_dim}, "
        f"fusion_mode={camera_risk_fusion_mode}, gate_alpha={camera_risk_gate_alpha}, "
        f"output_dim={expected_feature_dim}"
    )
    # ================= 🌟 致命 Bug 修复 =================
    # 重新绑定优化器，让它们追踪全新的双流网络参数。
    policy.actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=actor_lr)
    policy.critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)

    def _optimizer_has_param(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> bool:
        target_id = id(param)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) == target_id:
                    return True
        return False

    actor_probe = actor_backbone.get_probe_params()
    critic_probe = critic_backbone.get_probe_params()
    actor_param_bound = _optimizer_has_param(policy.actor_opt, actor_probe["ku_encoder"])
    critic_param_bound = _optimizer_has_param(policy.critic_opt, critic_probe["ku_encoder"])
    print(
        f"[debug] optimizer bind check | actor_backbone={actor_param_bound}, critic_backbone={critic_param_bound}, "
        f"actor_lr={actor_lr:.2e}, critic_lr={critic_lr:.2e}"
    )
    # ===============================================================

    # 训练初始化分支：支持从头训练或从 goodpt checkpoint 继续训练
    init_mode = str(cfg.get("init_mode", "scratch")).lower()
    goodpt_path_cfg = str(cfg.get("goodpt_path", "")).strip()
    if init_mode == "scratch":
        logging.info("Training init_mode=scratch: start from random initialization.")
    elif init_mode == "goodpt":
        if not goodpt_path_cfg:
            raise ValueError("init_mode=goodpt but goodpt_path is empty.")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_path = goodpt_path_cfg
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(script_dir, candidate_path)
        candidate_path = os.path.abspath(candidate_path)

        resolved_ckpt_path = candidate_path
        if os.path.isdir(candidate_path):
            final_ckpt = os.path.join(candidate_path, "checkpoint_final.pt")
            if os.path.exists(final_ckpt):
                resolved_ckpt_path = final_ckpt
            else:
                candidates = [
                    os.path.join(candidate_path, f)
                    for f in os.listdir(candidate_path)
                    if f.endswith(".pt") and os.path.isfile(os.path.join(candidate_path, f))
                ]
                if not candidates:
                    raise FileNotFoundError(
                        f"No .pt checkpoint files found in goodpt directory: {candidate_path}"
                    )
                resolved_ckpt_path = max(candidates, key=os.path.getmtime)
                logging.info(
                    f"goodpt_path is a directory, auto-selected latest checkpoint: {resolved_ckpt_path}"
                )

        if not os.path.exists(resolved_ckpt_path):
            raise FileNotFoundError(f"goodpt checkpoint not found: {resolved_ckpt_path}")

        _load_policy_checkpoint_compatible(policy, resolved_ckpt_path, map_location=base_env.device)
        logging.info(f"Training init_mode=goodpt: loaded checkpoint from {resolved_ckpt_path}")
    else:
        raise ValueError(
            f"Unsupported init_mode: {init_mode}. Expected one of ['scratch', 'goodpt']."
        )
    #计算每收集多少帧（步数）的数据，就执行一次神经网络的更新（训练）。
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    #强行把“总训练步数（total_frames）”砍掉一点尾数，使其变成“每次训练数据量（frames_per_batch）”的绝对整数倍，防止后面的数据维度不匹配
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    recommended_eval_interval = int(cfg.get("recommended_eval_interval", 20))
    if eval_interval <= 0:
        logging.warning(
            "eval_interval<=0: training will not run periodic formal evaluation. "
            f"Recommended eval_interval={recommended_eval_interval} for consistent model selection by eval/success_rate."
        )
    save_interval = cfg.get("save_interval", -1)
    max_return = -float("inf")
    last_best_return_ckpt_path = None
    last_best_success_ckpt_path = None
    last_ckpt_path = None

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE,
        export_topk: bool=False,
        topk: int=10,
    ):
        def _extract_env_obstacle_points(max_points: int = 20000):
            """Read terrain mesh points directly from USD stage as environment obstacle geometry."""
            try:
                import omni.usd  # type: ignore
                from pxr import UsdGeom
            except Exception:
                return None

            try:
                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    return None

                mesh_prims = []
                for prim in stage.Traverse():
                    if not prim.IsValid():
                        continue
                    path_str = str(prim.GetPath())
                    if not path_str.startswith("/World/ground"):
                        continue
                    if prim.IsA(UsdGeom.Mesh):
                        mesh_prims.append(prim)

                if len(mesh_prims) == 0:
                    return None

                all_pts = []
                for prim in mesh_prims:
                    mesh = UsdGeom.Mesh(prim)
                    pts = mesh.GetPointsAttr().Get()
                    if pts is None or len(pts) == 0:
                        continue

                    pts_np = np.asarray(pts, dtype=np.float32)
                    xform = UsdGeom.Xformable(prim)
                    mat = np.array(xform.ComputeLocalToWorldTransform(0.0), dtype=np.float64)
                    pts_h = np.concatenate([pts_np.astype(np.float64), np.ones((pts_np.shape[0], 1), dtype=np.float64)], axis=1)
                    pts_w = (pts_h @ mat.T)[:, :3].astype(np.float32)
                    all_pts.append(pts_w)

                if len(all_pts) == 0:
                    return None

                pts_world = np.concatenate(all_pts, axis=0)
                finite = np.isfinite(pts_world).all(axis=1)
                pts_world = pts_world[finite]

                # Keep above-ground vertices that mostly correspond to obstacle structures.
                pts_world = pts_world[pts_world[:, 2] > 0.15]
                if pts_world.shape[0] == 0:
                    return None

                if pts_world.shape[0] > max_points:
                    sel = np.random.choice(pts_world.shape[0], size=max_points, replace=False)
                    pts_world = pts_world[sel]

                return pts_world.astype(np.float32)
            except Exception:
                return None
        # ================= 🌟 显存急救：清空训练残留 =================
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # ==========================================================
        # ================= 🛡️ 智能防爆显存锁 =================
        should_render = (base_env.num_envs <= 200) and cfg.sim.enable_viewport
        should_record = should_render or cfg.sim.enable_replicator
        
        eval_video_interval = max(1, int(cfg.get("eval_video_interval", 1)))

        if should_record:
            base_env.enable_render(True)
            render_callback = RenderCallback(interval=eval_video_interval)
            print(f"🎥 [评估阶段] 环境数量较少，已开启视频渲染模式。采样间隔={eval_video_interval}")
        else:
            base_env.enable_render(False)
            render_callback = None
            print(
                f"⚡ [评估阶段] 已关闭视频渲染，仅进行数值评估。"
                f"(num_envs={base_env.num_envs}, viewport={cfg.sim.enable_viewport}, replicator={cfg.sim.enable_replicator})"
            )
        # ======================================================

        base_env.eval()
        env.eval()
        env.set_seed(seed)
        
        
        # with set_exploration_type(exploration_type):
        #     trajs = env.rollout(
        #         max_steps=2000,
        #         policy=policy,
        #         callback=render_callback, 
        #         auto_reset=True,
        #         break_when_any_done=False,
        #         return_contiguous=False,
        #     )
        
        # if should_render:
        #     base_env.enable_render(not cfg.headless)
        # env.reset()

        # done = trajs.get(("next", "done"))
        # first_done = torch.argmax(done.long(), dim=1).cpu()

        # def take_first_episode(tensor: torch.Tensor):
        #     indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        #     return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        # traj_stats = {
        #     k: take_first_episode(v)
        #     for k, v in trajs[("next", "stats")].cpu().items()
        # }

        # info = {
        #     "eval/stats." + k: torch.mean(v.float()).item()
        #     for k, v in traj_stats.items()
        # }
        # 🌟 新增导入 TorchRL 的状态推进神器
        from torchrl.envs.utils import step_mdp

        # ================= 🚀 终极 OOM 杀手：低内存评估循环 =================
        with set_exploration_type(exploration_type):
            td = env.reset()
            
            # 记录哪些无人机已经到达终点或撞毁
            has_finished = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            
            # 动态收集 stats，兼容不同任务/奖励配置新增的字段（如 reward_forward）
            first_episode_stats = {}

            # 记录整批无人机首回合的逐步轨迹与奖励因子，后续用于筛选 Top-K 导出
            max_steps = int(base_env.max_episode_length)
            num_envs_eval = int(base_env.num_envs)
            pos_buffer = np.full((max_steps, num_envs_eval, 3), np.nan, dtype=np.float32)
            done_buffer = np.zeros((max_steps, num_envs_eval), dtype=bool)
            reward_total_buffer = np.full((max_steps, num_envs_eval), np.nan, dtype=np.float32)
            speed_buffer = np.full((max_steps, num_envs_eval), np.nan, dtype=np.float32)
            reward_component_buffers = {}
            reward_component_keys = []
            prev_component_stats = {}

            # 终止原因与障碍物点云导出
            term_reason_code = torch.full(
                (num_envs_eval,),
                fill_value=0,
                dtype=torch.int32,
                device=base_env.device,
            )
            # 0=unknown, 1=goal, 2=timeout, 3=nan, 4=collision,
            # 5=z_low, 6=z_high, 7=overspeed, 8=out_of_bounds, 9=flip
            term_reason_names = [
                "unknown",
                "goal_reached",
                "timeout",
                "state_nan",
                "collision",
                "z_too_low",
                "z_too_high",
                "overspeed",
                "out_of_bounds",
                "flip",
            ]

            obstacle_points_count = int(cfg.get("traj_obstacle_points", 1024))
            obstacle_points_count = max(128, obstacle_points_count)
            obstacle_points = np.full((num_envs_eval, obstacle_points_count, 3), np.nan, dtype=np.float32)
            obstacle_captured = np.zeros((num_envs_eval,), dtype=bool)
            obstacle_env_points = _extract_env_obstacle_points(max_points=int(cfg.get("traj_obstacle_mesh_points", 20000)))

            # 手动一步步推演，彻底抛弃保存全量历史数据
            for step in range(base_env.max_episode_length):
                td = policy(td)
                td = env.step(td)

                if render_callback is not None:
                    try:
                        render_callback(base_env)
                    except Exception as e:
                        print(f"[!] 渲染回调失败(step={step}): {e}")
                        render_callback = None
                
                # 检查这一步的死亡/到达状态
                done = td.get(("next", "done")).squeeze(-1)
                if done.ndim > 1:
                    done = done.squeeze(-1)
                terminated = td.get(("next", "terminated")).squeeze(-1)
                truncated = td.get(("next", "truncated")).squeeze(-1)
                if terminated.ndim > 1:
                    terminated = terminated.squeeze(-1)
                if truncated.ndim > 1:
                    truncated = truncated.squeeze(-1)

                active_mask = ~has_finished
                active_mask_cpu = active_mask.detach().cpu().numpy()

                # 采样并缓存每个环境的障碍物点云（首帧一次）
                if (~obstacle_captured).any():
                    try:
                        ray_hits_w = base_env.lidar.data.ray_hits_w.reshape(num_envs_eval, -1, 3)
                        ray_hits_np = ray_hits_w.detach().cpu().numpy()
                        todo_idx = np.where(~obstacle_captured)[0]
                        for env_i in todo_idx:
                            pts = ray_hits_np[env_i]
                            finite = np.isfinite(pts).all(axis=1)
                            pts = pts[finite]
                            if pts.shape[0] == 0:
                                continue
                            # 过滤地面，优先保留障碍物立体点
                            high_pts = pts[pts[:, 2] > 0.15]
                            if high_pts.shape[0] > 0:
                                pts = high_pts
                            n_pick = min(obstacle_points_count, pts.shape[0])
                            sel = np.random.choice(pts.shape[0], size=n_pick, replace=False)
                            obstacle_points[env_i, :n_pick, :] = pts[sel].astype(np.float32)
                            obstacle_captured[env_i] = True
                    except Exception:
                        pass

                # 记录这一步尚未结束无人机的位置、总奖励、完成标志
                pos_now = base_env.drone.pos[..., :3].squeeze(1)
                pos_buffer[step, active_mask_cpu, :] = pos_now[active_mask].detach().cpu().numpy()
                done_buffer[step] = done.detach().cpu().numpy()

                step_reward = td.get(("next", "agents", "reward")).squeeze(-1)
                if step_reward.ndim > 1:
                    step_reward = step_reward.squeeze(-1)
                reward_total_buffer[step, active_mask_cpu] = step_reward[active_mask].detach().cpu().numpy()

                # 计算每个环境本步终止原因代码（与环境判定逻辑对齐）
                pos_now = base_env.drone.pos[..., :3].squeeze(1)
                vel_now = base_env.drone.vel_w[..., :3].squeeze(1)
                v_norm_now = vel_now.norm(dim=-1)
                z_now = pos_now[..., 2]
                speed_buffer[step, active_mask_cpu] = v_norm_now[active_mask].detach().cpu().numpy().astype(np.float32)

                if hasattr(base_env, "lidar_scan"):
                    d_min = (base_env.lidar_range - base_env.lidar_scan).amin(dim=2).squeeze(-1)
                    if d_min.ndim > 1:
                        d_min = d_min.squeeze(-1)
                    is_collision_step = d_min < base_env.collision_dist
                else:
                    is_collision_step = torch.zeros_like(done, dtype=torch.bool)

                if base_env.reset_on_collision:
                    contact_force = base_env.drone.base_link.get_net_contact_forces()
                    contact_coll = (contact_force.norm(dim=-1) > base_env.collision_force_threshold).any(-1)
                else:
                    contact_coll = torch.zeros_like(done, dtype=torch.bool)

                out_of_bounds_step = (torch.abs(pos_now[..., 0]) > 20.0) | (torch.abs(pos_now[..., 1]) > 30.0)
                flip_step = base_env.flip_counter.squeeze(-1) >= base_env.flip_consecutive_steps
                reached_goal_step = ((base_env.target_pos.squeeze(1) - pos_now).norm(dim=-1) < base_env.goal_radius)
                nan_step = torch.isnan(base_env.drone_state).any(-1)

                reason_step = torch.zeros_like(term_reason_code)
                reason_step = torch.where(truncated, torch.full_like(reason_step, 2), reason_step)
                reason_step = torch.where(nan_step, torch.full_like(reason_step, 3), reason_step)
                reason_step = torch.where(is_collision_step | contact_coll, torch.full_like(reason_step, 4), reason_step)
                reason_step = torch.where(z_now < base_env.terminate_z_min, torch.full_like(reason_step, 5), reason_step)
                reason_step = torch.where(z_now > base_env.terminate_z_max, torch.full_like(reason_step, 6), reason_step)
                reason_step = torch.where(v_norm_now > base_env.terminate_v_norm, torch.full_like(reason_step, 7), reason_step)
                reason_step = torch.where(out_of_bounds_step, torch.full_like(reason_step, 8), reason_step)
                reason_step = torch.where(flip_step, torch.full_like(reason_step, 9), reason_step)
                # Keep success labels stable: if goal is reached in this terminal step,
                # mark as goal_reached even when other safety constraints are also true.
                reason_step = torch.where(reached_goal_step, torch.full_like(reason_step, 1), reason_step)

                # 用累计 stats 的增量恢复每一步奖励分量
                stats_td = td.get(("next", "stats"))
                if not reward_component_keys:
                    for k, v in stats_td.items():
                        if isinstance(k, str) and k.startswith("reward_"):
                            reward_component_keys.append(k)
                            reward_component_buffers[k] = np.full((max_steps, num_envs_eval), np.nan, dtype=np.float32)
                            prev_component_stats[k] = torch.zeros(num_envs_eval, device=base_env.device, dtype=v.dtype)

                for k in reward_component_keys:
                    curr = stats_td[k].squeeze(-1)
                    if curr.ndim > 1:
                        curr = curr.squeeze(-1)
                    delta = (curr - prev_component_stats[k]).detach().float()
                    delta_cpu = delta.cpu().numpy()
                    reward_component_buffers[k][step, active_mask_cpu] = delta_cpu[active_mask_cpu]
                    prev_component_stats[k] = torch.where(active_mask, curr, prev_component_stats[k])
                
                # 找到 "在这一步刚刚完成" 的无人机
                just_finished = done & (~has_finished)
                term_reason_code = torch.where(just_finished, reason_step, term_reason_code)
                
                if just_finished.any():
                    # 仅把这些刚刚跑完的无人机的 stats 抠出来存好
                    for k, v in td.get(("next", "stats")).items():
                        if k not in first_episode_stats:
                            first_episode_stats[k] = torch.zeros(
                                base_env.num_envs,
                                device=base_env.device,
                                dtype=v.dtype,
                            )
                        first_episode_stats[k][just_finished] = v.squeeze(-1)[just_finished]
                
                # 更新完成名单
                has_finished = has_finished | done
                
                # 🌟 核心防爆显存：推进状态，立刻把巨大的旧雷达数据扔进垃圾桶！
                td = step_mdp(td) 
                
                # 如果所有无人机都跑完了一次，提前下班！
                if has_finished.all():
                    break

        if should_render:
            base_env.enable_render(not cfg.headless)
        env.reset()

        # 对齐原来战报打印的字典格式
        traj_stats = {k: v.cpu() for k, v in first_episode_stats.items()}
        # =====================================================================

        info = {}
        if len(traj_stats) > 0:
            info = {
                "eval/stats." + k: torch.mean(v.float()).item()
                for k, v in traj_stats.items()
            }

        if export_topk and len(traj_stats) > 0:
            returns = traj_stats.get("return", torch.zeros(base_env.num_envs)).float().view(-1)
            success = traj_stats.get("success", torch.zeros_like(returns)).float().view(-1)

            success_idx = torch.where(success > 0.5)[0]
            fail_idx = torch.where(success <= 0.5)[0]

            if success_idx.numel() > 0:
                success_idx = success_idx[torch.argsort(returns[success_idx], descending=True)]
            if fail_idx.numel() > 0:
                fail_idx = fail_idx[torch.argsort(returns[fail_idx], descending=True)]

            selected_idx = torch.cat([success_idx, fail_idx], dim=0)[: max(1, int(topk))]

            if selected_idx.numel() > 0:
                sel = selected_idx.cpu().numpy().astype(np.int32)
                valid_mask = ~np.isnan(pos_buffer[..., 0])
                reason_sel = term_reason_code[selected_idx].cpu().numpy().astype(np.int32).reshape(-1)
                reason_idx = np.clip(reason_sel, 0, len(term_reason_names) - 1)
                reason_name_sel = np.asarray(term_reason_names, dtype=object)[reason_idx]
                sim_dt_export = float(cfg.sim.dt) * float(cfg.sim.substeps)

                export_payload = {
                    "env_ids": sel,
                    "success": success[selected_idx].cpu().numpy().astype(np.float32),
                    "returns": returns[selected_idx].cpu().numpy().astype(np.float32),
                    "sim_dt": np.asarray([sim_dt_export], dtype=np.float32),
                    "control_dt": np.asarray([sim_dt_export], dtype=np.float32),
                    "death_reason_code": reason_sel,
                    "death_reason_name": reason_name_sel,
                    "episode_len": valid_mask[:, sel].sum(axis=0).astype(np.int32),
                    "xyz": np.transpose(pos_buffer[:, sel, :], (1, 0, 2)).astype(np.float32),
                    "obstacle_points": obstacle_points[sel].astype(np.float32),
                    "valid": np.transpose(valid_mask[:, sel], (1, 0)),
                    "done": np.transpose(done_buffer[:, sel], (1, 0)),
                    "speed_mps": np.transpose(speed_buffer[:, sel], (1, 0)).astype(np.float32),
                    "reward_total": np.transpose(reward_total_buffer[:, sel], (1, 0)).astype(np.float32),
                    "reward_keys": np.asarray(reward_component_keys, dtype=object),
                }

                if obstacle_env_points is not None:
                    export_payload["obstacle_env_points"] = obstacle_env_points.astype(np.float32)

                for k in reward_component_keys:
                    export_payload[f"factor__{k}"] = np.transpose(reward_component_buffers[k][:, sel], (1, 0)).astype(np.float32)

                export_path = os.path.join(run.dir, f"eval_top{len(sel)}_traj_step_{collector._frames}.npz")
                np.savez_compressed(export_path, **export_payload)
                info["eval/topk_traj_path"] = export_path
                info["eval/topk_traj_count"] = int(len(sel))
                print(f"📦 [轨迹导出] Top-{len(sel)} 轨迹与奖励因子已保存: {export_path}")

        # ================= 🚀 霸气战报统计 (读取真实的 success 标志) =================
        success_flags = traj_stats.get("success")
        if success_flags is not None:
            # 只要 success_flags > 0，就说明触碰过终点！
            success_count = int((success_flags > 0).sum().item())
            total_drones = len(success_flags)
            success_rate = success_count / total_drones
            
            print("\n" + "🏆" * 25)
            print(f"🏁 [考核战报] 共有 {success_count} / {total_drones} 架无人机成功抵达终点！")
            print(f"📈 [环境通过率]   {success_rate * 100:.1f} %")
            print("🏆" * 25 + "\n")
            
            info["eval/success_rate"] = success_rate

        # ================= 🎬 视频打包与本地保存 =================
        if should_record and render_callback is not None and len(render_callback.frames) > 0:
            video_array = render_callback.get_video_array(axes="t c h w")
            fps_val = max(10, int(1.0 / (cfg.sim.dt * cfg.sim.substeps * eval_video_interval)))
            
            info["recording"] = wandb.Video(
                video_array,
                fps=fps_val,
                format="mp4"
            )
            
            try:
                import imageio
                video_local = video_array.permute(0, 2, 3, 1).cpu().numpy()
                if video_local.dtype != np.uint8:
                    if video_local.max() <= 1.0:
                        video_local = (video_local * 255).astype(np.uint8)
                    else:
                        video_local = video_local.astype(np.uint8)

                local_video_name = os.path.join(run.dir, f"eval_video_step_{collector._frames}.mp4")
                imageio.mimwrite(
                    local_video_name,
                    video_local,
                    fps=fps_val,
                    macro_block_size=None,
                    quality=9,
                )
                print(f"🎥 [录像已保存] 本地视频路径: {local_video_name}")
            except ImportError:
                print("[!] 未安装 imageio，跳过本地视频保存。")
            except Exception as e:
                print(f"[!] 本地视频保存失败: {e}")

        return info

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    if max_iters > 0:
        total_len = max_iters
    elif total_frames > 0:
        total_len = total_frames // frames_per_batch
    else:
        total_len = None

    last_return = 0.0
    success_rate_sum = 0.0
    success_rate_count = 0
    best_train_success_rate = 0.0
    best_eval_success_rate = 0.0
    last_train_success_rate = None
    pbar = tqdm(collector, total=total_len, dynamic_ncols=True)
    env.train()
    # for i, data in enumerate(pbar):
    #     info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
    #     episode_stats.add(data.to_tensordict())

    #     if len(episode_stats) >= base_env.num_envs:
    #         stats = {
    #             "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
    #             for k, v in episode_stats.pop().items(True, True)
    #         }
    #         info.update(stats)
    #         if "train/stats.return" in info:
    #             last_return = info["train/stats.return"]

    #         if "train/stats.return" in info:
    #             current_return = info["train/stats.return"]
    #             if current_return > max_return:
    #                 max_return = current_return
    #                 try:
    #                     ckpt_path = os.path.join(run.dir, f"checkpoint_best_return_{max_return:.2f}.pt")
    #                     torch.save(policy.state_dict(), ckpt_path)
    #                     if last_best_ckpt_path is not None and last_best_ckpt_path != ckpt_path:
    #                         try:
    #                             if os.path.exists(last_best_ckpt_path):
    #                                 os.remove(last_best_ckpt_path)
    #                         except OSError:
    #                             pass
    #                     last_best_ckpt_path = ckpt_path
    #                 except AttributeError:
    #                     logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    #         # info.update(policy.train_op(data.to_tensordict()))
    #         # ===== 在 train_op 前后检查参数是否真的更新 =====
    #     with torch.no_grad():
    #         w_before = actor_backbone.ku_encoder[0].weight.detach().clone()

    #     train_info = policy.train_op(data.to_tensordict())

    #     with torch.no_grad():
    #         w_after = actor_backbone.ku_encoder[0].weight.detach()
    #         delta = (w_after - w_before).abs().mean().item()

    #     info.update(train_info)
    #     info["debug/actor_backbone_delta"] = delta

    #     if i % 20 == 0:
    #         print(f"[debug] iter={i}, actor backbone param delta = {delta:.8e}")

    #     if eval_interval > 0 and i % eval_interval == 0:
    #         logging.info(f"Eval at {collector._frames} steps.")
    #         info.update(evaluate())
    #         env.train()
    #         base_env.train()


    #         if eval_interval > 0 and i % eval_interval == 0:
    #             logging.info(f"Eval at {collector._frames} steps.")
    #             info.update(evaluate())
    #             env.train()
    #             base_env.train()

    #         if save_interval > 0 and i % save_interval == 0:
    #             try:
    #                 ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
    #                 torch.save(policy.state_dict(), ckpt_path)
    #                 logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    #             except AttributeError:
    #                 logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    #         run.log(info)


    #         pbar.set_postfix({
    #             "fps": f"{collector._fps:.1f}", 
    #             "frames": collector._frames,
    #             "return": f"{last_return:.2f}"
    #         })
    #         if max_iters > 0 and i >= max_iters - 1:
    #             break
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) > 0:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

            if "train/stats.success" in info:
                current_success_rate = float(info["train/stats.success"])
                success_rate_sum += current_success_rate
                success_rate_count += 1
                last_train_success_rate = current_success_rate
                if current_success_rate > best_train_success_rate:
                    best_train_success_rate = current_success_rate

            if "train/stats.return" in info:
                last_return = info["train/stats.return"]
                current_return = info["train/stats.return"]

                if current_return > max_return:
                    max_return = current_return
                    try:
                        ckpt_path = os.path.join(run.dir, f"checkpoint_best_return_{max_return:.2f}.pt")
                        torch.save(policy.state_dict(), ckpt_path)
                        if last_best_return_ckpt_path is not None and last_best_return_ckpt_path != ckpt_path:
                            try:
                                if os.path.exists(last_best_return_ckpt_path):
                                    os.remove(last_best_return_ckpt_path)
                            except OSError:
                                pass
                        last_best_return_ckpt_path = ckpt_path
                    except AttributeError:
                        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        with torch.no_grad():
            actor_probe = actor_backbone.get_probe_params()
            critic_probe = critic_backbone.get_probe_params()
            actor_ku_before = actor_probe["ku_encoder"].detach().clone()
            actor_fusion_before = actor_probe["fusion"].detach().clone()
            critic_ku_before = critic_probe["ku_encoder"].detach().clone()
            critic_fusion_before = critic_probe["fusion"].detach().clone()

        train_info = policy.train_op(data.to_tensordict())

        with torch.no_grad():
            actor_probe = actor_backbone.get_probe_params()
            critic_probe = critic_backbone.get_probe_params()
            actor_ku_after = actor_probe["ku_encoder"].detach()
            actor_fusion_after = actor_probe["fusion"].detach()
            critic_ku_after = critic_probe["ku_encoder"].detach()
            critic_fusion_after = critic_probe["fusion"].detach()

            actor_delta_ku = (actor_ku_after - actor_ku_before).abs().mean().item()
            actor_delta_fusion = (actor_fusion_after - actor_fusion_before).abs().mean().item()
            critic_delta_ku = (critic_ku_after - critic_ku_before).abs().mean().item()
            critic_delta_fusion = (critic_fusion_after - critic_fusion_before).abs().mean().item()

            actor_delta = max(actor_delta_ku, actor_delta_fusion)
            critic_delta = max(critic_delta_ku, critic_delta_fusion)

        info.update(train_info)
        info["debug/actor_backbone_delta"] = actor_delta
        info["debug/critic_backbone_delta"] = critic_delta
        info["debug/actor_delta_lidar"] = actor_delta_ku
        info["debug/actor_delta_fusion"] = actor_delta_fusion
        info["debug/critic_delta_lidar"] = critic_delta_ku
        info["debug/critic_delta_fusion"] = critic_delta_fusion

        if i % 20 == 0:
            approx_kl = float(train_info.get("approx_kl", 0.0))
            clip_fraction = float(train_info.get("clip_fraction", 0.0))
            actor_grad_norm = float(train_info.get("actor_grad_norm", 0.0))
            if "train/stats.success" in info:
                current_success_rate = float(info["train/stats.success"])
            elif last_train_success_rate is not None:
                current_success_rate = float(last_train_success_rate)
            else:
                current_success_rate = 0.0
            avg_success_rate = (
                success_rate_sum / success_rate_count if success_rate_count > 0 else 0.0
            )
            print(
                f"[debug] iter={i}, "
                f"actor_delta={actor_delta:.8e} (ku={actor_delta_ku:.8e}, fusion={actor_delta_fusion:.8e}), "
                f"critic_delta={critic_delta:.8e}, clip_frac={clip_fraction:.3f}, "
                f"approx_kl={approx_kl:.3e}, actor_grad_norm={actor_grad_norm:.3e}, "
                f"success={current_success_rate * 100:.2f}%, "
                f"avg_success={avg_success_rate * 100:.2f}%, "
                f"best_train_success={best_train_success_rate * 100:.2f}%, "
                f"best_eval_success={best_eval_success_rate * 100:.2f}%"
            )

        if eval_interval > 0 and i % eval_interval == 0:
            train_success_snapshot = None
            if "train/stats.success" in info:
                train_success_snapshot = float(info["train/stats.success"])
            elif last_train_success_rate is not None:
                train_success_snapshot = float(last_train_success_rate)

            # ================= 🛡️ 保存/恢复 collector 状态，防止 eval 破坏训练连续性 =================
            frames_before_eval = int(collector._frames)
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            # eval returns with env reset; just switch back to train mode.
            env.train()
            base_env.train()
            frames_after_eval = int(collector._frames)
            info["debug/eval_consumed_train_frames"] = frames_after_eval - frames_before_eval

            # Keep train curve stable at eval boundary: use the pre-eval train snapshot.
            if train_success_snapshot is not None:
                info["train/stats.success"] = train_success_snapshot

            if "eval/success_rate" in info:
                current_eval_success_rate = float(info["eval/success_rate"])
                if current_eval_success_rate > best_eval_success_rate:
                    best_eval_success_rate = current_eval_success_rate
                    try:
                        ckpt_path = os.path.join(
                            run.dir,
                            f"checkpoint_best_eval_success_{best_eval_success_rate:.4f}.pt"
                        )
                        torch.save(policy.state_dict(), ckpt_path)
                        if last_best_success_ckpt_path is not None and last_best_success_ckpt_path != ckpt_path:
                            try:
                                if os.path.exists(last_best_success_ckpt_path):
                                    os.remove(last_best_success_ckpt_path)
                            except OSError:
                                pass
                        last_best_success_ckpt_path = ckpt_path
                    except AttributeError:
                        logging.warning(f"Policy {policy} does not implement `.state_dict()`")
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)

        pbar.set_postfix({
            "fps": f"{collector._fps:.1f}",
            "frames": collector._frames,
            "return": f"{last_return:.2f}"
        })

        if max_iters > 0 and i >= max_iters - 1:
            break   

    # ====== 最终评估录像 ======
    final_eval_ckpt_path = None
    if last_best_success_ckpt_path is not None and os.path.exists(last_best_success_ckpt_path):
        final_eval_ckpt_path = last_best_success_ckpt_path
    elif last_best_return_ckpt_path is not None and os.path.exists(last_best_return_ckpt_path):
        final_eval_ckpt_path = last_best_return_ckpt_path

    if final_eval_ckpt_path is not None:
        logging.info(f"Loading final-eval checkpoint from {final_eval_ckpt_path} for final evaluation.")
        try:
            _load_policy_checkpoint_compatible(policy, final_eval_ckpt_path, map_location=base_env.device)
        except Exception as e:
            logging.warning(f"Failed to load best checkpoint: {e}")

    final_export_topk = bool(cfg.get("eval_export_topk", True))
    final_topk_count = max(1, int(cfg.get("eval_topk_trajectories", 150)))
    final_eval_rounds = max(1, int(cfg.get("final_eval_rounds", 50)))
    logging.info(f"Final Eval at {collector._frames} steps. rounds={final_eval_rounds}")
    info = {"env_frames": collector._frames}
    
    try:
        success_rates = []
        last_eval_info = {}
        base_seed = int(cfg.get("seed", 0))

        for round_idx in range(final_eval_rounds):
            # 仅最后一轮导出 Top-K 轨迹，避免重复导出大文件
            do_export_topk = final_export_topk and (round_idx == final_eval_rounds - 1)
            round_info = evaluate(
                seed=base_seed + round_idx,
                export_topk=do_export_topk,
                topk=final_topk_count,
            )
            last_eval_info = round_info
            if "eval/success_rate" in round_info:
                success_rates.append(float(round_info["eval/success_rate"]))

        info.update(last_eval_info)
        if success_rates:
            avg_success_rate = float(np.mean(success_rates))
            info["final_eval/rounds"] = final_eval_rounds
            info["final_eval/avg_success_rate"] = avg_success_rate
            print(
                f"[Final Eval] {final_eval_rounds}轮平均成功率: "
                f"{avg_success_rate * 100:.2f}%"
            )

        run.log(info)
    except Exception as e:
        print(f"\n[!] 最终评估跳过: {e}\n")

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
