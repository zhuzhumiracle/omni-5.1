# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

import torch
import torch.distributions as D
import math
from dataclasses import dataclass
from typing import Callable

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Unbounded, Composite

try:
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    from omni.isaac.core.utils.viewports import set_camera_view

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg


# -------------------------- Mid-360 lidar pattern -------------------------- #
def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = cfg.num_samples
    t = torch.linspace(0, 0.1, num_samples, device=device)
    PI = torch.pi

    yaw_deg = (-62050.63 * t + 3.11 * torch.cos(314159.2 * t) * torch.sin(628.318 * 2 * t)) % 360
    pitch_deg = 22.5 + 29.5 * torch.cos(20 * PI * t) + 4 * torch.cos(2 * PI / 0.006 * t) * torch.cos(10000 * PI * t)

    yaw_rad = torch.deg2rad(yaw_deg)
    pitch_rad = torch.deg2rad(pitch_deg)

    dx = torch.cos(pitch_rad) * torch.cos(yaw_rad)
    dy = torch.cos(pitch_rad) * torch.sin(yaw_rad)
    dz = torch.sin(pitch_rad)

    ray_directions = torch.stack([dx, dy, dz], dim=-1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions



@dataclass
class Mid360PatternCfg(PatternBaseCfg):
    func: Callable = mid360_pattern
    num_samples: int = 20000  # 先默认低一点，跑稳后再往上加

# ------------------------------ Environment -------------------------------- #
class forest_singal(IsaacEnv):
    def __init__(self, cfg, headless):
        # ---------------- basic cfg ----------------
        self.lidar_update_interval = int(cfg.task.get("lidar_update_interval", 5))  # 50Hz -> 10Hz
        self.k_hist = int(cfg.task.get("k_hist", 5))

        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()

        self.control_mode = str(cfg.task.get("control_mode", "rotor")).lower()
        if self.control_mode not in {"rotor", "velocity"}:
            raise ValueError(f"Unsupported control_mode={self.control_mode}. Expected 'rotor' or 'velocity'.")
        self.velocity_action_dim = int(cfg.task.get("velocity_action_dim", 4))
        if self.velocity_action_dim not in {3, 4, 5}:
            raise ValueError(
                f"velocity_action_dim={self.velocity_action_dim} is not supported yet. "
                "Use 3D [vx, vy, vz], 4D [vx, vy, vz, yaw], or "
                "5D [dir_x, dir_y, dir_z, speed_ratio, yaw] actions."
            )
        self.velocity_frame = str(cfg.task.get("velocity_frame", "world")).lower()
        if self.velocity_frame not in {"world", "body", "goal"}:
            raise ValueError(
                f"Unsupported velocity_frame={self.velocity_frame}. Expected 'world', 'body', or 'goal'."
            )
        self.target_yaw_mode = str(cfg.task.get("target_yaw_mode", "action")).lower()
        if self.target_yaw_mode not in {"goal", "velocity", "current", "action"}:
            raise ValueError(
                f"Unsupported target_yaw_mode={self.target_yaw_mode}. "
                "Expected 'goal', 'velocity', 'current', or 'action'."
            )
        if self.target_yaw_mode == "action" and self.velocity_action_dim < 4:
            raise ValueError("target_yaw_mode=action requires velocity_action_dim >= 4.")
        self.velocity_yaw_speed_threshold = float(cfg.task.get("velocity_yaw_speed_threshold", 0.2))

        self.vlim = float(cfg.task.get("vlim", cfg.task.get("v_max", 3.0)))
        self.vlim_randomize = bool(cfg.task.get("vlim_randomize", False))
        self.vlim_train_min = float(cfg.task.get("vlim_train_min", self.vlim))
        self.vlim_train_max = float(cfg.task.get("vlim_train_max", self.vlim))
        if self.vlim_train_max < self.vlim_train_min:
            raise ValueError(
                f"vlim_train_max ({self.vlim_train_max}) must be >= vlim_train_min ({self.vlim_train_min})."
            )
        self.observe_vlim = bool(cfg.task.get("observe_vlim", False))
        self.v_max = float(cfg.task.get("v_max", self.vlim))
        self.z_min = float(cfg.task.get("z_min", 0.5))
        self.z_max = float(cfg.task.get("z_max", 3.5))
        self.lambda_esdf = float(cfg.task.get("lambda_esdf", 1.0))
        self.k_esdf = float(cfg.task.get("k_esdf", 2.0))
        self.collision_dist = float(cfg.task.get("collision_dist", 0.3))
        self.goal_radius = float(cfg.task.get("goal_radius", 2.0))
        self.goal_bonus = float(cfg.task.get("goal_bonus", 1000.0))
        self.collision_penalty = float(cfg.task.get("collision_penalty", -20.0))
        # 速度相关碰撞惩罚系数: r_collision = -k * v
        # 默认使用原固定惩罚绝对值，保持量级连续可控
        self.collision_speed_k = float(cfg.task.get("collision_speed_k", abs(self.collision_penalty)))
        self.reset_on_collision = bool(cfg.task.get("reset_on_collision", True)) #开启了碰撞就重置的
        self.collision_force_threshold = float(cfg.task.get("collision_force_threshold", 1.0))
        self.flip_tilt_deg = float(cfg.task.get("flip_tilt_deg", 80.0))
        self.flip_consecutive_steps = int(cfg.task.get("flip_consecutive_steps", 10))

        self.milestone_segments = int(cfg.task.get("milestone_segments", 0))
        self.milestone_include_goal_segment = bool(cfg.task.get("milestone_include_goal_segment", False))
        self.milestone_mid_reward = float(cfg.task.get("milestone_mid_reward", self.goal_bonus * 0.2))
        raw_milestone_percents = cfg.task.get("milestone_reward_percents", [])
        self.milestone_reward_percents = [float(x) for x in raw_milestone_percents]

        # 必须在 super().__init__ 之前确定分段数量；_set_specs 在 super 里会用到。
        if self.milestone_segments <= 0:
            milestone_thresholds = []
        elif self.milestone_segments == 1:
            milestone_thresholds = [0.5]
        else:
            milestone_thresholds = [
                i / self.milestone_segments
                for i in range(1, self.milestone_segments)
            ]
            if self.milestone_include_goal_segment:
                milestone_thresholds.append(1.0)

        self.num_milestones = len(milestone_thresholds)

        if self.num_milestones > 0:
            if len(self.milestone_reward_percents) == 0:
                self.milestone_reward_percents = [100.0] * self.num_milestones
            elif len(self.milestone_reward_percents) < self.num_milestones:
                last_percent = self.milestone_reward_percents[-1]
                self.milestone_reward_percents.extend(
                    [last_percent] * (self.num_milestones - len(self.milestone_reward_percents))
                )
            else:
                self.milestone_reward_percents = self.milestone_reward_percents[: self.num_milestones]

        self._milestone_thresholds_list = milestone_thresholds
        self._milestone_rewards_list = [
            self.milestone_mid_reward * p / 100.0
            for p in self.milestone_reward_percents
        ]

        self.w_forward = float(cfg.task.get("w_forward", 5.0))
        self.w_smooth = float(cfg.task.get("w_smooth", -0.2))
        self.w_max_speed = float(cfg.task.get("w_max_speed", -0.5))
        self.w_z = float(cfg.task.get("w_z", -5.0))
        self.w_esdf = float(cfg.task.get("w_esdf", 1.5))
        self.w_yaw = float(cfg.task.get("w_yaw", 0.5))
        self.w_thrust = float(cfg.task.get("w_thrust", 0.0))

        self.terminate_z_min = float(cfg.task.get("terminate_z_min", 0.2))
        self.terminate_z_max = float(cfg.task.get("terminate_z_max", 5.0))
        self.terminate_v_norm = float(cfg.task.get("terminate_v_norm", 5.0))

        # ---------------- lidar preprocess cfg ----------------
        self.max_obs_dist = float(cfg.task.get("max_obs_dist", 10.0))
        self.voxel_size = float(cfg.task.get("voxel_size", 0.05))

        # 雷达采样数直接读取任务 YAML，避免代码默认值掩盖配置问题
        self.num_lidar_points = int(cfg.task.num_lidar_points)
        self.num_yaw_bins = int(cfg.task.get("num_yaw_bins", 80))
        self.num_pitch_bins = int(cfg.task.get("num_pitch_bins", 40))
        self.downsampled_dim = self.num_yaw_bins * self.num_pitch_bins

        # 全局步数
        self.global_step = 0

        super().__init__(cfg, headless)
        if self.controller is not None:
            self.controller = self.controller.to(self.device)

        self.vlim_episode = torch.full((self.num_envs, 1, 1), self.vlim, device=self.device)
        self.prev_pos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.steps_since_reset = torch.zeros((self.num_envs, 1, 1), dtype=torch.int32, device=self.device)
        self.flip_counter = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)

        self.lidar._initialize_impl()
        self.lidar_resolution = (self.num_lidar_points, 1)
        self.drone.initialize(track_contact_forces=self.reset_on_collision)

        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.last_actions = torch.zeros(self.num_envs, 1, self.action_dim, device=self.device)
        self.current_actions = torch.zeros_like(self.last_actions)
        self.last_target_vel = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.current_target_vel = torch.zeros_like(self.last_target_vel)

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = 24.
            self.target_pos[:, 0, 2] = 2.
            self.start_pos = torch.zeros_like(self.target_pos)

        # pitch_bin_centers = torch.linspace(-90.0, 90.0, self.num_pitch_bins, device=self.device)
        # self.fov_mask = (pitch_bin_centers >= -7.0) & (pitch_bin_centers <= 52.0)
        # self.fov_mask = self.fov_mask.unsqueeze(1).repeat(1, self.num_yaw_bins).reshape(1, 1, -1)
        # self.fov_mask = self.fov_mask.expand(self.num_envs, 1, self.downsampled_dim)

        # # 历史 seen mask
        # self.hist_seen_mask = torch.zeros(
        #     (self.k_hist, self.num_envs, 1, self.downsampled_dim),
        #     dtype=torch.bool,
        #     device=self.device
        # )
        # self.hist_head = 0
        # ---------------- history buffers for k-frame point clouds ----------------
        # 论文做法：缓存最近 k 帧点云，并在构造观测时统一变换到“当前时刻 body frame”
        # ---------------- history buffers for k-frame lidar / FoV ----------------
        self.hist_ray_hits_w = torch.full(
            (self.k_hist, self.num_envs, self.num_lidar_points, 3),
            float("nan"),
            device=self.device,
        )

        self.hist_sensor_pos_w = torch.full(
            (self.k_hist, self.num_envs, 3),
            float("nan"),
            device=self.device,
        )

        # 新增：历史传感器姿态（世界系四元数）
        self.hist_sensor_quat_w = torch.full(
            (self.k_hist, self.num_envs, 4),
            float("nan"),
            device=self.device,
        )

        # 新增：每条历史射线在“该历史帧”中观测到的自由边界距离
        # - 若命中障碍：就是 hit distance
        # - 若没命中：就是 max_obs_dist
        self.hist_ray_end_dist = torch.full(
            (self.k_hist, self.num_envs, self.num_lidar_points),
            self.max_obs_dist,
            device=self.device,
        )

        # 新增：哪些历史 ray 真正命中了障碍（用于 obstacle_dist）
        self.hist_has_hit = torch.zeros(
            (self.k_hist, self.num_envs, self.num_lidar_points),
            dtype=torch.bool,
            device=self.device,
        )

        self.hist_valid = torch.zeros(
            (self.k_hist, self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )

        self.hist_head = 0

        # 新增：预计算 lidar 在本体坐标系下的方向（固定模式）
        _, ray_dirs_local = mid360_pattern(
            Mid360PatternCfg(num_samples=self.num_lidar_points),
            self.device
        )
        self.ray_dirs_local = torch.nn.functional.normalize(ray_dirs_local, dim=-1)



        # voxel hash
        self._voxel_offset = int(math.ceil(self.max_obs_dist / self.voxel_size)) + 2
        self._voxel_base = 2 * self._voxel_offset + 1

        # ---------------- cache: 关键优化 ----------------
        self.encoded_lidar_cache = torch.full(
            (self.num_envs, 1, self.downsampled_dim),
            self.max_obs_dist,
            device=self.device
        )
        self.lidar_scan_cache = torch.zeros(
            (self.num_envs, 1, self.num_lidar_points),
            device=self.device
        )
        self.lidar_dirty = False

        if self.num_milestones > 0:
            self.milestone_thresholds = torch.tensor(self._milestone_thresholds_list, device=self.device)
            self.milestone_rewards = torch.tensor(self._milestone_rewards_list, device=self.device)
        else:
            self.milestone_thresholds = torch.zeros(0, device=self.device)
            self.milestone_rewards = torch.zeros(0, device=self.device)

        self.init_goal_dist = torch.ones((self.num_envs, 1), device=self.device)
        self.milestone_hit = torch.zeros((self.num_envs, self.num_milestones), dtype=torch.bool, device=self.device)

        # ================= 📊 新增：注册详细的奖励追踪项 =================
        # 这些 key 将会直接变成 WandB 图表上的曲线名字
        tracking_keys = [
            "reward_forward", 
            "reward_smooth", 
            "reward_max_speed",
            "reward_z", 
            "reward_esdf", 
            "reward_collision",
            "reward_yaw", 
            "reward_goal",
            "reward_death", # 如果你加上了坠机惩罚的话
            "reward_thrust",
            "reward_milestone",
            "action_sat",  # <==== 加入这行！
            "death_z_low",
            "death_z_high",
            "death_overspeed",
            "death_collision",
            "death_contact",
            "death_oob",
            "death_flip",
            "death_nan",
        ]
        for i in range(self.num_milestones):
            tracking_keys.append(f"reward_milestone_{i + 1}")
        
        # 为每个追踪项初始化一个全为 0 的 Tensor
        for key in tracking_keys:
            self.stats[key] = torch.zeros(self.num_envs, 1, device=self.device)
        # ===============================================================

    # --------------------------------------------------------------------- #
    # def _update_history_seen_mask(self):
    #     # 简化版：只标记当前 FoV 覆盖角度
    #     self.hist_seen_mask[self.hist_head].copy_(self.fov_mask.bool())
    #     self.hist_head = (self.hist_head + 1) % self.k_hist
    def _transform_points_w_to_body(self, points_w: torch.Tensor, curr_pos_w: torch.Tensor, curr_quat_w: torch.Tensor):
        """
        points_w: [num_envs, num_points, 3]，世界系点
        curr_pos_w: [num_envs, 3]，当前机体位置（世界系）
        curr_quat_w: [num_envs, 4]，当前机体姿态四元数（世界系）
        return: [num_envs, num_points, 3]，当前时刻 body frame 下的点
        """
        num_envs, num_points = points_w.shape[:2]
        rel_w = points_w - curr_pos_w[:, None, :]
        quat_expand = curr_quat_w[:, None, :].expand(-1, num_points, -1).reshape(-1, 4)
        rel_b = quat_rotate_inverse(quat_expand, rel_w.reshape(-1, 3)).reshape(num_envs, num_points, 3)
        return rel_b
    def _quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor):
        """
        quat: [..., 4]，默认 wxyz
        vec : [..., 3]
        return: [..., 3]
        """
        q_w = quat[..., :1]
        q_xyz = quat[..., 1:]
        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)
    def _push_current_lidar_frame_to_history(self):
        """
        把当前 lidar 帧压入历史缓存：
        1) 世界系命中点（仅真实命中障碍的 ray 保留）
        2) 当前传感器位姿
        3) 每条 ray 的“已观测自由边界距离”
        - hit: 用 hit distance
        - miss / invalid: 用 max_obs_dist
        """
        slot = self.hist_head

        ray_hits_w = self.lidar.data.ray_hits_w.reshape(self.num_envs, self.num_lidar_points, 3)
        sensor_pos_w = self.lidar.data.pos_w.reshape(self.num_envs, 3)

        # 由于你的 RayCaster offset=(0,0,0) 且 attach_yaw_only=False，
        # 这里直接用机体当前四元数作为传感器姿态是成立的
        sensor_quat_w = self.drone_state[..., 3:7].squeeze(1)  # [E, 4]

        rel_w = ray_hits_w - sensor_pos_w[:, None, :]
        raw_hit_dist = rel_w.norm(dim=-1)  # [E, P]

        finite_hit = torch.isfinite(raw_hit_dist)
        has_hit = finite_hit & (raw_hit_dist < (self.max_obs_dist - 1e-6))

        # 历史真实命中点：未命中或无效的 ray 不作为障碍点
        hist_hits_w = torch.where(
            has_hit[..., None],
            ray_hits_w,
            torch.full_like(ray_hits_w, float("nan"))
        )

        # 每条 ray 的历史 FoV 边界距离
        # 命中障碍：边界到障碍为止
        # 未命中 / 无效：边界记为 max_obs_dist，表示这段空间已观测为 free
        ray_end_dist = torch.where(
            finite_hit,
            raw_hit_dist.clamp(0.0, self.max_obs_dist),
            torch.full_like(raw_hit_dist, self.max_obs_dist),
        )

        self.hist_ray_hits_w[slot] = hist_hits_w
        self.hist_sensor_pos_w[slot] = sensor_pos_w
        self.hist_sensor_quat_w[slot] = sensor_quat_w
        self.hist_ray_end_dist[slot] = ray_end_dist
        self.hist_has_hit[slot] = has_hit
        self.hist_valid[slot] = True

        self.hist_head = (self.hist_head + 1) % self.k_hist
    def _scatter_points_to_bins(self, rel_points_body: torch.Tensor, reduce: str):
        """
        把当前 body frame 下的点云投到 yaw-pitch 分区中。
        reduce='amin' : 每个 bin 取最近点距离（障碍物距离）
        reduce='amax' : 每个 bin 取最远已观测距离（用于 d_unknown）
        """
        num_envs, num_points, _ = rel_points_body.shape
        point_dist = rel_points_body.norm(dim=-1)

        valid_mask = torch.isfinite(point_dist) & (point_dist > 1e-6) & (point_dist <= self.max_obs_dist)

        if reduce == "amin":
            out = torch.full(
                (num_envs, self.downsampled_dim),
                self.max_obs_dist,
                device=self.device,
                dtype=point_dist.dtype,
            )
        elif reduce == "amax":
            out = torch.zeros(
                (num_envs, self.downsampled_dim),
                device=self.device,
                dtype=point_dist.dtype,
            )
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")

        if not valid_mask.any():
            return out

        env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(num_envs, num_points)

        valid_env = env_ids[valid_mask]
        valid_pts = rel_points_body[valid_mask]
        valid_dist = point_dist[valid_mask].clamp(0.0, self.max_obs_dist)

        x = valid_pts[:, 0]
        y = valid_pts[:, 1]
        z = valid_pts[:, 2]

        yaw = torch.atan2(y, x)
        yaw = torch.remainder(yaw, 2 * torch.pi)

        pitch = torch.atan2(z, torch.sqrt(x * x + y * y + 1e-6))

        yaw_idx = torch.clamp(
            (yaw / (2 * torch.pi) * self.num_yaw_bins).long(),
            0, self.num_yaw_bins - 1
        )
        pitch_idx = torch.clamp(
            (((pitch + torch.pi / 2) / torch.pi) * self.num_pitch_bins).long(),
            0, self.num_pitch_bins - 1
        )

        flat_bin = pitch_idx * self.num_yaw_bins + yaw_idx
        flat_index = valid_env * self.downsampled_dim + flat_bin

        flat_out = out.reshape(-1)
        flat_out.scatter_reduce_(0, flat_index, valid_dist, reduce=reduce, include_self=True)
        out = flat_out.view(num_envs, self.downsampled_dim)

        return out
    # --------------------------------------------------------------------- #
    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import RayCaster, RayCasterCfg
        from isaaclab.terrains import (
            TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg,
        )

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0., 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            prim_path="/World/ground",
            terrain_type="generator",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                improve_patch_friction=None,
            ),
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=5,
                num_cols=5,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(8.0, 8.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=40,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 0.8),
                        obstacle_height_range=(3.0, 4.0),
                        platform_width=1.5,
                    )
                },
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        self.lidar_vfov = (
            max(-89., self.cfg.task.lidar_vfov[0]),
            min(89., self.cfg.task.lidar_vfov[1])
        )
        self.lidar_range = self.cfg.task.lidar_range

        from isaaclab.sensors import RayCaster, RayCasterCfg
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=False,
            pattern_cfg=Mid360PatternCfg(num_samples=self.num_lidar_points),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        return ["/World/ground"]

    # --------------------------------------------------------------------- #
    def _set_specs(self):
        self.motor_action_dim = self.drone.action_spec.shape[-1]
        if self.control_mode == "velocity":
            self.action_dim = self.velocity_action_dim
            policy_action_spec = Bounded(-1, 1, self.action_dim, device=self.device)
        else:
            self.action_dim = self.motor_action_dim
            policy_action_spec = self.drone.action_spec
        

        lidar_dim = self.downsampled_dim
        state_dim = 10 + self.action_dim + (1 if self.observe_vlim else 0)
        obs_dim = state_dim + lidar_dim

        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Unbounded((1, obs_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = Composite({
            "agents": Composite({
                "action": policy_action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        # stats_spec = Composite({
        #     "return": Unbounded(1),
        #     "episode_len": Unbounded(1),
        #     "action_smoothness": Unbounded(1),
        #     "safety": Unbounded(1),
        #     "success": Unbounded(1),
        # }).expand(self.num_envs).to(self.device)

        stats_spec_dict = {
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "action_smoothness": Unbounded(1),
            "safety": Unbounded(1),
            "success": Unbounded(1),
            # 下面是新加的！
            "reward_forward": Unbounded(1),
            "reward_smooth": Unbounded(1),
            "reward_max_speed": Unbounded(1),
            "reward_z": Unbounded(1),
            "reward_esdf": Unbounded(1),
            "reward_collision": Unbounded(1),
            "reward_yaw": Unbounded(1),
            "reward_goal": Unbounded(1),
            "reward_death": Unbounded(1),
            "reward_thrust": Unbounded(1),
            "reward_milestone": Unbounded(1),
            "death_z_low": Unbounded(1),
            "death_z_high": Unbounded(1),
            "death_overspeed": Unbounded(1),
            "death_collision": Unbounded(1),
            "death_contact": Unbounded(1),
            "death_oob": Unbounded(1),
            "death_flip": Unbounded(1),
            "death_nan": Unbounded(1),
        }
        for i in range(self.num_milestones):
            stats_spec_dict[f"reward_milestone_{i + 1}"] = Unbounded(1)

        stats_spec = Composite(stats_spec_dict).expand(self.num_envs).to(self.device)
        
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    # --------------------------------------------------------------------- #
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
        pos[:, 0, 1] = -24.
        pos[:, 0, 2] = 2.

        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)

        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.prev_pos[env_ids] = pos
        self.stats[env_ids] = 0.
        self.last_actions[env_ids] = 0.0
        self.current_actions[env_ids] = 0.0
        self.last_target_vel[env_ids] = 0.0
        self.current_target_vel[env_ids] = 0.0
        if self.vlim_randomize and self.training:
            self.vlim_episode[env_ids] = torch.empty(len(env_ids), 1, 1, device=self.device).uniform_(
                self.vlim_train_min,
                self.vlim_train_max,
            )
        else:
            self.vlim_episode[env_ids] = self.vlim
        self.steps_since_reset[env_ids] = 0
        self.flip_counter[env_ids] = 0
        # self.hist_seen_mask[:, env_ids] = False
        self.hist_ray_hits_w[:, env_ids] = float("nan")
        self.hist_sensor_pos_w[:, env_ids] = float("nan")
        self.hist_sensor_quat_w[:, env_ids] = float("nan")
        self.hist_ray_end_dist[:, env_ids] = self.max_obs_dist
        self.hist_has_hit[:, env_ids] = False
        self.hist_valid[:, env_ids] = False
        if self.num_milestones > 0:
            self.milestone_hit[env_ids] = False

        self.init_goal_dist[env_ids] = (
            (self.target_pos[env_ids] - pos).norm(dim=-1).clamp_min(1e-6)
        )
        self.start_pos[env_ids] = pos

        # reset 缓存，避免用到旧 episode 数据
        self.encoded_lidar_cache[env_ids] = self.max_obs_dist
        self.lidar_scan_cache[env_ids] = 0.0
        self.lidar_dirty = True

    def _prepare_velocity_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert raw policy output to executable velocity action.

        5D actions use [direction_xyz, speed_ratio, yaw], separating speed
        magnitude from direction before vlim scaling.  3D/4D actions keep the
        old normalized velocity-vector behavior for compatibility.
        """
        if self.velocity_action_dim >= 5:
            action_exec = torch.zeros_like(actions)
            dir_raw = torch.tanh(actions[..., :3])
            dir_raw = torch.nan_to_num(dir_raw, nan=0.0, posinf=1.0, neginf=-1.0)
            dir_norm = dir_raw.norm(dim=-1, keepdim=True)
            direction = torch.where(
                dir_norm > 1e-6,
                dir_raw / dir_norm.clamp_min(1e-6),
                torch.zeros_like(dir_raw),
            )
            speed_ratio = (actions[..., 3:4] + 1.0) / 2.0
            speed_ratio = torch.nan_to_num(speed_ratio, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            yaw = torch.tanh(actions[..., 4:5])
            yaw = torch.nan_to_num(yaw, nan=0.0, posinf=1.0, neginf=-1.0)

            action_exec[..., :3] = direction
            action_exec[..., 3:4] = speed_ratio.clamp(0.0, 1.0)
            action_exec[..., 4:5] = yaw.clamp(-1.0, 1.0)
            return action_exec

        action_norm = torch.tanh(actions)
        action_norm = torch.clamp(action_norm, min=-1.0, max=1.0)
        if torch.isnan(action_norm).any():
            action_norm = torch.nan_to_num(action_norm, nan=0.0)
        action_exec = action_norm.clone()
        vel_action = action_exec[..., :3]
        vel_norm = vel_action.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vel_action = torch.where(vel_norm > 1.0, vel_action / vel_norm, vel_action)
        action_exec[..., :3] = vel_action
        return action_exec

    def _map_velocity_action_to_world(self, action_norm: torch.Tensor, root_state: torch.Tensor) -> torch.Tensor:
        if self.velocity_action_dim >= 5:
            vel_action = action_norm[..., :3] * action_norm[..., 3:4]
        else:
            vel_action = action_norm[..., :3]
        target_vel_local = vel_action * self.vlim_episode
        if self.velocity_frame == "world":
            return target_vel_local
        if self.velocity_frame == "body":
            return quat_rotate(root_state[..., 3:7], target_vel_local)

        goal_vec = self.target_pos - self.start_pos
        goal_vec_xy = goal_vec.clone()
        goal_vec_xy[..., 2] = 0.0
        x_axis = goal_vec_xy / goal_vec_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z_axis = torch.zeros_like(x_axis)
        z_axis[..., 2] = 1.0
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (
            target_vel_local[..., 0:1] * x_axis
            + target_vel_local[..., 1:2] * y_axis
            + target_vel_local[..., 2:3] * z_axis
        )

    def _compute_target_yaw(
        self,
        root_state: torch.Tensor,
        target_vel_world: torch.Tensor,
        action_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.target_yaw_mode == "action":
            yaw_idx = 4 if self.velocity_action_dim >= 5 else 3
            if action_norm is None or action_norm.shape[-1] <= yaw_idx:
                raise RuntimeError("target_yaw_mode=action requires an action yaw channel.")
            return action_norm[..., yaw_idx:yaw_idx + 1] * math.pi

        goal_vec = self.target_pos - root_state[..., :3]
        goal_yaw = torch.atan2(goal_vec[..., 1:2], goal_vec[..., 0:1])
        if self.target_yaw_mode == "goal":
            return goal_yaw

        heading = self.drone.heading[..., :3]
        current_yaw = torch.atan2(heading[..., 1:2], heading[..., 0:1])
        if self.target_yaw_mode == "current":
            return current_yaw

        vel_yaw = torch.atan2(target_vel_world[..., 1:2], target_vel_world[..., 0:1])
        vel_xy_norm = target_vel_world[..., :2].norm(dim=-1, keepdim=True)
        return torch.where(vel_xy_norm > self.velocity_yaw_speed_threshold, vel_yaw, goal_yaw)

    # --------------------------------------------------------------------- #
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        if self.control_mode == "velocity":
            if self.controller is None:
                raise RuntimeError("control_mode=velocity requires a configured LeePositionController.")

            action_exec = self._prepare_velocity_action(actions)

            root_state = self.drone.get_state(env_frame=False)[..., :13]
            target_vel_world = self._map_velocity_action_to_world(action_exec, root_state)
            target_yaw = self._compute_target_yaw(root_state, target_vel_world, action_exec)
            target_pos = root_state[..., :3]
            target_acc = torch.zeros_like(target_vel_world)

            rotor_cmds = self.controller.compute(
                root_state,
                target_pos=target_pos,
                target_vel=target_vel_world,
                target_acc=target_acc,
                target_yaw=target_yaw,
            )
            rotor_cmds = torch.nan_to_num(rotor_cmds, nan=0.0, posinf=1.0, neginf=-1.0)
            rotor_cmds = torch.clamp(rotor_cmds, min=-1.0, max=1.0)

            self.last_actions.copy_(self.current_actions)
            self.last_target_vel.copy_(self.current_target_vel)
            self.effort = self.drone.apply_action(rotor_cmds)
            self.current_actions = action_exec.clone()
            self.current_target_vel = target_vel_world.clone()
            return

        # 先用 tanh 平滑限幅，避免策略原始输出过大导致控制突变
        actions_exec = torch.tanh(actions)
        # ================= 🛡️ 核心防线 1：动作截断 =================
        actions_exec = torch.clamp(actions_exec, min=-1.0, max=1.0)
        
        # 过滤 NaN，如果网络输出了 NaN，强制归零
        if torch.isnan(actions_exec).any():
            actions_exec = torch.nan_to_num(actions_exec, nan=0.0)
        # =========================================================
        # ⚠️ 不写回 tensordict，保持 ("agents", "action") 为策略原始输出
        # 这样训练时 log_prob 计算才与采样时一致，ratio 不会被扭曲
        actions = actions_exec
        # 先保存上一时刻动作，再写当前动作
        self.last_actions.copy_(self.current_actions)
        self.last_target_vel.copy_(self.current_target_vel)

        self.effort = self.drone.apply_action(actions)
        self.current_actions = actions.clone()
        self.current_target_vel.zero_()

    # --------------------------------------------------------------------- #
    def _post_sim_step(self, tensordict: TensorDictBase):
        self.global_step += 1
        self.steps_since_reset += 1

        # 🛡️ 物理引擎速度硬限制：防止碰撞穿透导致的数值爆炸
        # 原理：不是截断（clamp），而是等比例缩放（rescale），保持速度方向不变，只压缩大小。
        # 这样不会引入方向突变，对策略学习更友好。
        max_physics_speed = 20.0   # m/s，远超正常飞行但能阻止 1e30 爆炸
        max_ang_speed = 50.0       # rad/s
        vels = self.drone.get_velocities()  # [E, 1, 6]

        v_lin = vels[..., :3]
        v_lin_norm = v_lin.norm(dim=-1, keepdim=True)
        lin_scale = torch.where(
            v_lin_norm > max_physics_speed,
            max_physics_speed / v_lin_norm.clamp_min(1e-6),
            torch.ones_like(v_lin_norm)
        )
        vels[..., :3] = v_lin * lin_scale

        v_ang = vels[..., 3:]
        v_ang_norm = v_ang.norm(dim=-1, keepdim=True)
        ang_scale = torch.where(
            v_ang_norm > max_ang_speed,
            max_ang_speed / v_ang_norm.clamp_min(1e-6),
            torch.ones_like(v_ang_norm)
        )
        vels[..., 3:] = v_ang * ang_scale

        self.drone.set_velocities(vels)

        # if self.global_step % self.lidar_update_interval == 0:
        #     self.lidar.update(self.dt * self.lidar_update_interval)
        #     self._update_history_seen_mask()
        #     self.lidar_dirty = True
        if self.global_step % self.lidar_update_interval == 0:
            self.lidar.update(self.dt * self.lidar_update_interval)
            self.lidar_dirty = True

    # --------------------------------------------------------------------- #
    # def _encode_lidar_observation(self):
    #     # 1) 原始扫描缓存
    #     lidar_scan = self.lidar_range - (
    #         (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
    #         .norm(dim=-1)
    #         .clamp_max(self.lidar_range)
    #         .reshape(self.num_envs, 1, self.num_lidar_points)
    #     )

    #     # 2) 点云转相对坐标
    #     rel_points = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
    #     rel_points = rel_points.reshape(self.num_envs, self.num_lidar_points, 3)
    #     point_dist = rel_points.norm(dim=-1)

    #     valid_mask = point_dist <= self.max_obs_dist

    #     env_ids = (
    #         torch.arange(self.num_envs, device=self.device)
    #         .unsqueeze(1)
    #         .expand(self.num_envs, self.num_lidar_points)
    #     )

    #     valid_env = env_ids[valid_mask]
    #     valid_pts = rel_points[valid_mask]
    #     valid_dist = point_dist[valid_mask]

    #     max_dist = self.max_obs_dist
    #     downsampled_scan = torch.full(
    #         (self.num_envs, self.downsampled_dim),
    #         max_dist,
    #         device=self.device,
    #         dtype=lidar_scan.dtype,
    #     )

    #     if valid_pts.numel() > 0:
    #         voxel_idx = torch.floor(valid_pts / self.voxel_size).to(torch.int64)

    #         vx = voxel_idx[:, 0] + self._voxel_offset
    #         vy = voxel_idx[:, 1] + self._voxel_offset
    #         vz = voxel_idx[:, 2] + self._voxel_offset

    #         inside = (
    #             (vx >= 0) & (vx < self._voxel_base) &
    #             (vy >= 0) & (vy < self._voxel_base) &
    #             (vz >= 0) & (vz < self._voxel_base)
    #         )

    #         valid_env = valid_env[inside]
    #         valid_dist = valid_dist[inside]
    #         vx = vx[inside]
    #         vy = vy[inside]
    #         vz = vz[inside]

    #         if valid_env.numel() > 0:
    #             key = (((valid_env * self._voxel_base + vx) * self._voxel_base + vy) * self._voxel_base + vz)
    #             uniq_key, inv = torch.unique(key, return_inverse=True)

    #             voxel_min_dist = torch.full(
    #                 (uniq_key.numel(),),
    #                 max_dist,
    #                 device=self.device,
    #                 dtype=valid_dist.dtype,
    #             )
    #             voxel_min_dist.scatter_reduce_(0, inv, valid_dist, reduce="amin", include_self=True)

    #             tmp = uniq_key
    #             uz = tmp % self._voxel_base
    #             tmp = torch.div(tmp, self._voxel_base, rounding_mode="floor")
    #             uy = tmp % self._voxel_base
    #             tmp = torch.div(tmp, self._voxel_base, rounding_mode="floor")
    #             ux = tmp % self._voxel_base
    #             uenv = torch.div(tmp, self._voxel_base, rounding_mode="floor")

    #             cx = (ux - self._voxel_offset + 0.5).to(torch.float32) * self.voxel_size
    #             cy = (uy - self._voxel_offset + 0.5).to(torch.float32) * self.voxel_size
    #             cz = (uz - self._voxel_offset + 0.5).to(torch.float32) * self.voxel_size

    #             yaw = torch.atan2(cy, cx)
    #             yaw = torch.remainder(yaw, 2 * torch.pi)
    #             pitch = torch.atan2(cz, torch.sqrt(cx * cx + cy * cy + 1e-6))

    #             yaw_idx = torch.clamp(
    #                 (yaw / (2 * torch.pi) * self.num_yaw_bins).long(),
    #                 0, self.num_yaw_bins - 1
    #             )
    #             pitch_idx = torch.clamp(
    #                 (((pitch + torch.pi / 2) / torch.pi) * self.num_pitch_bins).long(),
    #                 0, self.num_pitch_bins - 1
    #             )

    #             flat_bin = pitch_idx * self.num_yaw_bins + yaw_idx
    #             out_key = uenv * self.downsampled_dim + flat_bin

    #             flat_out = downsampled_scan.reshape(-1)
    #             flat_out.scatter_reduce_(0, out_key, voxel_min_dist, reduce="amin", include_self=True)
    #             downsampled_scan = flat_out.view(self.num_envs, self.downsampled_dim)

    #     d_min = downsampled_scan.unsqueeze(1).clamp_max(self.max_obs_dist)

    #     seen_mask = self.hist_seen_mask.any(dim=0)
    #     has_point = d_min < self.max_obs_dist - 1e-6
    #     unknown_mask = (~has_point) & (~seen_mask)

    #     encoded_lidar = d_min.clone()
    #     encoded_lidar = torch.where(
    #         (~has_point) & seen_mask,
    #         torch.full_like(encoded_lidar, self.max_obs_dist),
    #         encoded_lidar
    #     )
    #     encoded_lidar = torch.where(
    #         unknown_mask,
    #         torch.full_like(encoded_lidar, 15.0),
    #         encoded_lidar
    #     )

    #     return encoded_lidar, lidar_scan
    def _encode_lidar_observation(self):
        """
        更接近论文的实现：
        1) 聚合最近 k 帧障碍点云，统一变换到当前 body frame，得到每个 bin 的最近障碍距离；
        2) 聚合最近 k 帧 FoV 射线终点，统一变换到当前 body frame，得到每个 bin 已观测 free 的最远边界；
        3) 若该 bin 有障碍点，编码为最近障碍距离；
        若没有障碍点，编码为 20 - d_unknown，
        其中 d_unknown = 已观测 free 边界距离（裁剪到 [0, max_obs_dist]）。
        """
        # ---------------- 当前 lidar_scan（保留原 reward / collision 逻辑） ----------------
        curr_ray_hits_w = self.lidar.data.ray_hits_w.reshape(self.num_envs, self.num_lidar_points, 3)
        curr_sensor_pos_w = self.lidar.data.pos_w.reshape(self.num_envs, 3)

        curr_rel_w = curr_ray_hits_w - curr_sensor_pos_w[:, None, :]
        curr_dist = curr_rel_w.norm(dim=-1)
        curr_dist = torch.nan_to_num(
            curr_dist,
            nan=self.lidar_range,
            posinf=self.lidar_range,
            neginf=self.lidar_range,
        ).clamp(0.0, self.lidar_range)

        lidar_scan = self.lidar_range - curr_dist.reshape(self.num_envs, 1, self.num_lidar_points)

        # ---------------- 当前机体位姿 ----------------
        curr_pos_w = self.drone.pos.squeeze(1)                  # [E, 3]
        curr_quat_w = self.drone_state[..., 3:7].squeeze(1)    # [E, 4]

        obstacle_dist_bin = torch.full(
            (self.num_envs, self.downsampled_dim),
            self.max_obs_dist,
            device=self.device,
        )

        # [E, P, 3]
        ray_dirs_local = self.ray_dirs_local.unsqueeze(0).expand(self.num_envs, -1, -1)

        # ---------------- 融合最近 k 帧（仅障碍点） ----------------
        for h in range(self.k_hist):
            if not self.hist_valid[h].any():
                continue

            # ---------- A. 障碍点：最近障碍距离 ----------
            hist_hits_w = self.hist_ray_hits_w[h]  # [E, P, 3]
            hist_hits_body = self._transform_points_w_to_body(
                hist_hits_w, curr_pos_w, curr_quat_w
            )
            hist_obstacle_dist = self._scatter_points_to_bins(hist_hits_body, reduce="amin")
            obstacle_dist_bin = torch.minimum(obstacle_dist_bin, hist_obstacle_dist)

        # ---------- B. FoV free 边界：仅用当前帧 ----------
        # 当前帧每条 ray 的终点在 body frame 下 = ray_dirs_local * dist
        # - hit ray: dist = 命中距离，表示该方向 free 到障碍处
        # - miss ray: dist = max_obs_dist，表示该方向已确认 free 到最远
        # - 没有 ray 覆盖的 bin: d_unknown = 0 → 编码为 20（完全未知）
        curr_ray_dist = curr_rel_w.norm(dim=-1)  # [E, P]，复用已有的世界系相对向量
        curr_ray_dist = torch.nan_to_num(curr_ray_dist, nan=self.max_obs_dist,
                                          posinf=self.max_obs_dist,
                                          neginf=self.max_obs_dist)
        curr_ray_dist = curr_ray_dist.clamp(0.0, self.max_obs_dist)

        # 直接用本体系 ray 方向 × 距离构造终点，无需从世界系旋转
        curr_ray_endpoints_body = ray_dirs_local * curr_ray_dist.unsqueeze(-1)  # [E, P, 3]

        # scatter 到 bin 中取 max，得到每个 bin 方向上已确认 free 的最远距离
        observed_free_dist_bin = self._scatter_points_to_bins(curr_ray_endpoints_body, reduce="amax")

        # ---------------- 按论文风格编码 ----------------
        has_obstacle = obstacle_dist_bin < (self.max_obs_dist - 1e-6)

        d_unknown = observed_free_dist_bin.clamp(0.0, self.max_obs_dist)

        encoded_lidar = torch.where(
            has_obstacle,
            obstacle_dist_bin,      # [0, max_obs_dist]
            20.0 - d_unknown        # [10, 20] if max_obs_dist=10
        )

        encoded_lidar = encoded_lidar.unsqueeze(1)  # [E, 1, downsampled_dim]
        return encoded_lidar, lidar_scan
    # --------------------------------------------------------------------- #
    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        self.rpos = self.target_pos - self.drone_state[..., :3]

        # 只在新雷达帧到来时重新编码；其余时刻直接复用缓存
        # if self.lidar_dirty:
        #     encoded_lidar, lidar_scan = self._encode_lidar_observation()
        #     self.encoded_lidar_cache.copy_(encoded_lidar)
        #     self.lidar_scan_cache.copy_(lidar_scan)
        #     self.lidar_dirty = False
        if self.lidar_dirty:
            # 先把当前 lidar 帧压入历史缓存，再基于最近 k 帧统一编码
            self._push_current_lidar_frame_to_history()

            encoded_lidar, lidar_scan = self._encode_lidar_observation()
            self.encoded_lidar_cache.copy_(encoded_lidar)
            self.lidar_scan_cache.copy_(lidar_scan)
            self.lidar_dirty = False

        encoded_lidar = self.encoded_lidar_cache
        self.lidar_scan = self.lidar_scan_cache

        # ---------------- 14D proprioception ----------------
        z_pos = self.drone.pos[..., 2:3]

        rpos_xy = self.rpos.clone()
        rpos_xy[..., 2] = 0.0
        dist_xy = rpos_xy.norm(dim=-1, keepdim=True)
        rpos_xy_clipped = rpos_xy[..., :2] / dist_xy.clamp(1e-6)

        quat = self.drone_state[..., 3:7]
        v_world = self.drone.vel_w[..., :3]
        v_body = quat_rotate_inverse(quat, v_world)

        attitude = quat
        last_act = self.last_actions

        state = torch.cat([
            rpos_xy_clipped,  # [2]
            z_pos,            # [1]
            v_body,           # [3]
            attitude,         # [4]
            last_act          # [action_dim]
        ], dim=-1)
        if self.observe_vlim:
            denom = max(self.vlim_train_max - self.vlim_train_min, 1e-6)
            vlim_norm = ((self.vlim_episode - self.vlim_train_min) / denom).clamp(0.0, 1.0)
            state = torch.cat([state, vlim_norm], dim=-1)

        lidar_flat = encoded_lidar.flatten(start_dim=2)
        obs = torch.cat([state, lidar_flat], dim=-1)

        # ================= 🛡️ 核心防线 2：观察值截断 =================
        # 防止任何不可预见的除 0 错误导致的 Inf
        obs = torch.clamp(obs, min=-10000.0, max=10000.0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1000.0, neginf=-1000.0)
        # =========================================================

        
        if self._should_render(0) and set_camera_view is not None:
            self.debug_draw.clear()

            if bool(self.cfg.task.get("follow_camera", False)):
                eye_offset = torch.as_tensor(
                    self.cfg.task.get("follow_camera_eye_offset", [6.0, 0.0, 2.0]),
                    dtype=torch.float32,
                    device=self.device,
                )
                lookat_offset = torch.as_tensor(
                    self.cfg.task.get("follow_camera_lookat_offset", [2.0, 0.0, 0.5]),
                    dtype=torch.float32,
                    device=self.device,
                )

                drone_pos, _ = self.drone.get_world_poses(clone=True)
                if drone_pos.ndim >= 3:
                    drone_pos = drone_pos[0, 0]
                else:
                    drone_pos = drone_pos[0]

                if hasattr(self.drone, "heading"):
                    heading = self.drone.heading
                    if heading.ndim >= 3:
                        forward = heading[0, 0, :3]
                    elif heading.ndim == 2:
                        forward = heading[0, :3]
                    else:
                        forward = heading[:3]
                else:
                    forward = torch.tensor([1.0, 0.0, 0.0], device=self.device)

                forward = forward / torch.linalg.norm(forward).clamp_min(1e-6)
                up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                right = torch.cross(up, forward, dim=0)
                right = right / torch.linalg.norm(right).clamp_min(1e-6)

                eye = (
                    drone_pos
                    - forward * eye_offset[0]
                    + right * eye_offset[1]
                    + up * eye_offset[2]
                )
                target = (
                    drone_pos
                    + forward * lookat_offset[0]
                    + right * lookat_offset[1]
                    + up * lookat_offset[2]
                )
                eye = eye.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                set_camera_view(eye=eye, target=target)
            else:
                # Preserve the original fixed top-down camera when follow camera is disabled.
                center_xyz_cfg = self.cfg.task.get("topdown_center_xyz", [0.0, 0.0, 0.0])
                center = torch.as_tensor(center_xyz_cfg, dtype=torch.float32)
                camera_z = float(self.cfg.task.get("topdown_camera_z", self.cfg.task.get("topdown_view_height", 50.0)))
                lookat_z = float(self.cfg.task.get("topdown_lookat_z", 0.0))

                eye = torch.tensor([center[0].item(), center[1].item(), camera_z], dtype=torch.float32)
                target = torch.tensor([center[0].item(), center[1].item(), lookat_z], dtype=torch.float32)
                set_camera_view(eye=eye, target=target)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics
                },
                "stats": self.stats.clone()
            },
            self.batch_size,
        )

    #--------------------------------------------------------------------- #
    def _compute_reward_and_done(self):
        v = self.drone.vel_w[..., :3]
        v_norm = v.norm(dim=-1)
        vel_direction = v / v_norm.unsqueeze(-1).clamp_min(1e-6)

        actual_dists = self.lidar_range - self.lidar_scan
        d = actual_dists.amin(dim=2)

        curr_pos = self.drone.pos
        curr_dist = (self.target_pos - curr_pos).norm(dim=-1)
        prev_dist = (self.target_pos - self.prev_pos).norm(dim=-1)

        r_forward = prev_dist - curr_dist
        distance = curr_dist

        # omega = self.drone.vel_w[..., 3:]
        # action_diff = (self.current_actions - self.last_actions).norm(dim=-1)
        # # r_smoothness = omega.norm(dim=-1) + action_diff
        # r_smoothness = torch.clamp(omega.norm(dim=-1) + action_diff, max=50.0)
        
        # # 1. 动作平滑度 (Action Rate Penalty)
        # # 使用平方惩罚，容忍微调，重拳出击高频抖动
        # action_diff = self.current_actions - self.last_actions
        # r_smoothness = torch.sum(torch.square(action_diff), dim=-1)
        omega = self.drone.vel_w[..., 3:]                  # [ωx, ωy, ωz]
        if self.control_mode == "velocity":
            speed_ref = self.vlim_episode.squeeze(-1).clamp_min(1e-6)
            target_vel_diff = (self.current_target_vel - self.last_target_vel).norm(dim=-1) / speed_ref
            r_smoothness = omega.norm(dim=-1) + target_vel_diff
        else:
            action_diff = self.current_actions - self.last_actions
            r_smoothness = omega.norm(dim=-1) + action_diff.norm(dim=-1)
       
        # # r_max_speed = torch.exp(torch.relu(v_norm - self.v_max)) - 1.0
        # speed_excess = torch.relu(v_norm - self.v_max)
        # # r_max_speed = torch.exp(torch.clamp(speed_excess, max=5.0)) - 1.0
        # # 3. 修改超速惩罚：【绝对不要用 exp】！改用二次方(平方)，温柔且有效
        speed_limit = self.vlim_episode.squeeze(-1).clamp_min(1e-6)
        speed_excess = torch.relu(v_norm - speed_limit)
        r_max_speed = torch.square(speed_excess)

        # ---- action_diff is always needed by stats below ----
        action_diff = self.current_actions - self.last_actions
        # speed_excess = torch.relu(v_norm - self.v_max)
        # r_max_speed = 1.0 - torch.exp(speed_excess)
        # z = self.drone.pos[..., 2]
        # # z_penalty = torch.relu(z - self.z_max) + torch.relu(self.z_min - z)
        # # r_z = z_penalty
        # # r_z = torch.abs(z - 2.0)  # 直接鼓励它保持在 2 米高度，偏离越多惩罚越大
        # z_err = torch.abs(z - 2.0)
        # r_z = torch.relu(z_err - 0.3)
        # z_err = torch.abs(z - 2.0)
        # r_z = z_err
        # r_z_hard = (z > 3.0).float() * (z - 3.0) * 10.0 + (z < 1.0).float() * (1.0 - z) * 10.0
        z = self.drone.pos[..., 2]
        r_z = torch.maximum(
            torch.maximum(z - self.z_max, self.z_min - z),
            torch.zeros_like(z)
        )
# 1. 计算当前总推力 t
        t = self.drone.thrusts[..., 2].sum(dim=-1) 
        t_val = t.view(-1, 1)  # 强行拍成 (150, 1)

        # # 2. 计算悬停所需的重力 g_force
        # 注意这里改成了 masses (带 s)
        g_force = self.drone.masses * 9.81 
        g_val = g_force.view(-1, 1)  # 强行拍成 (150, 1)，抹平 [150, 1, 1] 带来的多余维度

        # 3. 计算推力误差
        r_thrust = torch.abs(t_val - g_val)

        # 计算误差
        r_thrust = torch.abs(t_val - g_val)

        #r_esdf = -self.lambda_esdf * torch.exp(-self.k_esdf * (d ** 2))
        r_esdf = self.lambda_esdf * (1.0 - torch.exp(-self.k_esdf * (d ** 2)))
        is_collision = d < self.collision_dist
        if self.reset_on_collision:
            contact_force = self.drone.base_link.get_net_contact_forces()
            collision_force = contact_force.norm(dim=-1)
            is_contact_collision = (collision_force > self.collision_force_threshold).any(-1, keepdim=True)
        else:
            is_contact_collision = torch.zeros_like(is_collision, dtype=torch.bool)

        reached_goal = distance < self.goal_radius
        r_goal = reached_goal.float() *  self.goal_bonus
        collision_mask = (is_collision | is_contact_collision).float()
        r_collision = -self.collision_speed_k * v_norm * collision_mask

        if self.num_milestones > 0:
            progress_ratio = ((self.init_goal_dist - curr_dist) / self.init_goal_dist).clamp(0.0, 1.0)
            stage_reward_parts = []
            stage_reward_total = torch.zeros_like(curr_dist)
            for i in range(self.num_milestones):
                just_hit = torch.logical_and(~self.milestone_hit[:, i : i + 1], progress_ratio >= self.milestone_thresholds[i])
                self.milestone_hit[:, i : i + 1] = torch.logical_or(self.milestone_hit[:, i : i + 1], just_hit)
                stage_reward_i = just_hit.float() * self.milestone_rewards[i]
                stage_reward_parts.append(stage_reward_i)
                stage_reward_total = stage_reward_total + stage_reward_i
        else:
            stage_reward_parts = []
            stage_reward_total = torch.zeros_like(curr_dist)

        x_pos = self.drone.pos[..., 0]
        y_pos = self.drone.pos[..., 1]
        # ================= 翻滚/倾角过大检测 =================
        quat_curr = self.drone_state[..., 3:7]

        from omni_drones.utils.torch import quat_axis
        up_vector = quat_axis(quat_curr, axis=2)

        tilt_cos = up_vector[..., 2]

        cos_threshold = math.cos(self.flip_tilt_deg * math.pi / 180.0)

        flip_now = (tilt_cos < cos_threshold)
        # 连续剧烈翻滚才判死，减少单帧冲击/抖动误杀。
        self.flip_counter = torch.where(
            flip_now,
            self.flip_counter + 1,
            torch.zeros_like(self.flip_counter),
        )
        flip_early = self.flip_counter >= self.flip_consecutive_steps
        # ===================================================
        # 2. 定义空气墙的范围
        # 因为你的目标在 y=24，出生在 y=-24，可以给个适当的余量，比如 [-30, 30]
        out_of_bounds = (torch.abs(x_pos) > 20.0) | (torch.abs(y_pos) > 30.0)
        misbehave = (
            (z < self.terminate_z_min) |
            (z > self.terminate_z_max) |
            (v_norm > self.terminate_v_norm) |
            is_collision |
            is_contact_collision |
            out_of_bounds|
            flip_early  # 加入翻滚判定
        )
        # 核心改动 1：增加死亡惩罚！死一次扣 1000 分，让它不敢轻易死
        r_death = misbehave.float() * -1000#-1000             
        x_body = self.drone.heading[..., :3]
        # r_yaw = (x_body * vel_direction).sum(-1)

        # 把它改成这样，强迫机头看向目标
        # target_dir = (self.target_pos - curr_pos)[..., :3]
        # target_dir = target_dir / target_dir.norm(dim=-1).unsqueeze(-1).clamp_min(1e-6)
        # x_body = self.drone.heading[..., :3]
        # r_yaw = (x_body * target_dir).sum(-1)
        # speed_mask = (v_norm > 0.5).float()
        # move_dir = v / v_norm.unsqueeze(-1).clamp_min(1e-6)
        # x_body = self.drone.heading[..., :3]
        # r_yaw = ((x_body * move_dir).sum(-1)) * speed_mask
        
        # yaw reward: 同时鼓励机头顺着飞行方向，也鼓励机头朝向目标
        # speed_mask = (v_norm > 0.5).float()
        # move_dir = v / v_norm.unsqueeze(-1).clamp_min(1e-6)

        # goal_vec = self.target_pos - curr_pos
        # goal_dir = goal_vec / goal_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # x_body = self.drone.heading[..., :3]

        # align_vel = ((x_body * move_dir).sum(-1)) * speed_mask
        # align_goal = (x_body * goal_dir).sum(-1)

        # r_yaw = 0.4 * align_vel + 0.6 * align_goal
        move_dir = v / v_norm.unsqueeze(-1).clamp_min(1e-6)
        x_body = self.drone.heading[..., :3]
        r_yaw = (x_body * move_dir).sum(-1)      
        reward = (
            self.w_forward * r_forward +
            self.w_smooth * r_smoothness +
            self.w_max_speed * r_max_speed +
            self.w_z * r_z +
            self.w_esdf * r_esdf +
            r_collision +
            r_death +
            self.w_yaw * r_yaw +
            self.w_thrust * r_thrust+ 
            r_goal +
            stage_reward_total
            
        )
        # ================= 🛡️ 核心防线 =================
        # 必须在这里！立刻！马上！硬截断！
        # 绝不能让异常值流到下一行！
        reward = torch.nan_to_num(reward, nan=-10000.0, posinf=10000.0, neginf=-10000.0)
        reward = torch.clamp(reward, min=-10000.0, max=10000.0)
        # ===============================================
        reward_scale = 0.2 #0.2
        reward = reward * reward_scale
        
        # ================= 📊 新增：累加各项奖励到 stats =================
        # 注意：如果 r_forward 等变量是 (num_envs,) 形状，
        # 需要加上 .unsqueeze(-1) 变成 (num_envs, 1) 才能正确相加
        self.stats["reward_forward"].add_(reward_scale * self.w_forward * r_forward.view(-1, 1))
        self.stats["reward_smooth"].add_(reward_scale * self.w_smooth * r_smoothness.view(-1, 1))
        self.stats["reward_max_speed"].add_(reward_scale * self.w_max_speed * r_max_speed.view(-1, 1))
        self.stats["reward_z"].add_(reward_scale * self.w_z * r_z.view(-1, 1))
        self.stats["reward_esdf"].add_(reward_scale * self.w_esdf * r_esdf.view(-1, 1))
        self.stats["reward_collision"].add_(reward_scale * r_collision.view(-1, 1))
        self.stats["reward_yaw"].add_(reward_scale * self.w_yaw * r_yaw.view(-1, 1))
        self.stats["reward_goal"].add_(reward_scale * r_goal.view(-1, 1))
        self.stats["reward_death"].add_(reward_scale * r_death.view(-1, 1))
        # 第 694 行：此时 r_thrust 已经是严谨的 (150, 1) 了，直接计算即可
        self.stats["reward_thrust"].add_(reward_scale * self.w_thrust * r_thrust)
        self.stats["reward_milestone"].add_(reward_scale * stage_reward_total.view(-1, 1))
        for i, stage_reward_i in enumerate(stage_reward_parts):
            self.stats[f"reward_milestone_{i + 1}"].add_(reward_scale * stage_reward_i.view(-1, 1))
        # ===============================================================

        # 更新已有的基础统计数据 (记得删掉之前重复的 self.stats["return"] += reward)
        # self.stats["return"].add_(reward.view(-1, 1))
        self.stats["safety"].add_(r_esdf.view(-1, 1))
        self.stats["action_smoothness"].add_(action_diff.norm(dim=-1))
        
        
        # 然后再更新统计数据
        self.stats["return"] += reward
        # 👇 在这里加入动作饱和率的累加
        # 计算当前这一步 4 个电机的动作饱和度均值，然后累加到总和中
        step_sat_rate = (self.current_actions.abs() > 0.95).float().mean(dim=-1)
        self.stats["action_sat"].add_(step_sat_rate.view(-1, 1))
        # ===============================================================

        hasnan = torch.isnan(self.drone_state).any(-1)

        death_flags = {
            "death_z_low": z < self.terminate_z_min,
            "death_z_high": z > self.terminate_z_max,
            "death_overspeed": v_norm > self.terminate_v_norm,
            "death_collision": is_collision,
            "death_contact": is_contact_collision,
            "death_oob": out_of_bounds,
            "death_flip": flip_early,
            "death_nan": hasnan,
        }
        for key, flag in death_flags.items():
            self.stats[key] = torch.maximum(self.stats[key], flag.float().view(-1, 1))

        terminated = misbehave | hasnan | reached_goal
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # self.stats["safety"].add_(r_esdf)
        # self.stats["action_smoothness"].add_(action_diff)
        # self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["success"] = torch.maximum(
            self.stats["success"],
            reached_goal.float()
        )

        self.prev_pos.copy_(curr_pos.detach())
        # ================= 💡 核心修改：计算平均饱和度 =================
        # 1. 克隆一份当前的统计数据，准备发给日志系统
        stats_out = self.stats.clone()
        
        # 2. 将累加的饱和次数除以当前存活的步数 (progress_buf)
        # clamp(min=1) 是为了防止除以 0（虽然第一步结束 progress_buf 就是 1 了，但防患于未然）
        stats_out["action_sat"] = stats_out["action_sat"] / self.progress_buf.unsqueeze(1).clamp(min=1)
        # ==============================================================
        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
                # 👇 加上这一行！用包含死亡惩罚的最新状态，覆盖掉之前的旧记录
                "stats": stats_out,
            },
            self.batch_size,
        )
