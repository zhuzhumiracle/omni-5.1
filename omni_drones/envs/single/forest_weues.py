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
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite

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
    num_samples: int = 4000  # 先默认低一点，跑稳后再往上加


# ------------------------------ Environment -------------------------------- #
class forest_weues(IsaacEnv):
    def __init__(self, cfg, headless):
        # ---------------- basic cfg ----------------
        self.lidar_update_interval = int(cfg.task.get("lidar_update_interval", 5))  # 50Hz -> 10Hz
        self.k_hist = int(cfg.task.get("k_hist", 5))

        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()

        self.v_max = float(cfg.task.get("v_max", 3.0))
        self.z_min = float(cfg.task.get("z_min", 0.5))
        self.z_max = float(cfg.task.get("z_max", 3.5))
        self.lambda_esdf = float(cfg.task.get("lambda_esdf", 1.0))
        self.k_esdf = float(cfg.task.get("k_esdf", 2.0))
        self.collision_dist = float(cfg.task.get("collision_dist", 0.3))
        self.goal_radius = float(cfg.task.get("goal_radius", 2.0))
        self.goal_bonus = float(cfg.task.get("goal_bonus", 1000.0))
        self.collision_penalty = float(cfg.task.get("collision_penalty", -20.0))
        self.reset_on_collision = bool(cfg.task.get("reset_on_collision", True))
        self.collision_force_threshold = float(cfg.task.get("collision_force_threshold", 1.0))

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

        # 这里改成可配置；先默认 4000，稳定后再升到 20000
        self.num_lidar_points = int(cfg.task.get("num_lidar_points", 4000))
        self.num_yaw_bins = int(cfg.task.get("num_yaw_bins", 80))
        self.num_pitch_bins = int(cfg.task.get("num_pitch_bins", 40))
        self.downsampled_dim = self.num_yaw_bins * self.num_pitch_bins

        # 全局步数
        self.global_step = 0

        super().__init__(cfg, headless)

        self.prev_pos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.steps_since_reset = torch.zeros((self.num_envs, 1, 1), dtype=torch.int32, device=self.device)

        self.lidar._initialize_impl()
        self.lidar_resolution = (self.num_lidar_points, 1)
        self.drone.initialize(track_contact_forces=self.reset_on_collision)

        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.last_actions = torch.zeros(self.num_envs, 1, self.action_dim, device=self.device)
        self.current_actions = torch.zeros_like(self.last_actions)

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = 24.
            self.target_pos[:, 0, 2] = 2.

        pitch_bin_centers = torch.linspace(-90.0, 90.0, self.num_pitch_bins, device=self.device)
        self.fov_mask = (pitch_bin_centers >= -7.0) & (pitch_bin_centers <= 52.0)
        self.fov_mask = self.fov_mask.unsqueeze(1).repeat(1, self.num_yaw_bins).reshape(1, 1, -1)
        self.fov_mask = self.fov_mask.expand(self.num_envs, 1, self.downsampled_dim)

        # 历史 seen mask
        self.hist_seen_mask = torch.zeros(
            (self.k_hist, self.num_envs, 1, self.downsampled_dim),
            dtype=torch.bool,
            device=self.device
        )
        self.hist_head = 0

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
            "action_sat",  # <==== 加入这行！
        ]
        
        # 为每个追踪项初始化一个全为 0 的 Tensor
        for key in tracking_keys:
            self.stats[key] = torch.zeros(self.num_envs, 1, device=self.device)
        # ===============================================================

    # --------------------------------------------------------------------- #
    def _update_history_seen_mask(self):
        # 简化版：只标记当前 FoV 覆盖角度
        self.hist_seen_mask[self.hist_head].copy_(self.fov_mask.bool())
        self.hist_head = (self.hist_head + 1) % self.k_hist

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
        self.action_dim = self.drone.action_spec.shape[-1]
        

        lidar_dim = self.downsampled_dim
        obs_dim = 14 + lidar_dim  # 2 + 1 + 3 + 4 + 4 + 3200

        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Unbounded((1, obs_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = Composite({
            "agents": Composite({
                "action": self.drone.action_spec.unsqueeze(0),
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

        
        stats_spec = Composite({
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

        }).expand(self.num_envs).to(self.device)
        
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()
        
        # ================= 🌟 引入 LLM 动态奖励 =================
        self.use_llm_reward = False  # 初始默认关闭，用自带的打底
        self.llm_reward_module = None # 存放 import 进来的临时模块
        self._llm_reward_warned = False
        # ========================================================

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
        self.steps_since_reset[env_ids] = 0
        self.hist_seen_mask[:, env_ids] = False

        # reset 缓存，避免用到旧 episode 数据
        self.encoded_lidar_cache[env_ids] = self.max_obs_dist
        self.lidar_scan_cache[env_ids] = 0.0
        self.lidar_dirty = True

    # --------------------------------------------------------------------- #
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # ================= 🛡️ 核心防线 1：动作截断 =================
        # 无论神经网络发什么疯，传给物理引擎的动作绝对不能超过合理范围（比如 [-1.0, 1.0]）
        actions = torch.clamp(actions, min=-1.0, max=1.0)
            # 关键：把真正执行的动作写回 tensordict
        
        # 过滤 NaN，如果网络输出了 NaN，强制归零
        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        # =========================================================
        tensordict[("agents", "action")] = actions
        # 先保存上一时刻动作，再写当前动作
        self.last_actions.copy_(self.current_actions)

        self.effort = self.drone.apply_action(actions)
        self.current_actions = actions.clone()

    # --------------------------------------------------------------------- #
    def _post_sim_step(self, tensordict: TensorDictBase):
        self.global_step += 1
        self.steps_since_reset += 1

        if self.global_step % self.lidar_update_interval == 0:
            self.lidar.update(self.dt * self.lidar_update_interval)
            self._update_history_seen_mask()
            self.lidar_dirty = True

    # --------------------------------------------------------------------- #
    def _encode_lidar_observation(self):
        # 1) 原始扫描缓存
        lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, self.num_lidar_points)
        )

        # 2) 点云转相对坐标
        rel_points = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
        rel_points = rel_points.reshape(self.num_envs, self.num_lidar_points, 3)
        point_dist = rel_points.norm(dim=-1)

        valid_mask = point_dist <= self.max_obs_dist

        env_ids = (
            torch.arange(self.num_envs, device=self.device)
            .unsqueeze(1)
            .expand(self.num_envs, self.num_lidar_points)
        )

        valid_env = env_ids[valid_mask]
        valid_pts = rel_points[valid_mask]
        valid_dist = point_dist[valid_mask]

        max_dist = self.max_obs_dist
        downsampled_scan = torch.full(
            (self.num_envs, self.downsampled_dim),
            max_dist,
            device=self.device,
            dtype=lidar_scan.dtype,
        )

        if valid_pts.numel() > 0:
            voxel_idx = torch.floor(valid_pts / self.voxel_size).to(torch.int64)

            vx = voxel_idx[:, 0] + self._voxel_offset
            vy = voxel_idx[:, 1] + self._voxel_offset
            vz = voxel_idx[:, 2] + self._voxel_offset

            inside = (
                (vx >= 0) & (vx < self._voxel_base) &
                (vy >= 0) & (vy < self._voxel_base) &
                (vz >= 0) & (vz < self._voxel_base)
            )

            valid_env = valid_env[inside]
            valid_dist = valid_dist[inside]
            vx = vx[inside]
            vy = vy[inside]
            vz = vz[inside]

            if valid_env.numel() > 0:
                key = (((valid_env * self._voxel_base + vx) * self._voxel_base + vy) * self._voxel_base + vz)
                uniq_key, inv = torch.unique(key, return_inverse=True)

                voxel_min_dist = torch.full(
                    (uniq_key.numel(),),
                    max_dist,
                    device=self.device,
                    dtype=valid_dist.dtype,
                )
                voxel_min_dist.scatter_reduce_(0, inv, valid_dist, reduce="amin", include_self=True)

                tmp = uniq_key
                uz = tmp % self._voxel_base
                tmp = torch.div(tmp, self._voxel_base, rounding_mode="floor")
                uy = tmp % self._voxel_base
                tmp = torch.div(tmp, self._voxel_base, rounding_mode="floor")
                ux = tmp % self._voxel_base
                uenv = torch.div(tmp, self._voxel_base, rounding_mode="floor")

                cx = (ux - self._voxel_offset + 0.5).to(torch.float32) * self.voxel_size
                cy = (uy - self._voxel_offset + 0.5).to(torch.float32) * self.voxel_size
                cz = (uz - self._voxel_offset + 0.5).to(torch.float32) * self.voxel_size

                yaw = torch.atan2(cy, cx)
                yaw = torch.remainder(yaw, 2 * torch.pi)
                pitch = torch.atan2(cz, torch.sqrt(cx * cx + cy * cy + 1e-6))

                yaw_idx = torch.clamp(
                    (yaw / (2 * torch.pi) * self.num_yaw_bins).long(),
                    0, self.num_yaw_bins - 1
                )
                pitch_idx = torch.clamp(
                    (((pitch + torch.pi / 2) / torch.pi) * self.num_pitch_bins).long(),
                    0, self.num_pitch_bins - 1
                )

                flat_bin = pitch_idx * self.num_yaw_bins + yaw_idx
                out_key = uenv * self.downsampled_dim + flat_bin

                flat_out = downsampled_scan.reshape(-1)
                flat_out.scatter_reduce_(0, out_key, voxel_min_dist, reduce="amin", include_self=True)
                downsampled_scan = flat_out.view(self.num_envs, self.downsampled_dim)

        d_min = downsampled_scan.unsqueeze(1).clamp_max(self.max_obs_dist)

        seen_mask = self.hist_seen_mask.any(dim=0)
        has_point = d_min < self.max_obs_dist - 1e-6
        unknown_mask = (~has_point) & (~seen_mask)

        encoded_lidar = d_min.clone()
        encoded_lidar = torch.where(
            (~has_point) & seen_mask,
            torch.full_like(encoded_lidar, self.max_obs_dist),
            encoded_lidar
        )
        encoded_lidar = torch.where(
            unknown_mask,
            torch.full_like(encoded_lidar, 15.0),
            encoded_lidar
        )

        return encoded_lidar, lidar_scan

    # --------------------------------------------------------------------- #
    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        self.rpos = self.target_pos - self.drone_state[..., :3]

        # 只在新雷达帧到来时重新编码；其余时刻直接复用缓存
        if self.lidar_dirty:
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
            last_act          # [4]
        ], dim=-1)

        lidar_flat = encoded_lidar.flatten(start_dim=2)
        obs = torch.cat([state, lidar_flat], dim=-1)

        # ================= 🛡️ 核心防线 2：观察值截断 =================
        # 防止任何不可预见的除 0 错误导致的 Inf
        obs = torch.clamp(obs, min=-10000.0, max=10000.0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1000.0, neginf=-1000.0)
        # =========================================================

        
        if self._should_render(0) and set_camera_view is not None:
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

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

#     #--------------------------------------------------------------------- #
#     def _compute_reward_and_done(self):
#         v = self.drone.vel_w[..., :3]
#         v_norm = v.norm(dim=-1)
#         vel_direction = v / v_norm.unsqueeze(-1).clamp_min(1e-6)

#         actual_dists = self.lidar_range - self.lidar_scan
#         d = actual_dists.amin(dim=2)

#         curr_pos = self.drone.pos
#         curr_dist = (self.target_pos - curr_pos).norm(dim=-1)
#         prev_dist = (self.target_pos - self.prev_pos).norm(dim=-1)

#         r_forward = prev_dist - curr_dist
#         distance = curr_dist

#         omega = self.drone.vel_w[..., 3:]
#         action_diff = (self.current_actions - self.last_actions).norm(dim=-1)
#         # r_smoothness = omega.norm(dim=-1) + action_diff
#         r_smoothness = torch.clamp(omega.norm(dim=-1) + action_diff, max=50.0)

       
       
#         # r_max_speed = torch.exp(torch.relu(v_norm - self.v_max)) - 1.0
#         speed_excess = torch.relu(v_norm - self.v_max)
#         # r_max_speed = torch.exp(torch.clamp(speed_excess, max=5.0)) - 1.0
#         # 3. 修改超速惩罚：【绝对不要用 exp】！改用二次方(平方)，温柔且有效
#         speed_excess = torch.relu(v_norm - self.v_max)
#         r_max_speed = torch.square(speed_excess)
#         z = self.drone.pos[..., 2]
#         # z_penalty = torch.relu(z - self.z_max) + torch.relu(self.z_min - z)
#         # r_z = z_penalty
#         # r_z = torch.abs(z - 2.0)  # 直接鼓励它保持在 2 米高度，偏离越多惩罚越大
#         z_err = torch.abs(z - 2.0)
#         r_z = torch.relu(z_err - 0.3)
#         # z_err = torch.abs(z - 2.0)
#         # r_z = z_err
#         # r_z_hard = (z > 3.0).float() * (z - 3.0) * 10.0 + (z < 1.0).float() * (1.0 - z) * 10.0

# # 1. 计算当前总推力 t
#         t = self.drone.thrusts[..., 2].sum(dim=-1) 
#         t_val = t.view(-1, 1)  # 强行拍成 (150, 1)

#         # 2. 计算悬停所需的重力 g_force
#         # 注意这里改成了 masses (带 s)
#         g_force = self.drone.masses * 9.81 
#         g_val = g_force.view(-1, 1)  # 强行拍成 (150, 1)，抹平 [150, 1, 1] 带来的多余维度

#         # 3. 计算推力误差
#         r_thrust = torch.abs(t_val - g_val)

#         # 计算误差
#         r_thrust = torch.abs(t_val - g_val)

#         r_esdf = -self.lambda_esdf * torch.exp(-self.k_esdf * (d ** 2))

#         is_collision = d < self.collision_dist
#         if self.reset_on_collision:
#             contact_force = self.drone.base_link.get_net_contact_forces()
#             collision_force = contact_force.norm(dim=-1)
#             is_contact_collision = (collision_force > self.collision_force_threshold).any(-1, keepdim=True)
#         else:
#             is_contact_collision = torch.zeros_like(is_collision, dtype=torch.bool)

#         reached_goal = distance < self.goal_radius
#         r_goal = reached_goal.float() *  self.goal_bonus
#         r_collision = is_collision.float() * self.collision_penalty

#         x_pos = self.drone.pos[..., 0]
#         y_pos = self.drone.pos[..., 1]
#         # ================= 翻滚/倾角过大检测 =================
#         quat_curr = self.drone_state[..., 3:7]

#         from omni_drones.utils.torch import quat_axis
#         up_vector = quat_axis(quat_curr, axis=2)

#         tilt_cos = up_vector[..., 2]

#         cos_threshold = math.cos(80.0 * math.pi / 180.0)

#         flip_early = (tilt_cos < cos_threshold)   # 不要再 unsqueeze
#         # ===================================================
#         # 2. 定义空气墙的范围
#         # 因为你的目标在 y=24，出生在 y=-24，可以给个适当的余量，比如 [-30, 30]
#         out_of_bounds = (torch.abs(x_pos) > 20.0) | (torch.abs(y_pos) > 30.0)
#         misbehave = (
#             (z < 0.4) |
#             (z < self.terminate_z_min) |
#             (z > self.terminate_z_max) |
#             (v_norm > self.terminate_v_norm) |
#             is_collision |
#             is_contact_collision |
#             out_of_bounds|
#             flip_early  # 加入翻滚判定
#         )
#         # 核心改动 1：增加死亡惩罚！死一次扣 1000 分，让它不敢轻易死
#         r_death = misbehave.float() * -100.0             
#         x_body = self.drone.heading[..., :3]
#         # r_yaw = (x_body * vel_direction).sum(-1)

#         # 把它改成这样，强迫机头看向目标
#         target_dir = (self.target_pos - curr_pos)[..., :3]
#         target_dir = target_dir / target_dir.norm(dim=-1).unsqueeze(-1).clamp_min(1e-6)
#         x_body = self.drone.heading[..., :3]
#         r_yaw = (x_body * target_dir).sum(-1)
#         speed_mask = (v_norm > 0.5).float()
#         move_dir = v / v_norm.unsqueeze(-1).clamp_min(1e-6)
#         x_body = self.drone.heading[..., :3]
#         r_yaw = ((x_body * move_dir).sum(-1)) * speed_mask
#         # yaw reward: 同时鼓励机头顺着飞行方向，也鼓励机头朝向目标
#         # speed_mask = (v_norm > 0.5).float()
#         # move_dir = v / v_norm.unsqueeze(-1).clamp_min(1e-6)

#         # goal_vec = self.target_pos - curr_pos
#         # goal_dir = goal_vec / goal_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)

#         # x_body = self.drone.heading[..., :3]

#         # align_vel = ((x_body * move_dir).sum(-1)) * speed_mask
#         # align_goal = (x_body * goal_dir).sum(-1)

#         # r_yaw = 0.4 * align_vel + 0.6 * align_goal
                
#         reward = (
#             self.w_forward * r_forward +
#             self.w_smooth * r_smoothness +
#             self.w_max_speed * r_max_speed +
#             self.w_z * r_z +
#             self.w_esdf * r_esdf +
#             r_collision +
#             r_death +
#             self.w_yaw * r_yaw +
#             # self.w_thrust * r_thrust +
#             r_goal  
            
#         )
#         # ================= 🛡️ 核心防线 =================
#         # 必须在这里！立刻！马上！硬截断！
#         # 绝不能让异常值流到下一行！
#         reward = torch.nan_to_num(reward, nan=-10000.0, posinf=10000.0, neginf=-10000.0)
#         reward = torch.clamp(reward, min=-10000.0, max=10000.0)
#         # ===============================================
#         reward_scale = 0.2
#         reward = reward * reward_scale
        
#         # ================= 📊 新增：累加各项奖励到 stats =================
#         # 注意：如果 r_forward 等变量是 (num_envs,) 形状，
#         # 需要加上 .unsqueeze(-1) 变成 (num_envs, 1) 才能正确相加
#         self.stats["reward_forward"].add_(reward_scale * self.w_forward * r_forward.view(-1, 1))
#         self.stats["reward_smooth"].add_(reward_scale * self.w_smooth * r_smoothness.view(-1, 1))
#         self.stats["reward_max_speed"].add_(reward_scale * self.w_max_speed * r_max_speed.view(-1, 1))
#         self.stats["reward_z"].add_(reward_scale * self.w_z * r_z.view(-1, 1))
#         self.stats["reward_esdf"].add_(reward_scale * self.w_esdf * r_esdf.view(-1, 1))
#         self.stats["reward_collision"].add_(reward_scale * r_collision.view(-1, 1))
#         self.stats["reward_yaw"].add_(reward_scale * self.w_yaw * r_yaw.view(-1, 1))
#         self.stats["reward_goal"].add_(reward_scale * r_goal.view(-1, 1))
#         self.stats["reward_death"].add_(reward_scale * r_death.view(-1, 1))
#         # 第 694 行：此时 r_thrust 已经是严谨的 (150, 1) 了，直接计算即可
#         self.stats["reward_thrust"].add_(reward_scale * self.w_thrust * r_thrust)
#         # ===============================================================

#         # 更新已有的基础统计数据 (记得删掉之前重复的 self.stats["return"] += reward)
#         # self.stats["return"].add_(reward.view(-1, 1))
#         self.stats["safety"].add_(r_esdf.view(-1, 1))
#         self.stats["action_smoothness"].add_(action_diff.view(-1, 1))
        
        
#         # 然后再更新统计数据
#         self.stats["return"] += reward
#         # 👇 在这里加入动作饱和率的累加
#         # 计算当前这一步 4 个电机的动作饱和度均值，然后累加到总和中
#         step_sat_rate = (self.current_actions.abs() > 0.95).float().mean(dim=-1)
#         self.stats["action_sat"].add_(step_sat_rate.view(-1, 1))
#         # ===============================================================

#         hasnan = torch.isnan(self.drone_state).any(-1)

#         terminated = misbehave | hasnan | reached_goal
#         truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

#         # self.stats["safety"].add_(r_esdf)
#         # self.stats["action_smoothness"].add_(action_diff)
#         # self.stats["return"] += reward
#         self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
#         self.stats["success"] = torch.maximum(
#             self.stats["success"],
#             reached_goal.float()
#         )

#         self.prev_pos.copy_(curr_pos.detach())
#         # ================= 💡 核心修改：计算平均饱和度 =================
#         # 1. 克隆一份当前的统计数据，准备发给日志系统
#         stats_out = self.stats.clone()
        
#         # 2. 将累加的饱和次数除以当前存活的步数 (progress_buf)
#         # clamp(min=1) 是为了防止除以 0（虽然第一步结束 progress_buf 就是 1 了，但防患于未然）
#         stats_out["action_sat"] = stats_out["action_sat"] / self.progress_buf.unsqueeze(1).clamp(min=1)
#         # ==============================================================
#         return TensorDict(
#             {
#                 "agents": {"reward": reward.unsqueeze(-1)},
#                 "done": terminated | truncated,
#                 "terminated": terminated,
#                 "truncated": truncated,
#                 # 👇 加上这一行！用包含死亡惩罚的最新状态，覆盖掉之前的旧记录
#                 "stats": stats_out,
#             },
#             self.batch_size,
#         )
# --------------------------------------------------------------------- #
    # 第一块：纯净的状态提取器 (LLM 的“眼睛”)
    # --------------------------------------------------------------------- #
    def get_clean_states(self):
        """
        把物理引擎里的复杂属性，提炼成大模型能看懂的纯净张量字典。
        这些 key 就是将来大模型函数的 input_dict。
        """
        v = self.drone.vel_w[..., :3]
        v_norm = v.norm(dim=-1)
        
        # 雷达特征
        actual_dists = self.lidar_range - self.lidar_scan
        min_lidar_dist = actual_dists.amin(dim=2) # 最近障碍物距离 [num_envs, 1]
        
        curr_pos = self.drone.pos
        curr_dist = (self.target_pos - curr_pos).norm(dim=-1)
        prev_dist = (self.target_pos - self.prev_pos).norm(dim=-1)
        
        omega = self.drone.vel_w[..., 3:]
        action_diff = (self.current_actions - self.last_actions).norm(dim=-1)
        
        # 翻滚判定（姿态）
        quat_curr = self.drone_state[..., 3:7]
        from omni_drones.utils.torch import quat_axis
        up_vector = quat_axis(quat_curr, axis=2)
        tilt_cos = up_vector[..., 2]
        
        x_body = self.drone.heading[..., :3]

        return {
            "curr_pos": curr_pos,             # 当前位置 [num_envs, 3]
            "target_pos": self.target_pos,    # 目标位置 [num_envs, 1, 3]
            "curr_dist": curr_dist,           # 距目标距离 [num_envs, 1]
            "prev_dist": prev_dist,           # 距目标上一帧距离
            "v": v,                           # 当前线速度矢量 [num_envs, 3]
            "v_norm": v_norm,                 # 速度大小
            "omega": omega,                   # 角速度
            "min_lidar_dist": min_lidar_dist, # 距离最近障碍物的距离 [num_envs, 1]
            "action_diff": action_diff,       # 动作变化率
            "tilt_cos": tilt_cos,             # 姿态仰角余弦值
            "z_pos": curr_pos[..., 2],        # 高度
            "x_body": x_body                  # 机头朝向
        }

    def _normalize_llm_states(self, states: dict):
        """Normalize state tensor shapes before passing to generated reward code."""
        vector_keys = {"curr_pos", "target_pos", "v", "omega", "x_body"}

        def _to_vector3(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t.view(1, 1)
            elif t.dim() == 1:
                t = t.unsqueeze(-1)
            elif t.dim() >= 2 and t.shape[-2] == 1:
                t = t.squeeze(-2)

            if t.dim() > 2:
                t = t.reshape(t.shape[0], -1)
            elif t.dim() == 1:
                t = t.view(t.shape[0], 1)

            if t.shape[-1] >= 3:
                return t[..., :3]

            pad = torch.zeros(t.shape[0], 3 - t.shape[-1], device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=-1)

        normalized = {}
        for key, value in states.items():
            if not torch.is_tensor(value):
                normalized[key] = value
                continue

            if key in vector_keys:
                normalized[key] = _to_vector3(value)
                continue

            t = value
            while t.dim() > 1 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            if t.dim() > 2 and t.shape[-2] == 1:
                t = t.squeeze(-2)
            if t.dim() > 1:
                t = t.reshape(t.shape[0], -1).mean(dim=-1)
            normalized[key] = t

        return normalized

    # --------------------------------------------------------------------- #
    # 第二块：融合 LLM 与原生逻辑的打分系统
    # --------------------------------------------------------------------- #
    def _compute_reward_and_done(self):
        # 1. 提取大模型看得懂的纯张量状态
        states = self.get_clean_states()
        
        # 展开常用变量，方便底层物理逻辑计算
        v_norm = states["v_norm"]
        d = states["min_lidar_dist"]
        curr_pos = states["curr_pos"]
        curr_dist = states["curr_dist"]
        z = states["z_pos"]

        # ================= 🛡️ 状态与死亡判定 (严格保留原版硬逻辑) =================
        is_collision = d < self.collision_dist
        if self.reset_on_collision:
            contact_force = self.drone.base_link.get_net_contact_forces()
            collision_force = contact_force.norm(dim=-1)
            is_contact_collision = (collision_force > self.collision_force_threshold).any(-1, keepdim=True)
        else:
            is_contact_collision = torch.zeros_like(is_collision, dtype=torch.bool)

        reached_goal = curr_dist < self.goal_radius
        x_pos, y_pos = curr_pos[..., 0], curr_pos[..., 1]
        
        out_of_bounds = (torch.abs(x_pos) > 20.0) | (torch.abs(y_pos) > 30.0)
        cos_threshold = math.cos(80.0 * math.pi / 180.0)
        flip_early = (states["tilt_cos"] < cos_threshold)

        misbehave = (
            (z < 0.4) | (z < self.terminate_z_min) | (z > self.terminate_z_max) |
            (v_norm > self.terminate_v_norm) |
            is_collision | is_contact_collision | out_of_bounds | flip_early
        )

        hasnan = torch.isnan(self.drone_state).any(-1)
        terminated = misbehave | hasnan | reached_goal
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # ================= 🔧 基础原版奖励计算 (即使不作为主奖励，也算出来监控) =================
        r_forward = states["prev_dist"] - curr_dist
        r_smoothness = torch.clamp(states["omega"].norm(dim=-1) + states["action_diff"], max=50.0)
        r_max_speed = torch.square(torch.relu(v_norm - self.v_max))
        r_z = torch.relu(torch.abs(z - 2.0) - 0.3)
        r_esdf = -self.lambda_esdf * torch.exp(-self.k_esdf * (d ** 2))
        
        t = self.drone.thrusts[..., 2].sum(dim=-1) 
        r_thrust = torch.abs(t.view(-1, 1) - (self.drone.masses * 9.81).view(-1, 1))
        
        r_goal = reached_goal.float() * self.goal_bonus
        r_collision = is_collision.float() * self.collision_penalty
        r_death = misbehave.float() * -100.0

        speed_mask = (v_norm > 0.5).float()
        move_dir = states["v"] / v_norm.unsqueeze(-1).clamp_min(1e-6)
        r_yaw = ((states["x_body"] * move_dir).sum(-1)) * speed_mask

        # 你的打底原生总分
        base_reward = (
            self.w_forward * r_forward + self.w_smooth * r_smoothness +
            self.w_max_speed * r_max_speed + self.w_z * r_z +
            self.w_esdf * r_esdf + r_collision + r_death +
            self.w_yaw * r_yaw + r_goal  
        ) * 0.2

        # ================= 🚀 LLM 动态奖励分支 =================
        # 注意：使用 getattr 以防 __init__ 里的开关还没定义好
        if getattr(self, "use_llm_reward", False) and getattr(self, "llm_reward_module", None) is not None:
            try:
                # 核心魔法：大模型接管奖励计算！
                llm_states = self._normalize_llm_states(states)
                reward = self.llm_reward_module.compute_reward(**llm_states)
            except Exception as e:
                if not self._llm_reward_warned:
                    shape_text = {
                        k: tuple(v.shape) for k, v in states.items() if torch.is_tensor(v)
                    }
                    print(f"[!] LLM Reward 运行时报错，回退到原生奖励: {e}")
                    print(f"[!] 当前输入张量形状: {shape_text}")
                    self._llm_reward_warned = True
                reward = base_reward
        else:
            reward = base_reward

        # ================= 🛡️ 强制防线（防 LLM 产生 Inf/NaN） =================
        reward = torch.nan_to_num(reward, nan=-10000.0, posinf=10000.0, neginf=-10000.0)
        reward = torch.clamp(reward, min=-10000.0, max=10000.0)

        # ================= 📊 WandB 监控录入 (保持原有结构) =================
        reward_scale = 0.2
        # 我们坚持把底层的分项指标记录下来，用于透视大模型策略的物理表现！
        self.stats["reward_forward"].add_(reward_scale * self.w_forward * r_forward.view(-1, 1))
        self.stats["reward_smooth"].add_(reward_scale * self.w_smooth * r_smoothness.view(-1, 1))
        self.stats["reward_max_speed"].add_(reward_scale * self.w_max_speed * r_max_speed.view(-1, 1))
        self.stats["reward_z"].add_(reward_scale * self.w_z * r_z.view(-1, 1))
        self.stats["reward_esdf"].add_(reward_scale * self.w_esdf * r_esdf.view(-1, 1))
        self.stats["reward_collision"].add_(reward_scale * r_collision.view(-1, 1))
        self.stats["reward_yaw"].add_(reward_scale * self.w_yaw * r_yaw.view(-1, 1))
        self.stats["reward_goal"].add_(reward_scale * r_goal.view(-1, 1))
        self.stats["reward_death"].add_(reward_scale * r_death.view(-1, 1))
        self.stats["reward_thrust"].add_(reward_scale * self.w_thrust * r_thrust.view(-1, 1))

        self.stats["safety"].add_(r_esdf.view(-1, 1))
        self.stats["action_smoothness"].add_(states["action_diff"].view(-1, 1))
        
        # 兼容 LLM 输出：保证 reward 一定能和 return 相加
        if reward.dim() == 1:
            self.stats["return"].add_(reward.view(-1, 1))
        else:
            self.stats["return"].add_(reward)

        step_sat_rate = (self.current_actions.abs() > 0.95).float().mean(dim=-1)
        self.stats["action_sat"].add_(step_sat_rate.view(-1, 1))
        
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["success"] = torch.maximum(self.stats["success"], reached_goal.float())

        self.prev_pos.copy_(curr_pos.detach())

        stats_out = self.stats.clone()
        stats_out["action_sat"] = stats_out["action_sat"] / self.progress_buf.unsqueeze(1).clamp(min=1)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1) if reward.dim() == 1 else reward},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
                "stats": stats_out,
            },
            self.batch_size,
        )