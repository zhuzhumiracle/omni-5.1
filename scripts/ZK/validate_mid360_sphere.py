import math
from dataclasses import dataclass
from typing import Callable

import hydra
import torch
import torch.distributions as D
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict, TensorDictBase
from omni_drones import init_simulation_app
from torchrl.data import Composite, Unbounded


def _build_mid360_env_class():
    # NOTE: All Isaac/Omni imports must be after SimulationApp initialization.
    import omni_drones.utils.kit as kit_utils
    import omni.usd
    from pxr import UsdGeom, Vt
    from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion
    from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg

    def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = cfg.num_samples
        t = torch.linspace(0, 0.1, num_samples, device=device)
        pi = torch.pi

        yaw_deg = (-62050.63 * t + 3.11 * torch.cos(314159.2 * t) * torch.sin(628.318 * 2 * t)) % 360
        pitch_deg = 22.5 + 29.5 * torch.cos(20 * pi * t) + 4 * torch.cos(2 * pi / 0.006 * t) * torch.cos(10000 * pi * t)

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
        num_samples: int = 20000

    class Mid360SphereCheck(IsaacEnv):
        @staticmethod
        def _create_uv_sphere_mesh(prim_path: str, center_xyz: torch.Tensor, radius: float, lat_steps: int = 20, lon_steps: int = 40) -> str:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("USD stage is not available.")

            mesh = UsdGeom.Mesh.Define(stage, prim_path)

            cx, cy, cz = [float(x) for x in center_xyz.tolist()]
            pts = []
            for i in range(lat_steps + 1):
                theta = math.pi * i / lat_steps
                sin_t = math.sin(theta)
                cos_t = math.cos(theta)
                for j in range(lon_steps):
                    phi = 2.0 * math.pi * j / lon_steps
                    x = cx + radius * sin_t * math.cos(phi)
                    y = cy + radius * sin_t * math.sin(phi)
                    z = cz + radius * cos_t
                    pts.append((x, y, z))

            face_vertex_counts = []
            face_vertex_indices = []
            cols = lon_steps
            for i in range(lat_steps):
                row0 = i * cols
                row1 = (i + 1) * cols
                for j in range(cols):
                    jn = (j + 1) % cols
                    v00 = row0 + j
                    v01 = row0 + jn
                    v10 = row1 + j
                    v11 = row1 + jn

                    face_vertex_counts.extend([3, 3])
                    face_vertex_indices.extend([v00, v10, v11])
                    face_vertex_indices.extend([v00, v11, v01])

            mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))

            return prim_path

        def __init__(self, cfg, headless):
            self.num_lidar_points = int(cfg.task.get("num_lidar_points", 20000))
            self.lidar_range = float(cfg.task.get("lidar_range", 10.0))
            self.sphere_radius = float(cfg.task.get("check_sphere_radius", 1.0))
            self.sphere_center = torch.tensor(cfg.task.get("check_sphere_center", [8.0, 0.0, 2.0]), dtype=torch.float32)

            super().__init__(cfg, headless)

            self.drone.initialize(track_contact_forces=False)
            self.action_dim = self.drone.action_spec.shape[-1]

            self.init_vels = torch.zeros_like(self.drone.get_velocities())
            self.init_rpy = torch.zeros((1, 3), device=self.device)

            self.latest_min_dist = torch.zeros((self.num_envs, 1), device=self.device)
            self.latest_finite_ratio = torch.zeros((self.num_envs, 1), device=self.device)

        def _design_scene(self):
            from isaaclab.sensors import RayCaster, RayCasterCfg

            self.drone, self.controller = MultirotorBase.make("Hummingbird", "LeePositionController")

            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

            self.drone.spawn(translations=[(0.0, 0.0, 2.0)])

            obstacle_mesh_prim_path = self._create_uv_sphere_mesh(
                prim_path="/World/envs/env_0/obstacle_sphere_mesh",
                center_xyz=self.sphere_center,
                radius=self.sphere_radius,
                lat_steps=20,
                lon_steps=40,
            )

            ray_caster_cfg = RayCasterCfg(
                prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                attach_yaw_only=False,
                pattern_cfg=Mid360PatternCfg(num_samples=self.num_lidar_points),
                debug_vis=False,
                # IsaacLab RayCaster currently supports exactly one mesh prim.
                mesh_prim_paths=[obstacle_mesh_prim_path],
            )
            self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)

            return ["/World/defaultGroundPlane"]

        def _set_specs(self):
            obs_dim = 2
            self.observation_spec = Composite(
                {
                    "agents": Composite({"observation": Unbounded((1, obs_dim), device=self.device)})
                }
            ).expand(self.num_envs).to(self.device)

            self.action_spec = Composite(
                {"agents": Composite({"action": self.drone.action_spec.unsqueeze(0)})}
            ).expand(self.num_envs).to(self.device)

            self.reward_spec = Composite(
                {"agents": Composite({"reward": Unbounded((1, 1), device=self.device)})}
            ).expand(self.num_envs).to(self.device)

            self.agent_spec["drone"] = AgentSpec(
                "drone",
                1,
                observation_key=("agents", "observation"),
                action_key=("agents", "action"),
                reward_key=("agents", "reward"),
            )

            self.observation_spec["stats"] = Composite(
                {
                    "min_lidar_dist": Unbounded(1),
                    "finite_ratio": Unbounded(1),
                }
            ).expand(self.num_envs).to(self.device)

            self.stats = self.observation_spec["stats"].zero()

        def _reset_idx(self, env_ids: torch.Tensor):
            self.drone._reset_idx(env_ids, self.training)

            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, :] = torch.tensor([0.0, 0.0, 2.0], device=self.device)
            rpy = self.init_rpy.expand(len(env_ids), 1, 3)
            rot = euler_to_quaternion(rpy)

            self.drone.set_world_poses(pos, rot, env_ids)
            self.drone.set_velocities(self.init_vels[env_ids], env_ids)

            self.stats[env_ids] = 0.0

        def _pre_sim_step(self, tensordict: TensorDictBase):
            actions = tensordict[("agents", "action")]
            actions = torch.tanh(actions)
            self.drone.apply_action(actions)

        def _post_sim_step(self, tensordict: TensorDictBase):
            self.lidar.update(self.dt)

        def _compute_state_and_obs(self):
            ray_hits = self.lidar.data.ray_hits_w.reshape(self.num_envs, -1, 3)
            lidar_pos = self.lidar.data.pos_w.unsqueeze(1)
            dist = (ray_hits - lidar_pos).norm(dim=-1)
            dist = torch.nan_to_num(dist, nan=self.lidar_range, posinf=self.lidar_range, neginf=0.0)
            dist = torch.clamp(dist, min=0.0, max=self.lidar_range)

            finite_ratio = torch.isfinite(ray_hits).all(dim=-1).float().mean(dim=-1, keepdim=True)
            min_dist = dist.min(dim=-1, keepdim=True).values

            self.latest_min_dist.copy_(min_dist)
            self.latest_finite_ratio.copy_(finite_ratio)
            self.stats["min_lidar_dist"] = min_dist
            self.stats["finite_ratio"] = finite_ratio

            obs = torch.cat([min_dist, finite_ratio], dim=-1).unsqueeze(1)

            return TensorDict(
                {
                    "agents": {"observation": obs},
                    "stats": self.stats.clone(),
                },
                self.batch_size,
            )

        def _compute_reward_and_done(self):
            reward = torch.zeros((self.num_envs, 1, 1), device=self.device)
            terminated = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
            truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

            return TensorDict(
                {
                    "agents": {"reward": reward},
                    "done": terminated | truncated,
                    "terminated": terminated,
                    "truncated": truncated,
                    "stats": self.stats.clone(),
                },
                self.batch_size,
            )

    return Mid360SphereCheck


@hydra.main(version_base=None, config_path="../../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.check_headless = bool(cfg.get("check_headless", True))
    cfg.headless = cfg.check_headless
    cfg.sim.enable_viewport = not cfg.headless
    cfg.sim.enable_replicator = False
    cfg.env.num_envs = 1
    cfg.env.max_episode_length = 400

    cfg.task.lidar_range = float(cfg.task.get("lidar_range", 10.0))
    cfg.task.num_lidar_points = int(cfg.task.get("num_lidar_points", 20000))
    cfg.task.check_sphere_center = [8.0, 0.0, 2.0]
    cfg.task.check_sphere_radius = 1.0

    simulation_app = init_simulation_app(cfg)
    Mid360SphereCheck = _build_mid360_env_class()
    env = Mid360SphereCheck(cfg, headless=cfg.headless)

    td = env.reset()
    zero_action = torch.zeros((env.num_envs, 1, env.action_dim), device=env.device)

    expected = math.sqrt((8.0 - 0.0) ** 2 + (0.0 - 0.0) ** 2 + (2.0 - 2.0) ** 2) - cfg.task.check_sphere_radius
    min_history = []

    print("=" * 72)
    print("Mid360 单障碍物验证开始: 无人机(0,0,2), 球心(8,0,2), 半径=1.0")
    print(f"理论最近命中距离(中心距-半径): {expected:.3f} m")
    print("=" * 72)

    check_steps = int(cfg.get("check_steps", 200))
    for step in range(check_steps):
        td[("agents", "action")] = zero_action
        td = env.step(td)

        min_dist = env.latest_min_dist.squeeze().item()
        finite_ratio = env.latest_finite_ratio.squeeze().item()
        min_history.append(min_dist)

        if step % 10 == 0:
            print(
                f"step={step:03d} | min_dist={min_dist:6.3f} m | finite_ratio={finite_ratio:5.3f}"
            )

        simulation_app.update()

    min_tensor = torch.tensor(min_history)
    avg_min = min_tensor.mean().item()
    std_min = min_tensor.std(unbiased=False).item()
    abs_err = abs(avg_min - expected)

    print("-" * 72)
    print(f"统计: avg_min={avg_min:.3f} m, std={std_min:.3f}, |avg-expected|={abs_err:.3f} m")

    if abs_err <= 0.6:
        print("结论: PASS - mid360 返回距离与几何期望基本一致。")
    else:
        print("结论: WARN - 偏差较大，建议检查雷达坐标系/碰撞体/mesh_prim_paths。")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
