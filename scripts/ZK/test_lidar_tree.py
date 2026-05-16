import logging
import os
# 🌟 必须放在 import torch 和其他库的最前面！
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
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

from train_camlidar_trees import (
    DEFAULT_TREE_OBJ,
    _load_policy_checkpoint_compatible,
    patched_realtree_forest,
)


# class DualStreamBackbone(torch.nn.Module):
#     # 将默认的 output_dim 改为 128，严格对齐论文 Fusion 模块的最后一层
#     def __init__(self, state_dim, lidar_dim=3200, output_dim=128):
#         super().__init__()
#         self.state_dim = state_dim
#         self.lidar_dim = lidar_dim

#         # ================= 1. 雷达特征降维编码器 (MLP Encoder) =================
#         # 📝 论文设定：[128, 64, 64]
#         # 🎯 物理意义：将 3200 维的庞大雷达点云，极致压缩提炼成 64 维的核心避障特征
#         self.lidar_encoder = torch.nn.Sequential(
#             torch.nn.Linear(lidar_dim, 128),
#             torch.nn.ELU(),
#             torch.nn.Linear(128, 64),
#             torch.nn.ELU(),
#             torch.nn.Linear(64, 64),  # 🌟 修复点：降维到 64，绝不反弹！
#             torch.nn.ELU()
#         )

#         # ================= 2. 状态融合模块 (MLP Fusion) =================
#         # 📝 论文设定：[128, 256, 256, 128]
#         # 🎯 拼接维度：64 (雷达压缩特征) + 14 (精简的自身状态) = 78 维
#         fusion_input_dim = 64 + state_dim 
        
#         self.fusion_mlp = torch.nn.Sequential(
#             torch.nn.Linear(fusion_input_dim, 128),
#             torch.nn.ELU(),
#             torch.nn.Linear(128, 256),
#             torch.nn.ELU(),
#             torch.nn.Linear(256, 256),
#             torch.nn.ELU(),
#             torch.nn.Linear(256, output_dim), # 🌟 最终输出论文要求的 128 维特征
#             torch.nn.ELU()
#         )

#     def forward(self, obs):
#         # 按照拼接顺序切分输入
#         state = obs[..., :self.state_dim]
#         lidar = obs[..., self.state_dim:]

#         # 提炼雷达特征并与自身状态融合
#         lidar_features = self.lidar_encoder(lidar)
#         fused_input = torch.cat([state, lidar_features], dim=-1)

#         return self.fusion_mlp(fused_input)

 #雷达没有被归一化修改一版
class DualStreamBackbone(torch.nn.Module):
    def __init__(self, state_dim, lidar_dim=3200, output_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.lidar_dim = lidar_dim

        # ================= 1. 雷达特征降维编码器 =================
        # 🌟 修复 1：把 ELU 换成 LeakyReLU。
        # LeakyReLU 在负数区有一条微小的斜率(默认0.01)，绝不会产生 0 梯度，是防脑死的终极神器！
        self.lidar_encoder = torch.nn.Sequential(
            torch.nn.Linear(lidar_dim, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(0.1)
        )

        fusion_input_dim = 64 + state_dim 
        
        # ================= 2. 状态融合模块 =================
        # 这里维持 ELU 没问题，因为融合层的输入已经是被压缩过的安全数据了
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_dim, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.ELU()
        )

    def forward(self, obs):
        # 按照拼接顺序切分输入
        state = obs[..., :self.state_dim]
        lidar = obs[..., self.state_dim:]

        # ================= 🌟 修复 2：雷达点云“解毒”与归一化 =================
        # 1. 干掉物理引擎传回来的 inf 和 nan（假设没有击中障碍物时返回最大距离）
        lidar = torch.nan_to_num(lidar, posinf=20.0, neginf=0.0, nan=20.0) 
        
        # 2. 强行截断在合理物理范围内（假设你的雷达最远探测 20 米）
        # 如果你的 OmniDrones 配置雷达最远看 50 米，就把这里的 20.0 改成 50.0
        lidar = torch.clamp(lidar, min=0.0, max=20.0)
        
        # 3. 归一化到 [0, 1]！神经网络最喜欢的数据尺度！
        lidar_normalized = lidar / 20.0 
        # =================================================================

        # 提炼雷达特征并与自身状态融合
        lidar_features = self.lidar_encoder(lidar_normalized)
        fused_input = torch.cat([state, lidar_features], dim=-1)

        return self.fusion_mlp(fused_input)
@hydra.main(version_base=None, config_path="", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    if str(cfg.task.name) != "forest_singal":
        logging.warning(
            "test_lidar_tree.py is intended for forest_singal/forest_signal, got task=%s.",
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

    # 按规模自动控制渲染，避免大批量并行时触发 syntheticdata 崩溃（exit 139）
    num_envs = int(cfg.env.num_envs)
    user_viewport = bool(cfg.sim.get("enable_viewport", False))
    user_replicator = bool(cfg.sim.get("enable_replicator", True))

    # 仅在需要评估可视化且并行规模可控时开启
    allow_render = (not cfg.headless) and (num_envs <= 200)
    allow_replicator = (cfg.get("eval_interval", -1) > 0) and (num_envs <= 200)

    cfg.sim.enable_viewport = user_viewport or allow_render
    cfg.sim.enable_replicator = user_replicator and allow_replicator

    # 安全阈值：高并行下强制关闭复制器与视口
    if num_envs > 200:
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

    # Keep optimizer hyper-parameters from PPO implementation before replacing modules.
    actor_lr = policy.actor_opt.param_groups[0]["lr"]
    critic_lr = policy.critic_opt.param_groups[0]["lr"]

    # # ================= 🚀 注入论文同款双流网络架构 =================
    # obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
    # state_dim = obs_dim - 3200

    # # 🌟 动态获取 OmniDrones 动作头期待的输入维度
    # try:
    #     expected_feature_dim = policy.critic.module[1].in_features
    # except AttributeError:
    #     expected_feature_dim = 128  # 默认回退值，假设论文 Fusion 模块输出 128 维特征

    # # actor_backbone = DualStreamBackbone(state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim).to(base_env.device)
    # # critic_backbone = DualStreamBackbone(state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim).to(base_env.device)

    # # policy.actor.module[0] = actor_backbone
    # # policy.critic.module[0] = critic_backbone
    
    # actor_backbone = DualStreamBackbone(
    # state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim
    # ).to(base_env.device)

    # critic_backbone = DualStreamBackbone(
    #     state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim
    # ).to(base_env.device)

    # # actor: ProbabilisticActor -> TensorDictModule -> nn.Sequential(...)
    # policy.actor.module[0].module[0] = actor_backbone

    # # critic: TensorDictModule -> nn.Sequential(...)
    # policy.critic.module[0] = critic_backbone
    # print("actor wrapper type:", type(policy.actor.module))
    # print("actor core:", policy.actor.module.module)
    # print("critic core:", policy.critic.module)
    # print("actor backbone requires_grad:", actor_backbone.lidar_encoder[0].weight.requires_grad)
    # print("critic backbone requires_grad:", critic_backbone.lidar_encoder[0].weight.requires_grad)
    
    # print(f"✅ 成功注入双流网络！状态维度: {state_dim}, 雷达维度: 3200, 融合输出维度已适配为: {expected_feature_dim}")
    # # ===============================================================
    obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
    state_dim = obs_dim - 3200
    expected_feature_dim = 128

    actor_backbone = DualStreamBackbone(
        state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim
    ).to(base_env.device)

    critic_backbone = DualStreamBackbone(
        state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim
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
            # 使用极小的标准差，让初始特征极其平缓
            # torch.nn.init.orthogonal_(m.weight, 0.01)
            # torch.nn.init.constant_(m.bias, 0.0)
            torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            torch.nn.init.constant_(m.bias, 0.0)

    actor_backbone.apply(init_weights)
    critic_backbone.apply(init_weights)
    # ==========================================================


    if hasattr(policy.actor.module, "module"):
        print("actor after replace:", policy.actor.module.module)
    elif hasattr(policy.actor.module, "__getitem__") and hasattr(policy.actor.module[0], "module"):
        print("actor after replace:", policy.actor.module[0].module)
    print("critic after replace:", policy.critic.module)
    print(f"✅ 成功注入双流网络！state_dim={state_dim}, lidar_dim=3200, output_dim={expected_feature_dim}")
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

    actor_param_bound = _optimizer_has_param(policy.actor_opt, actor_backbone.lidar_encoder[0].weight)
    critic_param_bound = _optimizer_has_param(policy.critic_opt, critic_backbone.lidar_encoder[0].weight)
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
                import omni.usd
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
    #         w_before = actor_backbone.lidar_encoder[0].weight.detach().clone()

    #     train_info = policy.train_op(data.to_tensordict())

    #     with torch.no_grad():
    #         w_after = actor_backbone.lidar_encoder[0].weight.detach()
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
            actor_lidar_before = actor_backbone.lidar_encoder[0].weight.detach().clone()
            actor_fusion_before = actor_backbone.fusion_mlp[0].weight.detach().clone()
            critic_lidar_before = critic_backbone.lidar_encoder[0].weight.detach().clone()
            critic_fusion_before = critic_backbone.fusion_mlp[0].weight.detach().clone()

        train_info = policy.train_op(data.to_tensordict())

        with torch.no_grad():
            actor_lidar_after = actor_backbone.lidar_encoder[0].weight.detach()
            actor_fusion_after = actor_backbone.fusion_mlp[0].weight.detach()
            critic_lidar_after = critic_backbone.lidar_encoder[0].weight.detach()
            critic_fusion_after = critic_backbone.fusion_mlp[0].weight.detach()

            actor_delta_lidar = (actor_lidar_after - actor_lidar_before).abs().mean().item()
            actor_delta_fusion = (actor_fusion_after - actor_fusion_before).abs().mean().item()
            critic_delta_lidar = (critic_lidar_after - critic_lidar_before).abs().mean().item()
            critic_delta_fusion = (critic_fusion_after - critic_fusion_before).abs().mean().item()

            actor_delta = max(actor_delta_lidar, actor_delta_fusion)
            critic_delta = max(critic_delta_lidar, critic_delta_fusion)

        info.update(train_info)
        info["debug/actor_backbone_delta"] = actor_delta
        info["debug/critic_backbone_delta"] = critic_delta
        info["debug/actor_delta_lidar"] = actor_delta_lidar
        info["debug/actor_delta_fusion"] = actor_delta_fusion
        info["debug/critic_delta_lidar"] = critic_delta_lidar
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
                f"actor_delta={actor_delta:.8e} (lidar={actor_delta_lidar:.8e}, fusion={actor_delta_fusion:.8e}), "
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


def _ensure_default_task_override() -> None:
    has_task_override = any(
        arg == "task"
        or arg.startswith("task=")
        or arg.startswith("+task=")
        or arg.startswith("++task=")
        for arg in sys.argv[1:]
    )
    if not has_task_override:
        sys.argv.append("task=forest_signal")


if __name__ == "__main__":
    _ensure_default_task_override()
    main()
