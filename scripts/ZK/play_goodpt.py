import glob
import importlib
import os
from collections import defaultdict

# Keep Isaac/PhysX/Torch on one logical GPU unless the caller overrides it.
# If you want physical GPU 1, launch with CUDA_VISIBLE_DEVICES=1; it will still
# appear as cuda:0 inside this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.data import CompositeSpec
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from omni_drones.utils.torch import quat_axis
from omni_drones.utils.torchrl.transforms import (
    FromDiscreteAction,
    FromMultiDiscreteAction,
    ravel_composite,
)

FILE_PATH = os.path.dirname(__file__)


def _preflight_runtime_checks(cfg, eval_num_envs: int):
    typo_visible = os.environ.get("CUDA_VISIABLE_DEVICE")
    proper_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if typo_visible and not proper_visible:
        raise RuntimeError(
            "检测到环境变量 CUDA_VISIABLE_DEVICE（拼写错误）。"
            "请改为 CUDA_VISIBLE_DEVICES，例如：CUDA_VISIBLE_DEVICES=2 python play_goodpt.py"
        )

    if (not bool(cfg.headless)) and int(eval_num_envs) >= 64:
        print(
            "[Warning] 当前为可视化模式（headless=false）且并行环境较多，"
            "这可能在 PhysX GPU 初始化阶段触发崩溃。"
            "建议先用 headless=true 或降低 eval_num_envs 进行验证。"
        )


# =========================================================
# 1) 与正式训练代码保持一致的 DualStreamBackbone
# =========================================================
class DualStreamBackbone(torch.nn.Module):
    def __init__(self, state_dim, lidar_dim=3200, output_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.lidar_dim = lidar_dim

        self.lidar_encoder = torch.nn.Sequential(
            torch.nn.Linear(lidar_dim, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(0.1),
        )

        fusion_input_dim = 64 + state_dim
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_dim, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.ELU(),
        )

    def forward(self, obs):
        state = obs[..., :self.state_dim]
        lidar = obs[..., self.state_dim:]

        # 与正式训练代码保持一致
        lidar = torch.nan_to_num(lidar, posinf=20.0, neginf=0.0, nan=20.0)
        lidar = torch.clamp(lidar, min=0.0, max=20.0)
        lidar_normalized = lidar / 20.0

        lidar_features = self.lidar_encoder(lidar_normalized)
        fused_input = torch.cat([state, lidar_features], dim=-1)
        return self.fusion_mlp(fused_input)


# =========================================================
# 2) 基础工具
# =========================================================
def _safe_torch_load(path, map_location):
    """
    优先用 weights_only=True 消除 warning；
    如果当前 torch 版本或 checkpoint 格式不兼容，再退回普通 torch.load。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def _resolve_checkpoint_path(raw_path: str) -> str:
    if raw_path is None or str(raw_path).strip() == "":
        raise ValueError("checkpoint_path 不能为空，请传入 .pt 文件路径或目录路径")

    path = os.path.expanduser(str(raw_path))
    if not os.path.isabs(path):
        path = os.path.join(FILE_PATH, path)
    path = os.path.normpath(path)

    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        final_ckpt = os.path.join(path, "checkpoint_final.pt")
        if os.path.isfile(final_ckpt):
            return final_ckpt

        candidates = glob.glob(os.path.join(path, "*.pt"))
        if not candidates:
            raise FileNotFoundError(f"目录中没有找到 .pt 文件: {path}")
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError(f"checkpoint_path 不存在: {path}")


def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj

    for key in ("state_dict", "model_state_dict", "policy_state_dict", "policy"):
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]

    # 纯 state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
        return ckpt_obj

    return ckpt_obj


def _tensor_scalar(x):
    if not isinstance(x, torch.Tensor):
        return x
    y = x
    while y.ndim > 0 and y.numel() > 1:
        y = y[0]
    return y.item() if isinstance(y, torch.Tensor) else y


def _select_first_env_value(tensor):
    value = tensor[0]
    while isinstance(value, torch.Tensor) and value.ndim > 1:
        value = value[0]
    return value


# =========================================================
# 3) 相机（可选）
#    注意：你的正式环境里本身已经会在 _compute_state_and_obs 中
#    处理 follow_camera / topdown，这里只做一个兜底接口，不强改逻辑
# =========================================================
def _set_camera_view(eye, target):
    for module_name in ("isaacsim.core.utils.viewports", "omni.isaac.core.utils.viewports"):
        try:
            module = importlib.import_module(module_name)
            module.set_camera_view(eye=eye, target=target)
            return True
        except Exception:
            continue
    return False


# =========================================================
# 4) Backbone 注入：按正式训练代码的安全替换方式来
# =========================================================
def _inject_dualstream_backbone(policy, base_env, env):
    obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
    if obs_dim < 3200:
        raise ValueError(
            f"观测维度只有 {obs_dim}，小于 3200，当前 checkpoint 需要 state+lidar 的平铺观测。"
        )

    state_dim = obs_dim - 3200
    expected_feature_dim = 128

    actor_backbone = DualStreamBackbone(
        state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim
    ).to(base_env.device)

    critic_backbone = DualStreamBackbone(
        state_dim=state_dim, lidar_dim=3200, output_dim=expected_feature_dim
    ).to(base_env.device)

    actor_replaced = False
    critic_replaced = False

    # actor 结构 1: ProbabilisticActor.module -> TensorDictModule(module=nn.Sequential(...))
    if hasattr(policy.actor, "module") and hasattr(policy.actor.module, "module"):
        actor_core = policy.actor.module.module
        if isinstance(actor_core, torch.nn.Sequential) and len(actor_core) > 0:
            actor_core[0] = actor_backbone
            actor_replaced = True

    # actor 结构 2: ProbabilisticActor.module[0].module -> nn.Sequential(...)
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

    # critic 结构 2: TensorDictSequential([TensorDictModule(...), ...])
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

    print(f"[+] DualStreamBackbone 注入成功 | state_dim={state_dim}, lidar_dim=3200, output_dim={expected_feature_dim}")
    return actor_backbone, critic_backbone


# =========================================================
# 5) 智能 key 对齐
#    目标：解决你日志里 actor.module.0.module.0.xxx
#         对不上 actor.module.0.0.xxx 的问题，
#         同时尽量处理 critic 层级偏移
# =========================================================
def _canonical_key(k: str) -> str:
    # 只移除命名包装层里的 "module"，保留数字层级
    parts = [p for p in k.split(".") if p != "" and p != "module"]
    return ".".join(parts)


def _suffix_match_score(model_key: str, ckpt_key: str) -> int:
    a = _canonical_key(model_key).split(".")
    b = _canonical_key(ckpt_key).split(".")
    score = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            score += 1
        else:
            break
    return score


def _build_adapted_state_dict(model_state: dict, ckpt_state: dict):
    """
    返回:
      adapted_state: 尽量匹配到当前 model_state key 的新字典
      unmatched_model_keys: 当前模型里仍然没找到 checkpoint 来源的 key
      used_ckpt_keys: 已被消费的 checkpoint key
    """
    adapted = {}
    used_ckpt_keys = set()

    # 预构造 canonical 映射
    ckpt_by_canonical = defaultdict(list)
    for ck in ckpt_state.keys():
        ckpt_by_canonical[_canonical_key(ck)].append(ck)

    # 1) 先走精确匹配
    for mk, mv in model_state.items():
        if mk in ckpt_state and ckpt_state[mk].shape == mv.shape:
            adapted[mk] = ckpt_state[mk]
            used_ckpt_keys.add(mk)

    # 2) 再走 canonical 精确匹配（去掉 module 包装层）
    for mk, mv in model_state.items():
        if mk in adapted:
            continue
        mk_can = _canonical_key(mk)
        candidates = [
            ck for ck in ckpt_by_canonical.get(mk_can, [])
            if ck not in used_ckpt_keys and ckpt_state[ck].shape == mv.shape
        ]
        if len(candidates) == 1:
            adapted[mk] = ckpt_state[candidates[0]]
            used_ckpt_keys.add(candidates[0])

    # 3) 最后走“同前缀 + 同 shape + 最大后缀匹配”
    for mk, mv in model_state.items():
        if mk in adapted:
            continue

        mk_can = _canonical_key(mk)
        mk_prefix = mk_can.split(".")[0] if "." in mk_can else mk_can  # actor / critic / ...
        candidates = []

        for ck, cv in ckpt_state.items():
            if ck in used_ckpt_keys:
                continue
            if cv.shape != mv.shape:
                continue

            ck_can = _canonical_key(ck)
            if not (ck_can == mk_prefix or ck_can.startswith(mk_prefix + ".")):
                continue

            score = _suffix_match_score(mk, ck)
            if score >= 2:
                candidates.append((score, ck))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if len(candidates) == 1:
            adapted[mk] = ckpt_state[candidates[0][1]]
            used_ckpt_keys.add(candidates[0][1])
        elif len(candidates) > 1:
            best_score = candidates[0][0]
            best = [ck for score, ck in candidates if score == best_score]
            if len(best) == 1:
                adapted[mk] = ckpt_state[best[0]]
                used_ckpt_keys.add(best[0])

    unmatched_model_keys = [k for k in model_state.keys() if k not in adapted]
    return adapted, unmatched_model_keys, used_ckpt_keys


def _print_load_mismatch(missing_keys, unexpected_keys):
    def _split(keys):
        actor = [k for k in keys if k.startswith("actor.")]
        critic = [k for k in keys if k.startswith("critic.")]
        other = [k for k in keys if not (k.startswith("actor.") or k.startswith("critic."))]
        return actor, critic, other

    miss_actor, miss_critic, miss_other = _split(missing_keys)
    unexp_actor, unexp_critic, unexp_other = _split(unexpected_keys)

    print("[!] checkpoint 加载存在键不匹配")
    print(f"    missing(actor/critic/other): {len(miss_actor)}/{len(miss_critic)}/{len(miss_other)}")
    print(f"    unexpected(actor/critic/other): {len(unexp_actor)}/{len(unexp_critic)}/{len(unexp_other)}")
    if miss_actor:
        print(f"    missing actor sample: {miss_actor[:10]}")
    if unexp_actor:
        print(f"    unexpected actor sample: {unexp_actor[:10]}")
    if miss_critic:
        print(f"    missing critic sample: {miss_critic[:10]}")
    if unexp_critic:
        print(f"    unexpected critic sample: {unexp_critic[:10]}")


def _load_checkpoint_strictish(policy, checkpoint_path, device):
    """
    不再直接 strict=False 静默加载。
    先把 checkpoint 转成尽量匹配当前 policy 的 key，再检查是否仍有关键缺失。
    """
    checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint 解析后不是 state_dict 字典，无法加载。")

    model_state = policy.state_dict()
    adapted_state, unmatched_model_keys, used_ckpt_keys = _build_adapted_state_dict(model_state, state_dict)

    # 真正加载
    load_result = policy.load_state_dict(adapted_state, strict=False)

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))

    # strict=False 时 unexpected 通常为空，因为只喂 adapted_state
    # 这里我们自己算一下“未使用 checkpoint key”
    unused_ckpt_keys = [k for k in state_dict.keys() if k not in used_ckpt_keys]

    if missing_keys or unexpected_keys:
        _print_load_mismatch(missing_keys, unexpected_keys)

    # 关键：如果 actor/critic 还有参数没对上，直接报错，不继续假跑
    critical_missing = [
        k for k in missing_keys
        if k.startswith("actor.") or k.startswith("critic.")
    ]
    if critical_missing:
        print(f"[!] 仍有关键层未匹配成功，样例: {critical_missing[:20]}")
        print(f"[!] 未消费的 checkpoint keys 样例: {unused_ckpt_keys[:20]}")
        raise RuntimeError("checkpoint 与当前 policy 结构仍未完全对齐，停止运行。")

    print(f"[+] checkpoint 加载成功: {checkpoint_path}")
    print(f"[+] 匹配到的参数数: {len(adapted_state)} / {len(model_state)}")
    if unused_ckpt_keys:
        print(f"[Info] 仍有未使用 checkpoint keys，共 {len(unused_ckpt_keys)} 个，样例: {unused_ckpt_keys[:10]}")
    return checkpoint


# =========================================================
# 6) 终止原因诊断
#    必须与正式环境逻辑保持一致：
#    actual_dists = lidar_range - lidar_scan
# =========================================================
def _infer_done_reasons(base_env):
    try:
        env_idx = 0

        drone_state = _select_first_env_value(base_env.drone.get_state(env_frame=False))
        pos = drone_state[:3]
        quat = drone_state[3:7]

        vel_w = _select_first_env_value(base_env.drone.vel_w[..., :3])
        v_norm = float(torch.linalg.norm(vel_w).item())

        target_pos = _select_first_env_value(base_env.target_pos)
        dist_to_goal = float(torch.linalg.norm(target_pos - pos).item())

        x = float(pos[0].item())
        y = float(pos[1].item())
        z = float(pos[2].item())

        up_vec = quat_axis(quat.unsqueeze(0), axis=2).squeeze(0)
        tilt_cos = float(up_vec[2].item())
        cos_threshold = float(np.cos(float(base_env.flip_tilt_deg) * np.pi / 180.0))
        flip_now = tilt_cos < cos_threshold

        flip_counter = None
        if hasattr(base_env, "flip_counter"):
            flip_counter = int(_tensor_scalar(base_env.flip_counter[env_idx]))

        flip_early = False
        if flip_counter is not None:
            flip_early = flip_counter >= int(base_env.flip_consecutive_steps)
        else:
            flip_early = flip_now

        out_of_bounds = (abs(x) > 20.0) or (abs(y) > 30.0)
        z_low = (z < 0.4) or (z < float(base_env.terminate_z_min))
        z_high = z > float(base_env.terminate_z_max)
        v_high = v_norm > float(base_env.terminate_v_norm)
        reached_goal = dist_to_goal < float(base_env.goal_radius)

        is_collision = False
        collision_dist_now = None
        if hasattr(base_env, "lidar_scan"):
            lidar_scan = _select_first_env_value(base_env.lidar_scan)
            if isinstance(lidar_scan, torch.Tensor):
                actual_dists = float(base_env.lidar_range) - lidar_scan
                collision_dist_now = float(actual_dists.min().item())
                is_collision = bool((actual_dists < float(base_env.collision_dist)).any().item())

        is_contact_collision = False
        if bool(base_env.reset_on_collision):
            contact_force = base_env.drone.base_link.get_net_contact_forces()
            contact_force = _select_first_env_value(contact_force)
            collision_force = torch.linalg.norm(contact_force, dim=-1)
            is_contact_collision = bool((collision_force > float(base_env.collision_force_threshold)).any().item())

        reason_flags = {
            "goal_reached": reached_goal,
            "z_too_low": z_low,
            "z_too_high": z_high,
            "speed_too_high": v_high,
            "out_of_bounds": out_of_bounds,
            "collision_lidar": is_collision,
            "collision_contact": is_contact_collision,
            "flip": flip_early,
        }
        hit = [k for k, v in reason_flags.items() if v]
        if not hit:
            hit = ["unknown_or_internal_done"]

        metrics = {
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "speed": round(v_norm, 3),
            "dist_to_goal": round(dist_to_goal, 3),
            "tilt_cos": round(tilt_cos, 3),
            "flip_counter": flip_counter,
            "lidar_min_actual_dist": None if collision_dist_now is None else round(collision_dist_now, 3),
        }
        return hit, metrics
    except Exception as exc:
        return [f"done_reason_probe_failed: {type(exc).__name__}: {exc}"], {}


def _summarize_stats(stats_td):
    out = {}
    keys_to_show = [
        "return",
        "episode_len",
        "action_smoothness",
        "safety",
        "success",
        "reward_forward",
        "reward_smooth",
        "reward_max_speed",
        "reward_z",
        "reward_esdf",
        "reward_collision",
        "reward_yaw",
        "reward_goal",
        "reward_death",
        "reward_thrust",
        "reward_milestone",
        "action_sat",
    ]
    for k in keys_to_show:
        if k in stats_td.keys():
            try:
                out[k] = float(_tensor_scalar(stats_td[k]))
            except Exception:
                pass
    return out


def _summarize_stats_batch(stats_td):
    out = {}
    keys_to_show = [
        "return",
        "episode_len",
        "action_smoothness",
        "safety",
        "success",
        "reward_forward",
        "reward_smooth",
        "reward_max_speed",
        "reward_z",
        "reward_esdf",
        "reward_collision",
        "reward_yaw",
        "reward_goal",
        "reward_death",
        "reward_thrust",
        "reward_milestone",
        "action_sat",
    ]
    for k in keys_to_show:
        if k in stats_td.keys():
            try:
                v = stats_td[k]
                if isinstance(v, torch.Tensor):
                    out[k] = float(v.float().mean().item())
                else:
                    out[k] = float(v)
            except Exception:
                pass
    return out


def _extract_success_flag(stats_td):
    if stats_td is None:
        return None
    if "success" not in stats_td.keys():
        return None
    try:
        success_val = float(_tensor_scalar(stats_td["success"]))
        return 1 if success_val >= 0.5 else 0
    except Exception:
        return None


# =========================================================
# 7) 主程序
# =========================================================
@hydra.main(version_base=None, config_path="", config_name="play_goodpt")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    sim_gpu_index = int(cfg.get("sim_gpu_index", 0))
    cfg.sim.device = f"cuda:{sim_gpu_index}"
    cfg.sim.active_gpu = sim_gpu_index
    cfg.sim.physics_gpu = sim_gpu_index

    # 不再调用旧脚本的 _apply_demo_overrides()
    # 只保留“单环境评估”，不改你的终止阈值
    eval_num_envs = int(cfg.get("eval_num_envs", cfg.get("demo_num_envs", 1)))
    if "env" in cfg:
        cfg.env.num_envs = eval_num_envs
    if "task" in cfg and "env" in cfg.task:
        cfg.task.env.num_envs = eval_num_envs

    _preflight_runtime_checks(cfg, eval_num_envs)

    cfg.sim.enable_viewport = (not bool(cfg.headless)) and bool(cfg.get("enable_viewport", True))
    needs_depth_camera = bool(
        cfg.task.get("use_depth_ku_observation", False)
        or cfg.task.get("use_camera_risk_observation", False)
    )
    cfg.sim.enable_replicator = needs_depth_camera

    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    task_name = str(cfg.task.name)
    try:
        importlib.import_module(f"omni_drones.envs.single.{task_name.lower()}")
    except ModuleNotFoundError:
        pass

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    try:
        base_env = env_class(cfg, headless=cfg.headless)
    except Exception:
        simulation_app.close()
        raise

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
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1)
        )

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(int(cfg.seed))

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
    except KeyError as exc:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}") from exc

    _inject_dualstream_backbone(policy, base_env, env)

    checkpoint_path = _resolve_checkpoint_path(cfg.get("checkpoint_path"))
    _load_checkpoint_strictish(policy, checkpoint_path, base_env.device)
    policy.eval()

    base_env.enable_render(not cfg.headless)
    base_env.eval()
    env.eval()

    print(f"[Info] 当前环境数 num_envs = {base_env.num_envs}")
    print(f"[Info] terminate_z_min = {float(base_env.terminate_z_min)}")
    print(f"[Info] terminate_z_max = {float(base_env.terminate_z_max)}")
    print(f"[Info] terminate_v_norm = {float(base_env.terminate_v_norm)}")
    print(f"[Info] collision_dist = {float(base_env.collision_dist)}")
    print(f"[Info] goal_radius = {float(base_env.goal_radius)}")

    max_steps = int(cfg.get("max_steps", 1500))
    num_episodes = int(cfg.get("num_episodes", 3))
    print_every = int(cfg.get("print_every", 20))
    final_eval_rounds = max(1, int(cfg.get("final_eval_rounds", 1)))
    base_seed = int(cfg.get("seed", 0))

    print(f"[Info] final_eval_rounds = {final_eval_rounds}")
    print(f"[Info] base_seed = {base_seed}")

    all_round_success_rates = []
    all_episode_success_flags = []

    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for round_idx in range(final_eval_rounds):
                round_seed = base_seed + round_idx
                env.set_seed(round_seed)
                round_episode_success_flags = []

                print("\n" + "#" * 80)
                print(
                    f"[Round {round_idx + 1}/{final_eval_rounds}] "
                    f"开始评估 | seed={round_seed}"
                )

                for ep in range(num_episodes):
                    td = env.reset()
                    num_envs_eval = int(base_env.num_envs)
                    finished = torch.zeros(num_envs_eval, dtype=torch.bool, device=base_env.device)
                    episode_returns = torch.zeros(num_envs_eval, dtype=torch.float32, device=base_env.device)
                    episode_success_vec = torch.zeros(num_envs_eval, dtype=torch.int32, device=base_env.device)
                    last_stats = None

                    print("\n" + "=" * 80)
                    print(
                        f"[Round {round_idx + 1}/{final_eval_rounds}] "
                        f"[Episode {ep + 1}/{num_episodes}] 开始"
                    )

                    for step in range(max_steps):
                        td = policy(td)
                        td = env.step(td)

                        reward = td[("next", "agents", "reward")]
                        done = td[("next", "done")]
                        terminated = td[("next", "terminated")]
                        truncated = td[("next", "truncated")]
                        stats_td = td[("next", "stats")]
                        last_stats = stats_td

                        reward_vec = reward.reshape(-1).float()
                        done_vec = done.reshape(-1).bool()
                        terminated_vec = terminated.reshape(-1).bool()
                        truncated_vec = truncated.reshape(-1).bool()

                        active_mask = ~finished
                        if active_mask.any():
                            episode_returns[active_mask] += reward_vec[active_mask]

                        new_done_vec = done_vec & active_mask
                        finished = finished | done_vec

                        if "success" in stats_td.keys():
                            success_step = stats_td["success"].reshape(-1)
                            episode_success_vec = torch.maximum(
                                episode_success_vec,
                                (success_step >= 0.5).to(torch.int32),
                            )

                        reward_val = float(reward_vec[active_mask].mean().item()) if active_mask.any() else float(reward_vec.mean().item())
                        done_val = bool(new_done_vec.any().item())
                        terminated_val = bool((terminated_vec & active_mask).any().item())
                        truncated_val = bool((truncated_vec & active_mask).any().item())

                        if (step + 1) % print_every == 0 or step < 5 or done_val:
                            try:
                                pos = _select_first_env_value(base_env.drone.pos)
                                vel = _select_first_env_value(base_env.drone.vel_w[..., :3])
                                target_pos = _select_first_env_value(base_env.target_pos)
                                v_norm = float(torch.linalg.norm(vel).item())
                                dist_to_goal = float(torch.linalg.norm(target_pos - pos).item())
                                new_done_count = int(new_done_vec.sum().item())
                                done_total = int(finished.sum().item())
                                success_so_far = float(episode_success_vec.float().mean().item())
                                print(
                                    f"[Round {round_idx + 1}] [Episode {ep + 1}] step={step + 1}, "
                                    f"reward_mean_active={reward_val:.4f}, done={done_val}, "
                                    f"terminated={terminated_val}, truncated={truncated_val}, "
                                    f"new_done={new_done_count}, done_total={done_total}/{num_envs_eval}, "
                                    f"success_so_far={success_so_far * 100:.2f}%, "
                                    f"pos=({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f}), "
                                    f"speed={v_norm:.3f}, dist={dist_to_goal:.3f}"
                                )
                            except Exception:
                                new_done_count = int(new_done_vec.sum().item())
                                done_total = int(finished.sum().item())
                                success_so_far = float(episode_success_vec.float().mean().item())
                                print(
                                    f"[Round {round_idx + 1}] [Episode {ep + 1}] step={step + 1}, "
                                    f"reward_mean_active={reward_val:.4f}, done={done_val}, "
                                    f"terminated={terminated_val}, truncated={truncated_val}, "
                                    f"new_done={new_done_count}, done_total={done_total}/{num_envs_eval}, "
                                    f"success_so_far={success_so_far * 100:.2f}%"
                                )

                        if done_val:
                            reasons, metrics = _infer_done_reasons(base_env)
                            new_done_count = int(new_done_vec.sum().item())
                            print(
                                f"[Round {round_idx + 1}/{final_eval_rounds}] "
                                f"[Episode {ep + 1}/{num_episodes}] 在第 {step + 1} 步触发新增 done={new_done_count}"
                            )
                            print(
                                f"[Round {round_idx + 1}/{final_eval_rounds}] "
                                f"[Episode {ep + 1}/{num_episodes}] env0 done 原因: {reasons}"
                            )
                            print(
                                f"[Round {round_idx + 1}/{final_eval_rounds}] "
                                f"[Episode {ep + 1}/{num_episodes}] env0 终止瞬间状态: {metrics}"
                            )
                            print(
                                f"[Round {round_idx + 1}/{final_eval_rounds}] "
                                f"[Episode {ep + 1}/{num_episodes}] stats(mean over {num_envs_eval} envs): {_summarize_stats_batch(stats_td)}"
                            )

                        if finished.all():
                            print(
                                f"[Round {round_idx + 1}/{final_eval_rounds}] "
                                f"[Episode {ep + 1}/{num_episodes}] 所有环境已 done，提前结束于 step={step + 1}"
                            )
                            break

                        td = step_mdp(td)

                    if not finished.all():
                        print(
                            f"[Round {round_idx + 1}/{final_eval_rounds}] "
                            f"[Episode {ep + 1}/{num_episodes}] 跑满 {max_steps} 步，仍有 {int((~finished).sum().item())} 个环境未 done"
                        )
                        if last_stats is not None:
                            print(
                                f"[Round {round_idx + 1}/{final_eval_rounds}] "
                                f"[Episode {ep + 1}/{num_episodes}] stats(mean over {num_envs_eval} envs): {_summarize_stats_batch(last_stats)}"
                            )

                    if last_stats is not None and "success" in last_stats.keys():
                        success_final = (last_stats["success"].reshape(-1) >= 0.5).to(torch.int32)
                        episode_success_vec = torch.maximum(episode_success_vec, success_final)

                    episode_success_count = int(episode_success_vec.sum().item())
                    episode_success_rate = episode_success_count / max(1, num_envs_eval)
                    episode_return_mean = float(episode_returns.mean().item())
                    episode_return_std = float(episode_returns.std().item())

                    print(
                        f"[Round {round_idx + 1}/{final_eval_rounds}] "
                        f"[Episode {ep + 1}/{num_episodes}] episode_return(mean±std) = {episode_return_mean:.4f} ± {episode_return_std:.4f}"
                    )
                    print(
                        f"[Round {round_idx + 1}/{final_eval_rounds}] "
                        f"[Episode {ep + 1}/{num_episodes}] success_rate(batch) = {episode_success_rate * 100:.2f}% ({episode_success_count}/{num_envs_eval})"
                    )

                    episode_success_flags = episode_success_vec.detach().cpu().tolist()
                    round_episode_success_flags.extend(int(v) for v in episode_success_flags)
                    all_episode_success_flags.extend(int(v) for v in episode_success_flags)

                    round_running_count = int(sum(round_episode_success_flags))
                    round_running_rate = round_running_count / len(round_episode_success_flags)
                    print(
                        f"[Round {round_idx + 1}/{final_eval_rounds}] "
                        f"[Episode {ep + 1}/{num_episodes}] 本轮当前平均成功率: "
                        f"{round_running_rate * 100:.2f}% "
                        f"({round_running_count}/{len(round_episode_success_flags)})"
                    )

                round_success_count = int(sum(round_episode_success_flags))
                round_success_rate = round_success_count / max(1, len(round_episode_success_flags))
                all_round_success_rates.append(round_success_rate)

                print("\n" + "-" * 80)
                print(
                    f"[Round {round_idx + 1}/{final_eval_rounds}] 成功率: "
                    f"{round_success_rate * 100:.2f}% "
                    f"({round_success_count}/{len(round_episode_success_flags)})"
                )

            overall_success_count = int(sum(all_episode_success_flags))
            overall_success_rate = overall_success_count / max(1, len(all_episode_success_flags))
            avg_round_success_rate = float(np.mean(all_round_success_rates))

            print("\n" + "=" * 80)
            print(
                f"[Final Eval] {final_eval_rounds}轮按轮平均成功率: "
                f"{avg_round_success_rate * 100:.2f}%"
            )
            print(
                f"[Summary] 全部episode汇总成功率: {overall_success_rate * 100:.2f}% "
                f"({overall_success_count}/{len(all_episode_success_flags)})"
            )
    except KeyboardInterrupt:
        print("\n[Warning] 检测到中断信号（Ctrl+C），输出当前已收集的部分统计。")
        if all_episode_success_flags:
            partial_success_count = int(sum(all_episode_success_flags))
            partial_success_rate = partial_success_count / max(1, len(all_episode_success_flags))
            partial_round_rate = float(np.mean(all_round_success_rates)) if all_round_success_rates else partial_success_rate
            print("\n" + "=" * 80)
            print(
                f"[Partial Eval] 已完成轮次平均成功率: "
                f"{partial_round_rate * 100:.2f}%"
            )
            print(
                f"[Partial Summary] 已完成episode汇总成功率: "
                f"{partial_success_rate * 100:.2f}% "
                f"({partial_success_count}/{len(all_episode_success_flags)})"
            )
        else:
            print("[Partial Summary] 未收集到可用的episode统计。")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
