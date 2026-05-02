import logging
import os
import gc
import copy

# 🌟 显存优化配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import importlib
import math
import sys
import hydra
import torch

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from setproctitle import setproctitle

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType, step_mdp
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from omni_drones.utils.torchrl import SyncDataCollector, RenderCallback, EpisodeStats
from omni_drones.utils.torchrl.transforms import ravel_composite
from omni_drones.learning import ALGOS
from omni_drones.utils.wandb import init_wandb

# ================= 🌟 引入 LaRes 核心组件 =================
from replay_buffer import replay_buffer
from llm_manager import LLMManager
from scipy.stats import beta

# 关闭过多 HTTP / SDK 调试日志
for _name in ("openai", "openai._base_client", "httpx", "httpcore"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.WARNING)
    _logger.propagate = False


def cleanup_cuda():
    """主动释放 Python 垃圾和 CUDA cache，缓解评估阶段 OOM。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def tensor_to_env_scalar(x: torch.Tensor) -> torch.Tensor:
    """
    把形状统一拍成 [num_envs]。
    常见输入：
    [N], [N,1], [N,1,1], [N,3] -> [N]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor, got {type(x)}")

    if x.dim() == 0:
        return x.unsqueeze(0)

    if x.dim() == 1:
        return x

    return x.reshape(x.shape[0], -1).mean(dim=-1)


def tensor_to_env_bool(x: torch.Tensor) -> torch.Tensor:
    """
    把 done / terminated / truncated 统一成 [num_envs] 的 bool。
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor, got {type(x)}")

    if x.dtype != torch.bool:
        x = x > 0

    if x.dim() == 1:
        return x

    return x.reshape(x.shape[0], -1).any(dim=-1)


class ThompsonSampling:
    """LaRes 用于奖励函数选择的经典算法"""
    def __init__(self, n_arms):
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def add_arm(self):
        self.successes = np.append(self.successes, 1.0)
        self.failures = np.append(self.failures, 1.0)
        return len(self.successes) - 1

    def select_arm(self):
        if len(self.successes) == 0:
            raise RuntimeError("ThompsonSampling has no arms to select from")
        sampled_theta = [beta.rvs(s, f) for s, f in zip(self.successes, self.failures)]
        return np.argmax(sampled_theta)

    def remove_arm(self, arm):
        self.successes = np.delete(self.successes, arm)
        self.failures = np.delete(self.failures, arm)

    def posterior_mean(self, arm):
        total = float(self.successes[arm] + self.failures[arm])
        return float(self.successes[arm] / total) if total > 0 else 0.5

    def posterior_std(self, arm):
        alpha = float(self.successes[arm])
        beta_param = float(self.failures[arm])
        total = alpha + beta_param
        if total <= 0.0:
            return 0.0
        variance = (alpha * beta_param) / ((total * total) * (total + 1.0))
        return float(math.sqrt(max(variance, 0.0)))

    def acquisition_score(self, arm, exploration_coef=0.5):
        return self.posterior_mean(arm) + float(exploration_coef) * self.posterior_std(arm)

    def suggest_train_iters(self, arm, base_iters, min_scale=0.5, max_scale=1.8, exploration_coef=0.5):
        quality = self.acquisition_score(arm, exploration_coef=exploration_coef)
        max_quality = max(
            self.acquisition_score(i, exploration_coef=exploration_coef)
            for i in range(len(self.successes))
        ) if len(self.successes) > 0 else quality
        relative_quality = quality / max_quality if max_quality > 0 else 1.0
        relative_quality = float(np.clip(relative_quality, 0.0, 1.0))
        scale = min_scale + (max_scale - min_scale) * relative_quality
        return max(1, int(round(base_iters * scale)))

    def update(self, arm, success_rate):
        if success_rate > 0.5:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


class DualStreamBackbone(torch.nn.Module):
    def __init__(self, state_dim, lidar_dim=3200, output_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.lidar_encoder = torch.nn.Sequential(
            torch.nn.Linear(lidar_dim, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 64), torch.nn.ELU()
        )
        fusion_input_dim = 64 + state_dim
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_dim, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 256), torch.nn.ELU(),
            torch.nn.Linear(256, output_dim), torch.nn.ELU()
        )

    def forward(self, obs):
        state = obs[..., :self.state_dim]
        lidar = obs[..., self.state_dim:]
        lidar_features = self.lidar_encoder(lidar)
        return self.fusion_mlp(torch.cat([state, lidar_features], dim=-1))


def _normalize_llm_reward_inputs(kwargs):
    """Normalize env state tensors to shapes that are robust for generated reward code."""
    vector_keys = {"curr_pos", "target_pos", "v", "omega", "x_body"}
    normalized = {}

    def _to_vector3(t: torch.Tensor) -> torch.Tensor:
        """Force tensor into shape [num_envs, 3] for vector-valued state keys."""
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() >= 2 and t.shape[-2] == 1:
            t = t.squeeze(-2)

        if t.dim() > 2:
            t = t.reshape(t.shape[0], -1)
        elif t.dim() == 2:
            pass
        else:
            t = t.view(t.shape[0], 1)

        if t.shape[-1] >= 3:
            return t[..., :3]

        pad = torch.zeros(t.shape[0], 3 - t.shape[-1], device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=-1)

    for k, v in kwargs.items():
        if not torch.is_tensor(v):
            normalized[k] = v
            continue

        t = v
        if k in vector_keys:
            t = _to_vector3(t)
        else:
            while t.dim() > 1 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            if t.dim() > 2 and t.shape[-2] == 1:
                t = t.squeeze(-2)
            if t.dim() > 1:
                t = t.reshape(t.shape[0], -1).mean(dim=-1)

        normalized[k] = t
    return normalized


def _safe_fallback_reward(kwargs):
    """A robust fallback reward that always returns a 1D tensor [num_envs]."""
    curr_dist = kwargs.get("curr_dist")
    prev_dist = kwargs.get("prev_dist")
    min_lidar_dist = kwargs.get("min_lidar_dist")
    action_diff = kwargs.get("action_diff")
    z_pos = kwargs.get("z_pos")

    if not torch.is_tensor(curr_dist):
        return torch.zeros(1, dtype=torch.float32)

    progress = prev_dist - curr_dist if torch.is_tensor(prev_dist) else torch.zeros_like(curr_dist)
    safety = min_lidar_dist if torch.is_tensor(min_lidar_dist) else torch.ones_like(curr_dist)
    smooth = action_diff if torch.is_tensor(action_diff) else torch.zeros_like(curr_dist)
    altitude = z_pos if torch.is_tensor(z_pos) else torch.full_like(curr_dist, 2.0)

    reward = (
        2.0 * torch.tanh(progress)
        + 0.5 * torch.exp(-curr_dist.clamp_min(0.0))
        - 2.0 * torch.exp(-10.0 * safety.clamp_min(0.0))
        - 0.1 * smooth.abs()
        - 0.05 * (altitude - 2.0).abs()
    )
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-1e4)


def _wrap_llm_reward_fn(module):
    """Wrap compute_reward so output is always a finite 1D tensor [num_envs]."""
    if getattr(module, "_lares_wrapped", False):
        return

    original_fn = module.compute_reward

    def _safe_compute_reward(**kwargs):
        safe_kwargs = _normalize_llm_reward_inputs(kwargs)
        try:
            reward = original_fn(**safe_kwargs)

            if not torch.is_tensor(reward):
                ref = safe_kwargs.get("curr_dist")
                device = ref.device if torch.is_tensor(ref) else None
                reward = torch.as_tensor(reward, device=device)

            while reward.dim() > 1 and reward.shape[-1] == 1:
                reward = reward.squeeze(-1)

            if reward.dim() != 1:
                raise RuntimeError(f"LLM reward shape invalid: {tuple(reward.shape)}; expected [num_envs]")
        except Exception as e:
            if not getattr(module, "_lares_reward_runtime_warned", False):
                print(f"[!] LLM Reward runtime error in wrapped function, using safe fallback: {e}")
                module._lares_reward_runtime_warned = True
            reward = _safe_fallback_reward(safe_kwargs)

        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-1e4)

    module.compute_reward = _safe_compute_reward
    module._lares_wrapped = True


def _build_reward_param_context(cfg) -> str:
    keys = [
        "goal_radius", "goal_bonus", "collision_dist", "collision_penalty",
        "v_max", "lambda_esdf", "k_esdf",
        "w_forward", "w_smooth", "w_max_speed", "w_z", "w_esdf", "w_yaw", "w_thrust",
        "terminate_z_min", "terminate_z_max", "terminate_v_norm",
    ]
    lines = []
    task_cfg = cfg.get("task", None) if hasattr(cfg, "get") else None
    for k in keys:
        v = None
        if task_cfg is not None and hasattr(task_cfg, "get"):
            v = task_cfg.get(k, None)
        if v is None and hasattr(cfg, "get"):
            v = cfg.get(k, None)
        if v is not None:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _baseline_reward_reference() -> str:
    # Concise baseline skeleton from your historical reward design.
    return """def compute_reward(**kwargs):
    curr_pos = kwargs[\"curr_pos\"]
    target_pos = kwargs[\"target_pos\"]
    curr_dist = kwargs[\"curr_dist\"]
    prev_dist = kwargs[\"prev_dist\"]
    v = kwargs[\"v\"]
    v_norm = kwargs[\"v_norm\"]
    omega = kwargs[\"omega\"]
    min_lidar_dist = kwargs[\"min_lidar_dist\"]
    action_diff = kwargs[\"action_diff\"]
    z_pos = kwargs[\"z_pos\"]
    tilt_cos = kwargs[\"tilt_cos\"]
    x_body = kwargs[\"x_body\"]

    eps = 1e-6

    # Goal-directed shaping.
    progress_reward = torch.tanh(3.0 * (prev_dist - curr_dist))
    proximity_reward = 1.8 * torch.exp(-0.30 * curr_dist)
    goal_bonus = (curr_dist < 0.75).float() * 12.0

    target_vec = target_pos - curr_pos
    target_dir = target_vec / torch.norm(target_vec, dim=-1).clamp_min(eps).unsqueeze(-1)
    forward_align = (x_body * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
    forward_reward = 0.40 * torch.clamp(forward_align, 0.0, 1.0)

    # Always-on stability regularization.
    omega_norm = torch.norm(omega, dim=-1)
    smooth_penalty = 0.08 * action_diff + 0.05 * omega_norm
    global_tilt_penalty = 0.5 * torch.relu(0.9 - tilt_cos)

    # Quadratic altitude regularization around target altitude 2.0m.
    altitude_reg = -0.3 * torch.square(z_pos - 2.0)

    # Explicit death-like penalty (anti-termination-hacking).
    is_flipped = tilt_cos < 0.18
    is_out_of_bounds = (z_pos > 3.5) | (z_pos < 0.3)
    death_penalty = (is_flipped | is_out_of_bounds).float() * 50.0

    # Obstacle and near-obstacle dynamic penalties.
    safe_dist = min_lidar_dist.clamp_min(eps)
    near_mask = (min_lidar_dist < 2.0).float()
    critical_mask = (min_lidar_dist < 0.35).float()

    obstacle_penalty = (
        0.5 * torch.exp(-1.5 * safe_dist)
        + 1.5 * torch.exp(-4.0 * safe_dist)
        + critical_mask * 10.0
    )

    near_speed_penalty = near_mask * (0.2 * v_norm + 0.1 * torch.square(v_norm))
    near_vz_penalty = near_mask * (0.2 * torch.abs(v[:, 2]))

    reward = (
        3.8 * progress_reward
        + 1.6 * proximity_reward
        + goal_bonus
        + forward_reward
        + altitude_reg
        - smooth_penalty
        - global_tilt_penalty
        - obstacle_penalty
        - near_speed_penalty
        - near_vz_penalty
        - death_penalty
    )
    reward = torch.nan_to_num(reward, nan=-1e4, posinf=1e4, neginf=-1e4)
    reward = torch.clamp(reward, min=-1e4, max=1e4)
    return reward"""


@hydra.main(version_base=None, config_path="", config_name="LaRes")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    num_envs = int(cfg.env.num_envs)
    cfg.sim.enable_viewport = (not cfg.headless) and (num_envs <= 200)

    simulation_app = init_simulation_app(cfg)
    import omni_drones.envs.single.forest_weues
    run = init_wandb(cfg)
    setproctitle(run.name)

    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # Runtime hotfix:
    # forest_weues._compute_reward_and_done 里如果引用 module-level 的 action_diff，
    # 这里在每次 reward 前注入，避免 NameError。
    fw_module = importlib.import_module("omni_drones.envs.single.forest_weues")
    if not getattr(base_env.__class__, "_lares_action_diff_hotfix", False):
        _orig_compute_reward_and_done = base_env.__class__._compute_reward_and_done

        def _compute_reward_and_done_with_action_diff(self, *args, **kwargs):
            try:
                fw_module.action_diff = self.get_clean_states()["action_diff"]
            except Exception:
                fw_module.action_diff = torch.zeros(self.num_envs, device=self.device)
            return _orig_compute_reward_and_done(self, *args, **kwargs)

        base_env.__class__._compute_reward_and_done = _compute_reward_and_done_with_action_diff
        base_env.__class__._lares_action_diff_hotfix = True

    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # 初始化算法
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=base_env.device
    )

    # 注入双流 Backbone
    obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
    state_dim = obs_dim - 3200
    actor_backbone = DualStreamBackbone(state_dim=state_dim).to(base_env.device)
    critic_backbone = DualStreamBackbone(state_dim=state_dim).to(base_env.device)
    policy.actor.module[0].module[0] = actor_backbone
    policy.critic.module[0] = critic_backbone

    # 重新绑定优化器以追踪新 Backbone
    actor_lr = cfg.algo.get("actor_update_lr", cfg.algo.get("actor_lr", cfg.algo.get("lr", 3e-4)))
    critic_lr = cfg.algo.get("critic_update_lr", cfg.algo.get("critic_lr", cfg.algo.get("lr", 3e-4)))
    policy.actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=float(actor_lr))
    policy.critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=float(critic_lr))

    train_every = int(cfg.algo.train_every)
    frames_per_batch = env.num_envs * train_every
    arm_min_scale = float(cfg.get("arm_train_iter_min_scale", 0.5))
    arm_max_scale = float(cfg.get("arm_train_iter_max_scale", 1.8))
    arm_exploration_coef = float(cfg.get("arm_train_iter_exploration_coef", 0.5))

    max_llm_iters = int(cfg.get("max_llm_iters", 10))
    ppo_iters_per_llm = int(cfg.get("ppo_iters_per_llm", 3000))
    max_ppo_iters_per_gen = max(1, int(math.ceil(ppo_iters_per_llm * arm_max_scale)))
    min_frames_per_gen = int(frames_per_batch * max_ppo_iters_per_gen)
    min_total_frames = int(min_frames_per_gen * max_llm_iters)
    configured_total_frames = int(cfg.get("total_frames", 1000000))
    collector_total_frames = int(max(configured_total_frames, min_total_frames))

    if collector_total_frames != configured_total_frames:
        print(
            f"⚠️ total_frames={configured_total_frames} 小于 {max_llm_iters} 代所需 {min_total_frames}，已自动提升到 {collector_total_frames}。",
            flush=True,
        )

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=collector_total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    episode_stats = EpisodeStats(
        [k for k in base_env.observation_spec.keys(True, True) if k[0] == "stats"]
    )

    # ================= 🚀 LLM 进化大循环 =================
    llm = LLMManager()

    reflection_buffer = replay_buffer(memory_size=100)
    ts_selector = ThompsonSampling(n_arms=0)
    reward_arms = []
    max_reward_arms = max(1, int(cfg.get("max_reward_arms", 8)))
    arm_keep_recent = max(0, int(cfg.get("arm_keep_recent", 2)))

    def _register_reward_arm(code: str, label: str) -> int:
        arm_idx = ts_selector.add_arm()
        reward_arms.append({"code": code, "label": label, "created_iter": len(reward_arms)})
        print(f"🎯 注册奖励 arm[{arm_idx}] = {label}", flush=True)
        return arm_idx

    def _prune_reward_arms(selected_arm_idx: int | None = None):
        if len(reward_arms) <= max_reward_arms:
            return

        protected_indices = set(range(max(0, len(reward_arms) - arm_keep_recent), len(reward_arms)))
        if selected_arm_idx is not None:
            protected_indices.add(int(selected_arm_idx))

        if len(reward_arms) > 0:
            acquisition_scores = [
                ts_selector.acquisition_score(i, exploration_coef=arm_exploration_coef)
                for i in range(len(reward_arms))
            ]
            best_arm_idx = int(np.argmax(acquisition_scores))
            protected_indices.add(best_arm_idx)

        removable_indices = [i for i in range(len(reward_arms)) if i not in protected_indices]
        if not removable_indices:
            return

        removable_indices.sort(
            key=lambda i: (
                ts_selector.acquisition_score(i, exploration_coef=arm_exploration_coef),
                reward_arms[i]["created_iter"],
            )
        )
        excess = len(reward_arms) - max_reward_arms
        prune_indices = sorted(removable_indices[:excess], reverse=True)
        for prune_idx in prune_indices:
            removed_arm = reward_arms.pop(prune_idx)
            ts_selector.remove_arm(prune_idx)
            print(
                f"🧹 淘汰 arm[{prune_idx}] ({removed_arm['label']})，避免垃圾奖励公式长期占池。",
                flush=True,
            )

    state_desc = '"curr_pos", "target_pos", "curr_dist", "prev_dist", "v", "v_norm", "omega", "min_lidar_dist", "action_diff", "tilt_cos", "z_pos", "x_body"'

    current_code = ""
    best_sr = 0.0
    current_success_rate = 0.0
    global_ppo_step = 0
    reward_param_context = _build_reward_param_context(cfg)
    baseline_reward_ref = _baseline_reward_reference()

    def evaluate(
        seed: int = 0,
        exploration_type: ExplorationType = ExplorationType.MODE,
        eval_iter: int = 0,
    ):
        cleanup_cuda()

        should_record_video = bool(cfg.get("eval_record_video", True))
        # In this task/environment, RGB recording requires Replicator explicitly.
        can_record_video = bool(cfg.sim.get("enable_replicator", False))
        record_video_this_eval = should_record_video and can_record_video

        render_callback = RenderCallback(interval=int(cfg.get("eval_video_interval", 2))) if record_video_this_eval else None
        base_env.enable_render(record_video_this_eval)

        if should_record_video and (not record_video_this_eval):
            print(
                f"⚠️ 第 {eval_iter} 代评估跳过视频录制（当前需要 cfg.sim.enable_replicator=True）。",
                flush=True,
            )

        base_env.eval()
        env.eval()
        try:
            policy.eval()
        except Exception:
            pass

        env.set_seed(seed)

        max_steps = int(getattr(base_env, "max_episode_length", 2000))
        num_envs_eval = base_env.num_envs
        device = base_env.device

        finished_mask = torch.zeros(num_envs_eval, dtype=torch.bool, device=device)
        finished_stats = {}
        last_next_stats = None

        with torch.inference_mode(), set_exploration_type(exploration_type):
            td = env.reset()

            for step in range(max_steps):
                td = policy(td)
                td = env.step(td)

                if render_callback is not None:
                    try:
                        render_callback(base_env)
                    except Exception as e:
                        print(f"[!] 第 {eval_iter} 代评估渲染失败(step={step})，本轮跳过视频: {e}")
                        render_callback = None

                next_done = td.get(("next", "done"))
                done_mask = tensor_to_env_bool(next_done)

                next_stats = td.get(("next", "stats"))
                last_next_stats = next_stats

                just_finished = done_mask & (~finished_mask)
                if just_finished.any():
                    for k, v in next_stats.items():
                        v_scalar = tensor_to_env_scalar(v).float()
                        if k not in finished_stats:
                            finished_stats[k] = torch.zeros(num_envs_eval, device=device, dtype=torch.float32)
                        finished_stats[k][just_finished] = v_scalar[just_finished]
                    finished_mask |= just_finished

                if finished_mask.all():
                    break

                td = step_mdp(td)

            if last_next_stats is not None and (~finished_mask).any():
                unfinished = ~finished_mask
                for k, v in last_next_stats.items():
                    v_scalar = tensor_to_env_scalar(v).float()
                    if k not in finished_stats:
                        finished_stats[k] = torch.zeros(num_envs_eval, device=device, dtype=torch.float32)
                    finished_stats[k][unfinished] = v_scalar[unfinished]

            # 关键：把 reset 放到 inference_mode 里面
            env.reset()

        info = {}
        for k, v in finished_stats.items():
            info[f"eval/stats.{k}"] = float(v.mean().detach().cpu().item())

        success_flags = finished_stats.get("success")
        if success_flags is not None:
            info["eval/success_rate"] = float((success_flags > 0).float().mean().detach().cpu().item())
        else:
            info["eval/success_rate"] = 0.0

        if render_callback is not None and len(render_callback.frames) > 0:
            info["eval/recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=max(10, int(0.5 / (cfg.sim.dt * cfg.sim.substeps))),
                format="mp4",
            )

        base_env.train()
        env.train()
        try:
            policy.train()
        except Exception:
            pass

        base_env.enable_render((not cfg.headless) and cfg.sim.enable_viewport)

        del td
        del last_next_stats
        cleanup_cuda()
        return info
    for llm_iter in range(max_llm_iters):
        print(f"\n🌀 [Iteration {llm_iter + 1}] LLM 进化中...")
        collector_iter = iter(collector)
        best_iter_success = float("-inf")
        best_iter_return = float("-inf")
        best_iter_policy_state = None
        selected_arm_idx = None

        # 1. LLM 生成代码
        if llm_iter == 0:
            current_code = llm.generate_initial_reward(
                "无人机森林避障与目标抵达",
                state_desc,
                baseline_reward_reference=baseline_reward_ref,
                reward_param_context=reward_param_context,
            )
        else:
            current_code = llm.generate_feedback_with_relabel(
                reflection_buffer,
                current_code,
                current_success_rate,
                reward_param_context=reward_param_context,
                baseline_reward_reference=baseline_reward_ref,
            )

        current_arm_idx = _register_reward_arm(current_code, f"gen_{llm_iter + 1}")
        selected_arm_idx = ts_selector.select_arm()
        selected_arm = reward_arms[selected_arm_idx]
        selected_arm_train_iters = ts_selector.suggest_train_iters(
            selected_arm_idx,
            ppo_iters_per_llm,
            min_scale=arm_min_scale,
            max_scale=arm_max_scale,
            exploration_coef=arm_exploration_coef,
        )
        selected_arm_quality = ts_selector.posterior_mean(selected_arm_idx)
        selected_arm_acquisition = ts_selector.acquisition_score(
            selected_arm_idx,
            exploration_coef=arm_exploration_coef,
        )
        if selected_arm_idx != current_arm_idx:
            current_code = selected_arm["code"]
            llm.save_reward_code(current_code)
            print(
                f"🎲 Thompson 采样选择 arm[{selected_arm_idx}] 作为本代训练奖励（本轮新生成 arm[{current_arm_idx}] 保留在池中）。",
                flush=True,
            )
        else:
            print(
                f"🎲 Thompson 采样选择 arm[{selected_arm_idx}] 作为本代训练奖励。",
                flush=True,
            )
        print(
            f"📐 arm[{selected_arm_idx}] posterior_mean={selected_arm_quality:.3f}, acquisition={selected_arm_acquisition:.3f} -> 本代训练步数={selected_arm_train_iters}（基准={ppo_iters_per_llm}, scale=[{arm_min_scale:.2f}, {arm_max_scale:.2f}], exploration_coef={arm_exploration_coef:.2f}）",
            flush=True,
        )

        # 2. 挂载模块
        try:
            import generated_reward_fn
            importlib.reload(generated_reward_fn)
            _wrap_llm_reward_fn(generated_reward_fn)
            base_env.use_llm_reward = True
            base_env.llm_reward_module = generated_reward_fn
            print("🔗 LLM 奖励函数注入成功。")
        except Exception as e:
            print(f"⚠️ 注入失败: {e}")
            base_env.use_llm_reward = False

        policy.actor_opt.zero_grad(set_to_none=True)
        print("⏳ 开始 PPO 采样与训练（首次采样可能需要几十秒）...", flush=True)
        print(
            f"📦 采样配置: num_envs={env.num_envs}, train_every={train_every}, frames_per_batch={frames_per_batch}, ppo_iters={selected_arm_train_iters}",
            flush=True,
        )
        phase_start = time.time()

        # 3. PPO 训练
        is_tty = sys.stdout.isatty()
        # 默认关闭 tqdm，避免在部分终端里变成“多行堆叠”而不是单行刷新
        enable_tqdm = bool(cfg.get("enable_tqdm", False))
        progress_log_interval = int(cfg.get("progress_log_interval", 10))
        progress_render_interval_seconds = float(cfg.get("progress_render_interval_seconds", 0.5))
        last_render_time = 0.0

        def _extract_batch_return(td):
            """Try common reward keys and return scalar mean for logging."""
            candidates = [
                ("next", "agents", "reward"),
                ("next", "reward"),
                ("agents", "reward"),
                ("reward",),
            ]
            reward_tensor = None
            for key in candidates:
                try:
                    reward_tensor = td.get(key, None)
                except Exception:
                    reward_tensor = None
                if torch.is_tensor(reward_tensor):
                    break

            if not torch.is_tensor(reward_tensor):
                try:
                    stats = td.get(("next", "stats"), None)
                    if stats is not None:
                        reward_tensor = stats.get("reward", None)
                except Exception:
                    reward_tensor = None

            if not torch.is_tensor(reward_tensor):
                return float("nan")

            try:
                return float(tensor_to_env_scalar(reward_tensor).float().mean().detach().cpu().item())
            except Exception:
                return float("nan")

        def _extract_batch_success_rate(td):
            """Extract success rate from rollout batch stats as a scalar in [0, 1]."""
            success_tensor = None
            try:
                stats = td.get(("next", "stats"), None)
                if stats is not None:
                    success_tensor = stats.get("success", None)
            except Exception:
                success_tensor = None

            if not torch.is_tensor(success_tensor):
                return float("nan")

            try:
                success_rate = (tensor_to_env_scalar(success_tensor).float() > 0).float().mean()
                return float(success_rate.detach().cpu().item())
            except Exception:
                return float("nan")

        pbar = tqdm(
            range(selected_arm_train_iters),
            desc=f"Training Gen {llm_iter + 1}",
            dynamic_ncols=True,
            mininterval=0.2,
            miniters=1,
            file=sys.stdout,
            disable=not enable_tqdm,
        )
        try:
            for step_idx, _ in enumerate(pbar, start=1):
                collect_start = time.time()
                try:
                    data = next(collector_iter)
                except StopIteration:
                    break
                collect_elapsed = time.time() - collect_start

                if step_idx == 1:
                    print(
                        f"🕒 首个采样 batch 收集完成，耗时 {collect_elapsed:.1f}s。",
                        flush=True,
                    )
                    if collect_elapsed > float(cfg.get("first_batch_warn_seconds", 120.0)):
                        print(
                            "⚠️ 首批采样明显偏慢：通常是场景初始化/传感器预热或 frames_per_batch 过大导致。",
                            flush=True,
                        )

                episode_stats.add(data.to_tensordict())

                # LaRes：采样碰撞/成功样本入池
                with torch.no_grad():
                    stats = data["next", "stats"]
                    collision_mask = stats.get(
                        "reward_collision",
                        torch.zeros_like(stats["success"])
                    ).squeeze(-1) < -0.1
                    success_mask = stats["success"].squeeze(-1) > 0.5
                    combined_mask = collision_mask | success_mask

                    # `data` can include a time dimension [T, N].
                    # Select env indices from the last rollout step so indexing stays on env axis.
                    if combined_mask.dim() == 0:
                        combined_env_mask = combined_mask.view(1)
                    elif combined_mask.dim() == 1:
                        combined_env_mask = combined_mask
                    else:
                        combined_env_mask = combined_mask[-1].reshape(-1)

                    if combined_env_mask.any():
                        idx = torch.where(combined_env_mask)[0][0].item()
                        clean_states = base_env.get_clean_states()
                        obs_numpy = {
                            k: v[idx].detach().cpu().numpy()
                            for k, v in clean_states.items()
                        }
                        if collision_mask.dim() <= 1:
                            collision_flag = bool(collision_mask.reshape(-1)[idx].item())
                            success_flag = bool(success_mask.reshape(-1)[idx].item())
                        else:
                            collision_flag = bool(collision_mask[-1].reshape(-1)[idx].item())
                            success_flag = bool(success_mask[-1].reshape(-1)[idx].item())
                        reflection_buffer.add(
                            {"collision": collision_flag, "success": success_flag},
                            obs_numpy,
                            data["agents", "action"][idx].detach().cpu().numpy(),
                            [0],
                            None,
                            False
                        )

                policy.train_op(data.to_tensordict())

                elapsed = time.time() - phase_start
                now = time.time()
                batch_return = _extract_batch_return(data)
                batch_success_rate = _extract_batch_success_rate(data)
                batch_return_str = f"{batch_return:.3f}" if not math.isnan(batch_return) else "nan"
                batch_success_str = f"{batch_success_rate:.3f}" if not math.isnan(batch_success_rate) else "nan"
                global_ppo_step += 1

                # Keep a per-generation best policy snapshot for end-of-iteration video rendering.
                candidate_success = batch_success_rate if not math.isnan(batch_success_rate) else float("-inf")
                candidate_return = batch_return if not math.isnan(batch_return) else float("-inf")
                if (
                    (candidate_success > best_iter_success)
                    or (
                        candidate_success == best_iter_success
                        and candidate_return > best_iter_return
                    )
                ):
                    best_iter_success = candidate_success
                    best_iter_return = candidate_return
                    best_iter_policy_state = copy.deepcopy(policy.state_dict())

                if wandb.run is not None:
                    train_log = {
                        "train/collect_seconds": float(collect_elapsed),
                        "train/elapsed_seconds": float(elapsed),
                        "train/llm_iter": llm_iter + 1,
                        "train/ppo_step_in_iter": step_idx,
                        "train/ppo_global_step": global_ppo_step,
                    }
                    if not math.isnan(batch_return):
                        train_log["train/batch_return"] = batch_return
                        # Split curves by generation to compare reward updates directly in one chart.
                        train_log[f"train/batch_return_gen_{llm_iter + 1}"] = batch_return
                    if not math.isnan(batch_success_rate):
                        train_log["train/success_rate"] = batch_success_rate
                        # Per-generation success-rate curves.
                        train_log[f"train/success_rate_gen_{llm_iter + 1}"] = batch_success_rate

                    wandb.log(train_log, step=global_ppo_step)

                if is_tty and enable_tqdm:
                    if step_idx % 10 == 0:
                        pbar.set_postfix({
                            "step": f"{step_idx}/{selected_arm_train_iters}",
                            "elapsed_s": f"{elapsed:.1f}",
                            "collect_s": f"{collect_elapsed:.2f}",
                            "return": batch_return_str,
                            "succ": batch_success_str,
                        })
                elif is_tty and (not enable_tqdm):
                    # 终端单行覆盖刷新：不换行，避免日志堆叠
                    if (
                        step_idx == 1
                        or step_idx == selected_arm_train_iters
                        or (now - last_render_time) >= progress_render_interval_seconds
                    ):
                        pct = 100.0 * step_idx / max(1, selected_arm_train_iters)
                        sys.stdout.write(
                            f"\rTraining Gen {llm_iter + 1}: {step_idx}/{selected_arm_train_iters} "
                            f"({pct:5.1f}%) collect={collect_elapsed:.2f}s elapsed={elapsed:.1f}s "
                            f"return={batch_return_str} succ={batch_success_str}"
                        )
                        sys.stdout.flush()
                        last_render_time = now
                        if step_idx == selected_arm_train_iters:
                            sys.stdout.write("\n")
                            sys.stdout.flush()

                if (
                    step_idx == 1
                    or step_idx % progress_log_interval == 0
                    or step_idx == selected_arm_train_iters
                ):
                    print(
                        f"[Gen {llm_iter + 1}] PPO step {step_idx}/{selected_arm_train_iters} "
                        f"(collect {collect_elapsed:.2f}s, elapsed {elapsed:.1f}s, "
                        f"return {batch_return_str}, success {batch_success_str})",
                        flush=True,
                    )

                # 及时释放 batch，缓解显存压力
                del data
        finally:
            if is_tty and (not enable_tqdm):
                sys.stdout.write("\n")
                sys.stdout.flush()
            pbar.close()

        cleanup_cuda()

        # 4. 代末评估
        print(f"📊 第 {llm_iter + 1} 代评估...")
        restore_state = None
        if best_iter_policy_state is not None:
            restore_state = copy.deepcopy(policy.state_dict())
            policy.load_state_dict(best_iter_policy_state)
            best_success_text = f"{best_iter_success:.3f}" if best_iter_success > float("-inf") else "nan"
            best_return_text = f"{best_iter_return:.3f}" if best_iter_return > float("-inf") else "nan"
            print(
                f"🎬 第 {llm_iter + 1} 代视频将使用代内最优策略（success={best_success_text}, return={best_return_text}）。",
                flush=True,
            )

        eval_info = evaluate(seed=cfg.seed, eval_iter=llm_iter + 1)

        if restore_state is not None:
            policy.load_state_dict(restore_state)
        if best_iter_policy_state is not None:
            policy.load_state_dict(best_iter_policy_state)
            print(
                f"🌱 下一代训练将继承第 {llm_iter + 1} 代的最优策略权重。",
                flush=True,
            )
        current_success_rate = eval_info.get("eval/success_rate", 0.0)
        if selected_arm_idx is not None:
            ts_selector.update(selected_arm_idx, current_success_rate)
            print(
                f"📈 arm[{selected_arm_idx}] posterior 更新：success_rate={current_success_rate:.2%}",
                flush=True,
            )
            _prune_reward_arms(selected_arm_idx=selected_arm_idx)

        if wandb.run is not None:
            eval_log = {
                "eval/success_rate": float(current_success_rate),
                "eval/llm_iter": llm_iter + 1,
                "eval/best_success_rate_so_far": float(max(best_sr, current_success_rate)),
            }
            for k, v in eval_info.items():
                if k.startswith("eval/stats."):
                    eval_log[k] = float(v)
            if "eval/recording" in eval_info:
                eval_log["eval/recording"] = eval_info["eval/recording"]
                eval_log[f"eval/recording_gen_{llm_iter + 1}"] = eval_info["eval/recording"]
            wandb.log(eval_log, step=global_ppo_step)

        print(f"✅ 第 {llm_iter + 1} 代 success_rate = {current_success_rate:.2%}")

        if current_success_rate > best_sr:
            best_sr = current_success_rate
            torch.save(policy.state_dict(), os.path.join(run.dir, "best_llm_model.pt"))
            print(f"🌟 新纪录: {best_sr:.2%}")

        cleanup_cuda()

    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()