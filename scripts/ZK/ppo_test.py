import argparse
import importlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded


# Ensure the repository root (OmniDrones) is importable when run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omni_drones.learning import ALGOS
from omni_drones.learning.ppo.ppo import PPOConfig as RepoPPOConfig


def build_specs(num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
    observation_spec = Composite(
        {
            "agents": Composite(
                {
                    "observation": Unbounded((1, obs_dim), device=device),
                }
            )
        }
    ).expand(num_envs).to(device)

    action_spec = Composite(
        {
            "agents": Composite(
                {
                    "action": Unbounded((1, action_dim), device=device),
                }
            )
        }
    ).expand(num_envs).to(device)

    reward_spec = Composite(
        {
            "agents": Composite(
                {
                    "reward": Unbounded((1, 1), device=device),
                }
            )
        }
    ).expand(num_envs).to(device)
    return observation_spec, action_spec, reward_spec


def make_vec_env(env_id: str, num_envs: int):
    return gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])


@torch.no_grad()
def collect_repo_rollout(policy, vec_env, device: torch.device, horizon: int):
    num_envs = vec_env.num_envs
    obs, _ = vec_env.reset()

    obs_hist = []
    action_hist = []
    logp_hist = []
    value_hist = []
    reward_hist = []
    done_hist = []
    next_obs_hist = []

    for _ in range(horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(1)
        td = TensorDict(
            {
                ("agents", "observation"): obs_t,
            },
            batch_size=[num_envs],
            device=device,
        )

        policy(td)
        action_t = td[("agents", "action")]
        env_action = action_t.squeeze(1).cpu().numpy().astype(np.float32)

        next_obs, reward, terminated, truncated, _ = vec_env.step(env_action)
        done = np.logical_or(terminated, truncated)

        obs_hist.append(obs_t)
        action_hist.append(action_t)
        logp_hist.append(td["sample_log_prob"])
        value_hist.append(td["state_value"])
        reward_hist.append(
            torch.as_tensor(reward, dtype=torch.float32, device=device).view(num_envs, 1, 1)
        )
        done_hist.append(
            torch.as_tensor(done, dtype=torch.float32, device=device).view(num_envs, 1)
        )
        next_obs_hist.append(
            torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(1)
        )

        obs = next_obs

    obs_tensor = torch.stack(obs_hist, dim=1)
    action_tensor = torch.stack(action_hist, dim=1)
    logp_tensor = torch.stack(logp_hist, dim=1)
    value_tensor = torch.stack(value_hist, dim=1)
    reward_tensor = torch.stack(reward_hist, dim=1)
    done_tensor = torch.stack(done_hist, dim=1)
    next_obs_tensor = torch.stack(next_obs_hist, dim=1)

    return TensorDict(
        {
            ("agents", "observation"): obs_tensor,
            ("agents", "action"): action_tensor,
            "sample_log_prob": logp_tensor,
            "state_value": value_tensor,
            "next": TensorDict(
                {
                    ("agents", "observation"): next_obs_tensor,
                    ("agents", "reward"): reward_tensor,
                    "terminated": done_tensor,
                },
                batch_size=[num_envs, horizon],
                device=device,
            ),
        },
        batch_size=[num_envs, horizon],
        device=device,
    )


@torch.no_grad()
def evaluate_repo_policy(policy, env_id: str, device: torch.device, n_eval_episodes: int, seed: int):
    env = gym.make(env_id)
    returns = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            td = TensorDict({("agents", "observation"): obs_t}, batch_size=[1], device=device)
            dist = policy.actor.get_dist(td)
            action = dist.base_dist.loc.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


def run_repo_ppo(args):
    # 如果电脑有 3060 显卡，使用 cuda 加速神经网络计算
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    probe_env = gym.make(args.env_id)
    if not isinstance(probe_env.action_space, gym.spaces.Box):
        raise ValueError("Repo PPO in omni_drones.learning.ppo.ppo is continuous-action; please use a Box action env.")
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    action_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    observation_spec, action_spec, reward_spec = build_specs(
        args.repo_num_envs,
        obs_dim,
        action_dim,
        device,
    )

    repo_cfg = RepoPPOConfig(
        train_every=args.repo_horizon,
        ppo_epochs=args.repo_ppo_epochs,
        num_minibatches=args.repo_num_minibatches,
    )
    policy = ALGOS["ppo"](
        repo_cfg,
        observation_spec,
        action_spec,
        reward_spec,
        device=device,
    )

    pre_ret = evaluate_repo_policy(policy, args.env_id, device, args.eval_episodes, args.seed)
    print(f"[Repo PPO][Before] mean_eval_return={pre_ret:.2f}")

    vec_env = make_vec_env(args.env_id, args.repo_num_envs)
    steps_per_update = args.repo_num_envs * args.repo_horizon
    num_updates = max(1, args.repo_total_steps // steps_per_update)

    for update in range(1, num_updates + 1):
        rollout = collect_repo_rollout(policy, vec_env, device, args.repo_horizon)
        info = policy.train_op(rollout)

        if update == 1 or update % args.repo_log_interval == 0 or update == num_updates:
            mean_ret = evaluate_repo_policy(policy, args.env_id, device, args.eval_episodes, args.seed + 1000)
            print(
                f"[Repo PPO][Update {update:03d}/{num_updates:03d}] "
                f"mean_eval_return={mean_ret:.2f}, "
                f"policy_loss={info['policy_loss']:.4f}, value_loss={info['value_loss']:.4f}, "
                f"approx_kl={info.get('approx_kl', 0.0):.5f}, clip_fraction={info.get('clip_fraction', 0.0):.3f}"
            )

    vec_env.close()
    post_ret = evaluate_repo_policy(policy, args.env_id, device, args.eval_episodes, args.seed + 2000)
    print(f"[Repo PPO][After ] mean_eval_return={post_ret:.2f}")
    return pre_ret, post_ret


def run_sb3_ppo(args):
    try:
        sb3 = importlib.import_module("stable_baselines3")
        sb3_eval_mod = importlib.import_module("stable_baselines3.common.evaluation")
        sb3_envutil_mod = importlib.import_module("stable_baselines3.common.env_util")
        PPO = sb3.PPO
        evaluate_policy = sb3_eval_mod.evaluate_policy
        make_vec_env = sb3_envutil_mod.make_vec_env
    except ImportError:
        print("[SB3] stable-baselines3 is not installed.")
        print("[SB3] Install command:")
        print("/home/descfly/anaconda3/envs/omni-5.1/bin/pip install stable-baselines3")
        return None, None

    vec_env = make_vec_env(args.env_id, n_envs=args.sb_num_envs, seed=args.seed)
    eval_env = gym.make(args.env_id)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.sb_lr,
        n_steps=args.sb_n_steps,
        batch_size=args.sb_batch_size,
        n_epochs=args.sb_n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        seed=args.seed,
        verbose=0,
    )

    pre_ret, _ = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
    print(f"[SB3 PPO ][Before] mean_eval_return={pre_ret:.2f}")

    model.learn(total_timesteps=args.sb_total_steps, progress_bar=False)

    post_ret, _ = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
    print(f"[SB3 PPO ][After ] mean_eval_return={post_ret:.2f}")

    vec_env.close()
    eval_env.close()
    return float(pre_ret), float(post_ret)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SB3 PPO and repository PPO on BipedalWalker-v3.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "sb3", "repo"])
    # [修改点 1]：默认环境改为双足机器人
    parser.add_argument("--env-id", type=str, default="BipedalWalker-v3") 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)

    # [修改点 2]：双足机器人很难，SB3 需要 100 万步才能走好，同时适配多 CPU 训练
    parser.add_argument("--sb-total-steps", type=int, default=1_000_000)
    parser.add_argument("--sb-num-envs", type=int, default=16) 
    parser.add_argument("--sb-lr", type=float, default=3e-4)
    parser.add_argument("--sb-n-steps", type=int, default=1024)
    parser.add_argument("--sb-batch-size", type=int, default=256)
    parser.add_argument("--sb-n-epochs", type=int, default=10)

    # [修改点 3]：大幅降低环境并发数，防止 CPU 爆炸，同时拉长单环境采样步数
    parser.add_argument("--repo-total-steps", type=int, default=1_000_000)
    parser.add_argument("--repo-num-envs", type=int, default=16)  # 从 256 降到 16
    parser.add_argument("--repo-horizon", type=int, default=1024) # 从 32 提升到 1024
    parser.add_argument("--repo-ppo-epochs", type=int, default=10)
    parser.add_argument("--repo-num-minibatches", type=int, default=32)
    parser.add_argument("--repo-log-interval", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sb3_result = None
    repo_result = None

    if args.mode in ("all", "sb3"):
        sb3_result = run_sb3_ppo(args)

    if args.mode in ("all", "repo"):
        repo_result = run_repo_ppo(args)

    print("\n=== Summary ===")
    if sb3_result is not None:
        print(f"SB3 PPO : before={sb3_result[0]}, after={sb3_result[1]}")
    if repo_result is not None:
        print(f"Repo PPO: before={repo_result[0]}, after={repo_result[1]}")


if __name__ == "__main__":
    main()