# ================= test_base_lidar.py =================
import logging
import os
import hydra
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf
from setproctitle import setproctitle

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
import wandb

# 导入上面重写的环境模块，确保它被注册
# 导入上面重写的环境模块，确保它被注册到 OmniDrones


# ================= 论文专属：双流网络架构 =================
class PaperActorNet(nn.Module):
    def __init__(self, state_dim, lidar_dim, out_dim):
        super().__init__()
        self.state_dim = state_dim
        self.lidar_dim = lidar_dim
        
        # 激光雷达特征编码器 (Lidar Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(lidar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 状态特征融合与输出 (MLP Fusion)
        self.fusion = nn.Sequential(
            nn.Linear(state_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, obs):
        # 将传入的扁平化 obs 切割为 state 和 lidar
        state = obs[..., :self.state_dim]
        lidar = obs[..., self.state_dim:]
        
        # 走两条不同的网络流然后拼接
        lidar_feat = self.encoder(lidar)
        fusion_input = torch.cat([state, lidar_feat], dim=-1)
        return self.fusion(fusion_input)

# ================= 主训练逻辑 =================
@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # 强制将调用的环境指定为新编写的论文版雷达环境
    cfg.task.name = "ForestLidar"
    cfg.sim.enable_replicator = True
    cfg.sim.enable_viewport = True

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))
    import omni_drones.envs.single.forest_lidar
    
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
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
        
        # 🔑 【核心注入】将策略网络替换为论文架构
        # 为了兼容不同的分布输出头（比如多维动作的均值和方差），先获取原网络的输出维度
        dummy_in = torch.zeros(1, 15 + 3200, device=base_env.device)
        
        # 替换 Actor
        orig_actor = policy.actor_critic.actor_module[0]
        out_dim_actor = orig_actor(dummy_in).shape[-1]
        policy.actor_critic.actor_module[0] = PaperActorNet(15, 3200, out_dim_actor).to(base_env.device)
        
        # 替换 Critic (Value Network)
        orig_critic = policy.actor_critic.critic_module[0]
        out_dim_critic = orig_critic(dummy_in).shape[-1]
        policy.actor_critic.critic_module[0] = PaperActorNet(15, 3200, out_dim_critic).to(base_env.device)
        
        logging.info("✅ 成功注入论文指定的 Encoder-Fusion 双流网络架构！")
        
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    max_return = -float("inf")
    last_best_ckpt_path = None

    stats_keys = [k for k in base_env.observation_spec.keys(True, True) if isinstance(k, tuple) and k[0]=="stats"]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env, policy=policy, frames_per_batch=frames_per_batch,
        total_frames=total_frames, device=cfg.sim.device, return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(seed: int=0, exploration_type: ExplorationType=ExplorationType.MODE):
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=2000, policy=policy, callback=render_callback,
                auto_reset=True, break_when_any_done=False, return_contiguous=False,
            )

        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {k: take_first_episode(v) for k, v in trajs[("next", "stats")].cpu().items()}
        info = {"eval/stats." + k: torch.mean(v.float()).item() for k, v in traj_stats.items()}

        ep_lens = traj_stats.get("episode_len")
        if ep_lens is not None:
            max_len = getattr(base_env, "max_episode_length", 800)
            success_count = int((ep_lens >= (max_len - 5)).sum().item())
            total_drones = len(ep_lens)
            success_rate = success_count / total_drones
            
            print("\n" + "🏆" * 25)
            print(f"🏁 [录像考核战报] 共有 {success_count} / {total_drones} 架无人机成功抵达终点！")
            print(f"📈 [环境通过率]   {success_rate * 100:.1f} %")
            print("🏆" * 25 + "\n")
            info["eval/success_rate"] = success_rate

        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5/ (cfg.sim.dt * cfg.sim.substeps), format="mp4"
        )
        return info

    if max_iters > 0:
        total_len = max_iters
    elif total_frames > 0:
        total_len = total_frames // frames_per_batch
    else:
        total_len = None

    last_return = 0.0
    pbar = tqdm(collector, total=total_len, dynamic_ncols=True)
    env.train()
    
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
            if "train/stats.return" in info:
                current_return = info["train/stats.return"]
                last_return = current_return
                if current_return > max_return:
                    max_return = current_return
                    try:
                        ckpt_path = os.path.join(run.dir, f"checkpoint_best_return_{max_return:.2f}.pt")
                        torch.save(policy.state_dict(), ckpt_path)
                        if last_best_ckpt_path is not None and last_best_ckpt_path != ckpt_path:
                            try:
                                if os.path.exists(last_best_ckpt_path):
                                    os.remove(last_best_ckpt_path)
                            except OSError:
                                pass
                        last_best_ckpt_path = ckpt_path
                    except AttributeError:
                        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
            except AttributeError:
                pass

        run.log(info)
        pbar.set_postfix({"fps": f"{collector._fps:.1f}", "frames": collector._frames, "return": f"{last_return:.2f}"})
        
        if max_iters > 0 and i >= max_iters - 1:
            break

    if last_best_ckpt_path is not None and os.path.exists(last_best_ckpt_path):
        try:
            policy.load_state_dict(torch.load(last_best_ckpt_path, weights_only=True))
        except Exception:
            pass

    try:
        info = {"env_frames": collector._frames}
        info.update(evaluate()) 
        run.log(info)
    except Exception as e:
        print(f"\n[!] 最终评估跳过: {e}\n")

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)
    except AttributeError:
        pass

    wandb.finish()
    simulation_app.close()

if __name__ == "__main__":
    main()