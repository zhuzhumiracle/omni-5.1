import logging
import os
import time

import hydra
import torch

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
)
from omni_drones.utils.torchrl import EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


FILE_PATH = os.path.dirname(__file__)

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    
    OmegaConf.set_struct(cfg, False)
    # 在 init_simulation_app(cfg) 上方添加这两行：
    cfg.sim.enable_replicator = True
    cfg.sim.enable_viewport = True 
    
    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # if cfg.task.get("history", False):
    #     # transforms.append(History([("info", "drone_state"), ("info", "prev_action")]))
    #     transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
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
    # --- 新增：加载训练好的模型权重 ---
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"[*] 正在从以下路径加载预训练模型: {checkpoint_path}")
        # 加载权重文件
        state_dict = torch.load(checkpoint_path, map_location=base_env.device)
        
        # 将权重注入到策略网络中
        # 注意：如果是从 wandb 下载的，可能需要 policy.load_state_dict(state_dict) 
        # 或者 policy.load_state_dict(state_dict['model_state_dict'])，取决于你保存时的格式
        policy.load_state_dict(state_dict)
        
        # 极其重要：将策略设为评估模式，关闭随机探索
        policy.eval() 
        print("[+] 模型权重加载成功！")
    else:
        print(f"[!] 警告：未找到有效的 checkpoint_path 或路径不存在: {checkpoint_path}")
    # --------------------------------
    frames_per_batch = env.num_envs * 32

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=cfg.total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
# 1. 开启底层环境的渲染开关
    base_env.enable_render(True)
    
    # 2. 将底层环境和 TorchRL 包装环境都切换为评估模式
    base_env.eval()
    env.eval()
    
    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

    simulation_app.close()


if __name__ == "__main__":
    main()
