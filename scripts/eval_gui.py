import os
import torch
import hydra
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType

# 1. 设置环境变量
os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"
os.environ["OMNI_EXT_RELOAD_ENABLED"] = "0"

@hydra.main(config_path=os.path.dirname(__file__), config_name="eval_gui", version_base=None)
def main(cfg):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass
        
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # 2. 强制开启 GUI
    cfg.headless = False
    cfg.sim.enable_replicator = False # 避免退出崩溃
    cfg.sim.enable_viewport = True

    # 3. 强制对齐训练参数 (必须与 train.py 完全一致)
    cfg.task.lidar_range = 4.0
    cfg.task.lidar_vfov = [-10.0, 20.0]
    
    # 初始化仿真器
    simulation_app = init_simulation_app(cfg)

    # 4. 构建环境 (复制自 train.py)
    from omni_drones.envs.isaac_env import IsaacEnv
    from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
    from omni_drones.utils.torchrl.transforms import FromMultiDiscreteAction, FromDiscreteAction, ravel_composite
    from omni_drones.learning import ALGOS

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=False)

    transforms = [InitTracker()]

    # 必须保证 transform 顺序和类型与训练时一致
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))

    env = TransformedEnv(base_env, Compose(*transforms))

    # 5. 加载模型
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=base_env.device
    )

    # !!! 请在这里填入您的最佳模型路径 !!!
    checkpoint_path = "/home/descfly/visualandlidar/base/omni-5.1/OmniDrones/scripts/wandb/run-20260219_215847-zp1j3gmn/files/checkpoint_best_return_2939.65.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"\n[*] 正在加载模型: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=base_env.device)
        policy.load_state_dict(state_dict)
        print("[+] 模型加载成功！\n")
    else:
        print(f"\n[!!!] 错误: 找不到模型文件: {checkpoint_path}")
        print("无人机将随机乱飞！请检查路径是否正确。\n")

    # 6. 切换到评估模式
    policy.eval()
    env.eval()
    base_env.enable_render(True)
    
    # 7. 运行推演
    print("🚀 开始可视化，按 Ctrl+C 退出...")
    
    try:
        # 使用 ExplorationType.MODE 确保确定性行为 (不随机采样)
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            while simulation_app.is_running():
                # 使用 rollout 而不是手动 step，保证逻辑与 evaluate() 一致
                env.rollout(
                    max_steps=1000,
                    policy=policy,
                    auto_reset=True,
                    break_when_any_done=False
                )
    except KeyboardInterrupt:
        pass

    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()