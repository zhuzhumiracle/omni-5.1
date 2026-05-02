import hydra
import torch
from omegaconf import OmegaConf

# 必须在最前面初始化引擎
from omni_drones import init_simulation_app

@hydra.main(version_base=None, config_path="", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    
    # ================= 强制开启可视化 =================
    cfg.sim.headless = False
    cfg.headless = False
    cfg.sim.enable_viewport = True
    
    # 启动 Isaac Sim 仿真引擎
    simulation_app = init_simulation_app(cfg)

    # 导入环境注册表
    from omni_drones.envs.isaac_env import IsaacEnv
    # 这一句是为了让系统读取并注册你写的 Forest 类
    import omni_drones.envs.single.forest  

    print("\n" + "="*50)
    print("🚀 [雷达测试模式] 正在生成森林和无人机...")
    print("="*50 + "\n")

    # 实例化你的 Forest 环境
    base_env = IsaacEnv.REGISTRY["Forest"](cfg, headless=False)
    base_env.enable_render(True)
    
    # 初始化无人机位置
    tensordict = base_env.reset()

    print("\n" + "🌟"*20)
    print("👀 仿真已启动！请在弹出的 Isaac Sim 窗口中观察：")
    print("   -> 你可以用鼠标【右键 + WASD】游历场景")
    print("   -> 观察无人机周围是否有 360 度的射线和碰撞点云")
    print("🌟"*20 + "\n")

    # 运行 1000 步（大约十几秒），让无人机做随机动作，方便你观察射线动态
    for _ in range(1000):
        # 随机乱飞
        action = base_env.action_spec.rand()
        tensordict[("agents", "action")] = action
        
        # 步进环境
        tensordict = base_env.step(tensordict)
        
        # 强制刷新画面
        simulation_app.update()

    print("\n✅ 动态测试结束，画面已定格。你可以继续用鼠标随意观察。")
    print("按下终端的 Ctrl+C 即可完全退出。")
    
    # 死循环保持窗口不关，让你看个够
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()

if __name__ == "__main__":
    main()