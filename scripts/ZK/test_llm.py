import torch
import os
from llm_manager import LLMManager

def test_llm_setup():
    print("🚀 启动 LaRes LLM 接口测试 (VectorEngine 版)...")

    # 1. 实例化 LLMManager
    # 填入你刚才提供的专属中转 API 信息
    llm = LLMManager()

    # 2. 模拟环境提供给大模型的状态描述（必须与 get_clean_states 返回的 key 一致）
    state_desc = """
    "curr_pos": 当前位置 [num_envs, 3],
    "target_pos": 目标位置 [num_envs, 1, 3],
    "curr_dist": 距目标距离 [num_envs, 1],
    "v_norm": 速度大小 [num_envs, 1],
    "min_lidar_dist": 距离最近障碍物的距离 [num_envs, 1],
    "z_pos": 高度 [num_envs, 1]
    """

    print(f"📡 正在尝试连接中转服务器: {llm.client.base_url}")
    print(f"🤖 目标模型: {llm.model_name}")

    # 3. 测试第一轮：生成初始奖励函数
    print("\n--- [测试 1: 代码生成] ---")
    try:
        # 这个函数会去读取 prompts/initial_system.txt 等文件
        code = llm.generate_initial_reward(
            task_desc="无人机在密林中避开障碍物并安全抵达目标点", 
            state_info=state_desc
        )
        
        print("\n✅ API 连接及提取逻辑成功！")
        print("-" * 50)
        print("🔍 生成的代码片段预览:")
        print("\n".join(code.split("\n")[:15])) # 只打印前15行预览
        print("-" * 50)
        
        # 4. 关键：张量逻辑校验
        if "torch." in code and "compute_reward" in code:
            print("💎 校验通过: 代码中包含 PyTorch 张量操作和核心函数名。")
            
            # 检查是否有 python 的 if 判断（如果不小心写了，会提醒大模型重写）
            if "if " in code and "torch.where" not in code:
                print("⚠️ 警告: 代码中发现 Python 'if' 关键字，这在张量并行中可能报错，请关注后续进化迭代。")
        else:
            print("❌ 错误: 提取的内容似乎不是有效的 Python 代码。")

    except Exception as e:
        print(f"❌ 测试过程中发生错误!")
        print(f"错误详情: {str(e)}")
        print("\n💡 排查指南:")
        print("1. 检查网络是否能访问 api.vectorengine.ai")
        print("2. 确认 prompts/ 文件夹下 5 个 .txt 文件没有丢失")
        print("3. 确认 API Key 余额充足且未过期")

    # 5. 测试本地文件落盘
    if os.path.exists("generated_reward_fn.py"):
        print("\n📂 文件落盘测试: 成功！(generated_reward_fn.py 已创建)")
    else:
        print("\n📂 文件落盘测试: 失败！")

if __name__ == "__main__":
    test_llm_setup()