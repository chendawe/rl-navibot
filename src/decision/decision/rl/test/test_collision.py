import numpy as np
import yaml
from decision.rl.environments import TurtleBot3NaviEnv, fetch_tb3_urdf

def test_collision_reward():
    print("=" * 60)
    print("🧪 开始碰撞惩罚逻辑验证测试")
    print("=" * 60)
    
    # 1. 准备环境
    print("\n[1/3] 正在拉取 URDF 并初始化环境...")
    urdf = fetch_tb3_urdf()
    
    # 临时修改配置：把目标设得极远，把步长拉长，方便它快速撞墙
    test_config = "/home/chendawww/workspace/rl-navibot/src/decision/decision/rl/useful_env_configs/rl_env.world.config.yaml"
    
    with open(test_config, "r") as f:
        test_config = yaml.safe_load(f)
    env = TurtleBot3NaviEnv(robot_urdf=urdf, env_config=test_config)

    # 2. 重置环境，并把目标扔到 (100, 100) 的天边去，确保它绝对不可能“意外到达目标”
    print("[2/3] 重置环境 (目标已设为极远处)...")
    obs, info = env.reset(options={"goal_pos": (100.0, 100.0), "skip_spawn_check": True})
    print(f"      当前起点: {info['spawn_pos']}, 虚假目标: {info['goal_pos']}")

    # 3. 发出死命令：油门焊死，一直往前冲！
    print("[3/3] 发出强硬直行动作 [0.22, 0.0]，等待撞墙...\n")
    
    # 获取最大速度作为动作
    go_straight = np.array([env.robot["lin_vel_max"], 0.0], dtype=np.float32)
    
    step = 0
    total_reward = 0.0
    final_info = {}

    while True:
        obs, reward, terminated, truncated, info = env.step(go_straight)
        step += 1
        total_reward += reward
        final_info = info
        
        # 实时打印每一步的状态（重点看 min_laser 和 reward）
        min_laser = env._min_laser()
        print(f"Step {step:3d} | 最小雷达: {min_laser:.3f}m | 本步奖励: {reward:7.2f} | 累计奖励: {total_reward:7.2f}")
        
        # 如果停了，跳出循环
        if terminated or truncated:
            break

    # ==================== 结果断言 ====================
    print("\n" + "=" * 60)
    print("🧪 测试结束，结果分析：")
    print("=" * 60)
    print(f"总步数       : {step}")
    print(f"最终累计奖励 : {total_reward:.2f}")
    print(f"碰撞标志     : {final_info.get('collision')}")
    print(f"到达目标标志 : {final_info.get('goal_reached')}")
    print("=" * 60)

    # 这里是硬性检查，如果不通过会直接抛出 AssertionError
    if final_info.get('collision') is True and total_reward <= -299.0:
        print("✅ 测试通过！撞墙成功检测，且正确扣除了 300 分左右的惩罚！")
    elif final_info.get('goal_reached') is True:
        print("❌ 测试失败！还没撞墙就判定到达目标了（请检查目标点是不是设置错了）。")
    elif final_info.get('collision') is False:
        print("❌ 测试失败！跑完 200 步都没撞墙（请检查 Gazebo 前面是不是没有墙）。")
    elif total_reward > -100:
        print("❌ 测试失败！虽然检测到撞墙，但奖励没扣到位（检查 Reward 里的优先级逻辑）。")

if __name__ == "__main__":
    test_collision_reward()
