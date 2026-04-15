import sys
sys.path.append("/home/chendawww/workspace/rl-navibot/src")

import numpy as np
import matplotlib.pyplot as plt
from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf
import yaml
import pickle
from pathlib import Path
import re
from glob import glob
import rclpy
import math

from stable_baselines3 import SAC


BASE_DIR = Path("/home/chendawww/workspace/rl-navibot")
CONFIG_DIR = BASE_DIR / "src/decision/rl_agent/config"
TRY_DIR = Path(__file__).parent.parent
print(TRY_DIR)
MODEL_DIR = TRY_DIR.parent / "burger_navi_in_world" / "saved_models"
MIN_STEP = 39000
MAX_STEP = 81000
TEST_EPISODES = 30
TEST_SUITE_PATH = TRY_DIR / "scripts/test_suite_house_30eps.yaml"

CONFIG_PATH = CONFIG_DIR / "rl_env.house.config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ==================== ROS2 初始化 ====================
rclpy.init()
my_robot_urdf = fetch_tb3_urdf()
env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config, world="ttb3_house")


# ==================== 核心：读现成的，没有再调环境方法生成 ====================
def get_or_generate_test_suite(env, num_episodes, save_path):
    # 1. 先看有没有现成的
    if save_path.exists():
        print(f"检测到标准考卷: {save_path}，直接加载...")
        with open(save_path, "r") as f:
            return yaml.safe_load(f)["test_tasks"]

    # 2. 没有现成的，现造
    print(f"未找到考卷，调用环境采样方法生成 {num_episodes} 个固定任务...")
    
    # ⚠️ 注意：必须先 dummy reset 一次，激活环境的 self.np_random
    env.reset(seed=42) 
    
    tasks = []
    min_dist_req = env.dist_to_goal_gen_min 
    
    while len(tasks) < num_episodes:
        # 完全复用环境里的安全区采样逻辑！
        s_x, s_y = env._sample_safe_position()
        s_yaw = env.np_random.uniform(-math.pi, math.pi)
        g_x, g_y = env._sample_safe_position()
        
        # 过滤距离太近的废题
        if math.hypot(s_x - g_x, s_y - g_y) >= min_dist_req:
            tasks.append({
                "start_pos": [round(s_x, 3), round(s_y, 3), round(s_yaw, 3)],
                "goal_pos": [round(g_x, 3), round(g_y, 3)]
            })
            
    # 存起来，下次就不用再生了
    with open(save_path, "w") as f:
        yaml.dump({"test_tasks": tasks}, f, default_flow_style=False)
        
    print(f"✅ 考卷已生成并保存至: {save_path}")
    return tasks

# 拿到考题列表
test_tasks = get_or_generate_test_suite(env, TEST_EPISODES, TEST_SUITE_PATH)
start_list = [t["start_pos"] for t in test_tasks]
goal_list = [t["goal_pos"] for t in test_tasks]



# ==================== 遍历 Checkpoint 测试 ====================
SAC_MODEL_DIR = MODEL_DIR / "SAC"
checkpoint_data = [] 

for path in glob(str(SAC_MODEL_DIR / "sac_nav_model_*_steps.zip")):
    match = re.search(r"sac_nav_model_(\d+)_steps", path)
    if not match: continue
    step = int(match.group(1))
    if step >= MIN_STEP:
        checkpoint_data.append((step, path))

checkpoint_data.sort(key=lambda x: x[0])

best_model = None
best_metrics = {"success_rate": 0, "mean_reward": 0, "mean_steps": 0, "path": "", "step": 0}

all_model_results = [] 

print(f"\n开始评估 {len(checkpoint_data)} 个模型，共用同一套 {len(start_list)} 个任务...")

for step, checkpoint_path in checkpoint_data:
    print(f"\nTesting: {Path(checkpoint_path).name}")
    try:
        model = SAC.load(checkpoint_path, env=env, device="cuda")
        
        summary = env.run_episodes(
            model, 
            start_pos_list=start_list, 
            goal_pos_list=goal_list, 
            deterministic=True
        )
        
        current_sr = summary["success_rate"]
        current_mr = summary["mean_reward"]
        current_ms = summary["mean_steps"]
        
        # 👇 核心改动：把 summary 的 details (每一局的明细) 也原封不动存进来
        all_model_results.append({
            "train_step": step,
            "model_path": str(checkpoint_path),
            "success_rate": current_sr,
            "mean_reward": current_mr,
            "mean_steps": current_ms,
            "ep_details": summary["details"]  # 这是一个 list of dict
        })
        
        if (current_sr > best_metrics["success_rate"]) or \
           (current_sr == best_metrics["success_rate"] and current_mr > best_metrics["mean_reward"]):
            best_metrics.update({
                "success_rate": current_sr,
                "mean_reward": current_mr,
                "mean_steps": current_ms,
                "path": checkpoint_path,
                "step": step
            })
            best_model = model
            
    except Exception as e:
        print(f"  -> 失败: {e}")


# ==================== 保存全量记录 (YAML + PKL) ====================
RESULTS_YAML_PATH = TRY_DIR / "all_checkpoints_eval_results.yaml"
RESULTS_PKL_PATH = TRY_DIR / "all_checkpoints_eval_data.pkl"

# 1. 存 YAML（方便人看、方便快速找路径）
# 注意：pkl 里存了 details，yaml 存 details 会太长太乱，所以 yaml 我们只存聚合指标
yaml_data_for_human = [{k: v for k, v in r.items() if k != "ep_details"} for r in all_model_results]
with open(RESULTS_YAML_PATH, "w") as f:
    yaml.dump(yaml_data_for_human, f, default_flow_style=False, allow_unicode=True)
print(f"📄 聚合指标已保存至 (YAML): {RESULTS_YAML_PATH}")

# 2. 存 PKL（包含所有细节，方便 pandas/写论文做高级分析）
with open(RESULTS_PKL_PATH, "wb") as f:
    pickle.dump(all_model_results, f)
print(f"📦 全量明细数据已保存至 (PKL): {RESULTS_PKL_PATH}")


# ==================== 绘制性能演化曲线 ====================
if all_model_results:
    steps = [r["train_step"] for r in all_model_results]
    sr_list = [r["success_rate"] for r in all_model_results]
    mr_list = [r["mean_reward"] for r in all_model_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:green'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Success Rate (%)', color=color)
    ax1.plot(steps, sr_list, color=color, marker='o', linestyle='-', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Mean Reward', color=color)
    ax2.plot(steps, mr_list, color=color, marker='x', linestyle='--', label='Mean Reward')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Model Performance Evolution across Checkpoints (Fixed Test Suite)')
    fig.tight_layout()
    
    evo_plot_path = TRY_DIR / "plots" / "best_ttb3_world_SAC_for_ttb3_house" / "performance_evolution.png"
    evo_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(evo_plot_path)
    print(f"📈 模型性能演化曲线已保存至: {evo_plot_path}")
    plt.close(fig) 

print("\n=== 最优基座模型 ===")
print(f"训练步数: {best_metrics['step']}")
print(f"路径: {best_metrics['path']}")
print(f"成功率: {best_metrics['success_rate']:.1f}%")
print(f"平均奖励: {best_metrics['mean_reward']:.2f}")
print(f"平均步数: {best_metrics['mean_steps']:.1f}")

# ==================== 画图 (逻辑不变) ====================
if best_model is not None:
    final_summary = env.run_episodes(
        best_model, start_pos_list=start_list, goal_pos_list=goal_list, deterministic=True
    )
    
    details = final_summary["details"]
    episode_total_rewards = [r["reward"] for r in details]
    episode_successes = [r["success"] for r in details]
    episode_steps = [r["steps"] for r in details]
    
    mean_reward = final_summary["mean_reward"]
    success_rate = final_summary["success_rate"]
    mean_steps = final_summary["mean_steps"]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    axs[0].plot(episode_total_rewards, color='blue', marker='.', linestyle='-', linewidth=1, markersize=4)
    axs[0].axhline(y=mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    axs[0].set_title(f"Best Model Test (Fixed Suite) - Reward")
    axs[0].set_ylabel("Total Reward")
    axs[0].grid(True, alpha=0.3); axs[0].legend()
    
    axs[1].bar(range(len(episode_successes)), episode_successes, color=['green' if s else 'red' for s in episode_successes])
    axs[1].axhline(y=success_rate/100, color='blue', linestyle='--', label=f'Success: {success_rate:.1f}%')
    axs[1].set_title("Success per Task (Green=Success, Red=Fail)")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].grid(True, alpha=0.3); axs[1].legend()
    
    axs[2].plot(episode_steps, color='purple', marker='.', linestyle='-', linewidth=1, markersize=4)
    axs[2].axhline(y=mean_steps, color='orange', linestyle='--', label=f'Mean: {mean_steps:.1f}')
    axs[2].set_title("Steps per Task")
    axs[2].set_xlabel("Task Index")
    axs[2].set_ylabel("Steps")
    axs[2].grid(True, alpha=0.3); axs[2].legend()
    
    plt.tight_layout()
    
    save_dir = TRY_DIR / "plots" / "best_ttb3_world_SAC_for_ttb3_house"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / "best_model_fixed_suite_results.png"
    plt.savefig(plot_path)
    print(f"\n最优模型测试图表已保存至: {plot_path}")

env.close()
rclpy.shutdown()
print("Done!")
