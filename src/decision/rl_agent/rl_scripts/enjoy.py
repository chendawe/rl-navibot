import sys
sys.path.append("/home/chendawww/workspace/rl-navibot/src")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from decision.rl_agent.rl_agent.rl.algorithms import get_algorithm
from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config" / "rl.config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


my_robot_urdf = fetch_tb3_urdf()

# my_robot_urdf = None

# 2. 实例化环境
env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config)

# 直接实例化 Rule Baseline（不需要 load 模型）
model = get_algorithm("rule_baseline", env, config)

num_episodes = 50
episode_total_rewards = [] # 记录每一局的总分（用于画宏观图）
last_episode_step_rewards = [] # 记录最后一局的每步得分（用于画微观图）

print(f"开始评估 Rule Baseline，共 {num_episodes} 局...\n")

for ep in range(num_episodes):
    obs, _ = env.reset()
    ep_reward = 0
    step_rewards = []
    step = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = action[0]  # (1, 2) -> (2,)
        obs, reward, terminated, truncated, info = env.step(action)
        
        ep_reward += reward
        step_rewards.append(reward)
        step += 1
        done = terminated or truncated
        
        # 动态刷新控制台：不换行，只更新当前行的文字
        print(f"\rEp {ep+1:3d}/{num_episodes} | Steps: {step:4d} | Current Ep Reward: {ep_reward:8.2f}", end="", flush=True)
        
    # 这一局结束了，换行打印最终结果
    print(f" -> Final: {ep_reward:8.2f} ({'Success' if info.get('is_success') else 'Fail'})")
    
    episode_total_rewards.append(ep_reward)
    # 保存最后一局的 step 级别数据
    if ep == num_episodes - 1:
        last_episode_step_rewards = step_rewards

# ==================== 评估结束，开始画图 ====================
print("\n正在生成 Reward 变化图表...")

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 图 1：宏观变化（每一局的总分）
axs[0].plot(episode_total_rewards, color='blue', marker='.', linestyle='-', linewidth=1, markersize=4)
axs[0].axhline(y=np.mean(episode_total_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_total_rewards):.2f}')
axs[0].set_title(f"Rule Baseline: Episode Total Reward (Avg: {np.mean(episode_total_rewards):.2f})")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Total Reward")
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# 图 2：微观变化（最后一局的每一步得分）
# 累加画图，能更清楚地看到分数的走势
cumulative_rewards = np.cumsum(last_episode_step_rewards)
axs[1].plot(last_episode_step_rewards, color='gray', alpha=0.5, label='Step Reward')
axs[1].plot(cumulative_rewards, color='green', linewidth=2, label='Cumulative Reward')
axs[1].set_title(f"Rule Baseline: Step Reward Change (Last Episode)")
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Reward")
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.tight_layout()

# 保存图片
save_dir = BASE_DIR / "results" / "plots"
save_dir.mkdir(parents=True, exist_ok=True)
plot_path = save_dir / "rule_baseline_reward_change.png"
plt.savefig(plot_path)
print(f"图表已保存至: {plot_path}")

# 弹出窗口显示（如果是在有界面的服务器上跑的话）
# plt.show()
