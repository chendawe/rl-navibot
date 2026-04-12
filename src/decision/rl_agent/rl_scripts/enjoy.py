import sys
# sys.path.append("~/workspace/rl-navibot/src")
sys.path.append("/home/chendawww/workspace/rl-navibot/src")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from decision.rl_agent.rl_agent.rl.algorithms import get_algorithm
from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf
import yaml
from pathlib import Path

from stable_baselines3 import SAC, PPO

BASE_DIR = Path(__file__).parent.parent
# DECISION_PATH=
# MODEL_PATH

CONFIG_PATH = BASE_DIR / "config" / "rl.config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


my_robot_urdf = fetch_tb3_urdf()

# my_robot_urdf = None

# 2. 实例化环境
env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config)

MODEL_PATH = BASE_DIR / "saved_models/SAC" / "sac_nav_model_148000_steps"

MODEL_PATH = "/home/chendawww/workspace/rl-navibot/src/decision/rl_agent/saved_models/SAC/sac_nav_model_148000_steps"
MODEL_PATH = "/home/chendawww/workspace/rl-navibot/src/decision/rl_agent/saved_models/SAC/sac_nav_model_100000_steps"
model = SAC.load(MODEL_PATH, env=env, device="cuda")

# MODEL_PATH = "/home/chendawww/workspace/rl-navibot/src/decision/rl_agent/saved_models/PPO/ppo_nav_model_120000_steps"
# model = PPO.load(MODEL_PATH, env=env, device="cuda")
print(f"模型训练步数: {model.num_timesteps}")  # 如果打印 148000，说明文件没问题

num_episodes = 50
episode_total_rewards = []
episode_successes = []
episode_steps = []
last_episode_step_rewards = []

print(f"开始评估训练模型，共 {num_episodes} 局...\n")

for ep in range(num_episodes):
    obs, info = env.reset()
    print(f"观测数据形状: {obs.shape}")
    print(f"观测数据前 10 个值: {obs[:10]}")

    ep_reward = 0
    step_rewards = []
    step = 0
    done = False
    
    print(f"spawn xy: {info['spawn_pos']}; goal xy: {info['goal_pos']}")
    
    while not done:

        action, _ = model.predict(obs, deterministic=True)
        # # 换成这行（直接调用底层策略网络，跳过 predict 的缩放）：
        # import torch
        # obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        # action = model.policy(obs_tensor, deterministic=True)[0].cpu().detach().numpy()
        action = np.array(action).flatten()
        print(f"预测动作: 线速度={action[0]:.3f}, 角速度={action[1]:.3f}")
        
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"观测数据形状: {obs.shape}")
        print(f"观测数据前 10 个值: {obs[:10]}")
        ep_reward += reward
        step_rewards.append(reward)
        step += 1
        done = terminated or truncated
        
        # 动态刷新控制台
        print(f"\rEp {ep+1:3d}/{num_episodes} | Steps: {step:4d} | Current Ep Reward: {ep_reward:8.2f}", end="", flush=True)
    
    # 记录结果
    episode_total_rewards.append(ep_reward)
    episode_successes.append(info.get('goal_reached', False))
    episode_steps.append(step)
    
    # 保存最后一局的 step 级别数据
    if ep == num_episodes - 1:
        last_episode_step_rewards = step_rewards
    
    # 打印最终结果
    print(f" -> Final: {ep_reward:8.2f} | Steps: {step:4d} | {'Success' if info.get('goal_reached') else 'Fail'}")

# ==================== 评估结束，开始画图 ====================
print("\n正在生成评估图表...")

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# 图 1：总奖励变化
axs[0].plot(episode_total_rewards, color='blue', marker='.', linestyle='-', linewidth=1, markersize=4)
mean_reward = np.mean(episode_total_rewards)
axs[0].axhline(y=mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.2f}')
axs[0].set_title(f"Test Results: Episode Total Reward (Avg: {mean_reward:.2f})")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Total Reward")
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# 图 2：成功率
success_rate = np.mean(episode_successes) * 100
axs[1].bar(range(num_episodes), episode_successes, color=['green' if s else 'red' for s in episode_successes])
axs[1].axhline(y=success_rate, color='blue', linestyle='--', label=f'Success Rate: {success_rate:.1f}%')
axs[1].set_title(f"Test Results: Success Rate ({success_rate:.1f}%)")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Success (1=Success, 0=Fail)")
axs[1].set_ylim(-0.1, 1.1)
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# 图 3：步数变化
axs[2].plot(episode_steps, color='purple', marker='.', linestyle='-', linewidth=1, markersize=4)
mean_steps = np.mean(episode_steps)
axs[2].axhline(y=mean_steps, color='orange', linestyle='--', label=f'Mean Steps: {mean_steps:.1f}')
axs[2].set_title(f"Test Results: Steps per Episode (Avg: {mean_steps:.1f})")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Steps")
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.tight_layout()

# 保存图片
save_dir = BASE_DIR / "saved_models/SAC" / "plots"
save_dir.mkdir(parents=True, exist_ok=True)
plot_path = save_dir / "rl_model_test_results.png"
plt.savefig(plot_path)
print(f"评估图表已保存至: {plot_path}")

# 打印汇总统计
print("\n=== 评估汇总 ===")
print(f"总测试局数: {num_episodes}")
print(f"平均总奖励: {mean_reward:.2f}")
print(f"成功率: {success_rate:.1f}%")
print(f"平均步数: {mean_steps:.1f}")
print(f"成功局数: {sum(episode_successes)}/{num_episodes}")





        # import torch
        # import numpy as np

        # print("\n" + "=" * 60)
        # print("SAC 模型底层诊断")
        # print("=" * 60)

        # # ============ 方法1：直接读取网络内部参数 ============
        # try:
        #     obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
        #     actor = model.policy.actor

        #     with torch.no_grad():
        #         # 提取特征
        #         latent_pi = actor.extract_features(obs_tensor)

        #         # action_net 通常是 Sequential(Linear, Tanh)
        #         # 取 tanh 之前的输出，看网络"真正想说什么"
        #         if isinstance(actor.action_net, torch.nn.Sequential):
        #             raw_mean = actor.action_net[0](latent_pi).cpu().numpy().flatten()
        #         else:
        #             raw_mean = actor.action_net(latent_pi).cpu().numpy().flatten()

        #         tanh_mean = np.tanh(raw_mean)

        #         # log_std 是网络的可学习参数
        #         if hasattr(actor, 'log_std'):
        #             log_std_val = actor.log_std.detach().cpu().numpy().flatten()
        #             raw_std = np.exp(log_std_val)
        #         else:
        #             raw_std = np.array([float('nan')])

        #     print(f"\n  [网络内部] tanh之前的原始均值: {raw_mean}")
        #     print(f"  [网络内部] tanh之后的均值:     {tanh_mean}  <-- deterministic看到的就是这个")
        #     print(f"  [网络内部] 原始标准差 std:      {raw_std}")
        #     if raw_std[0] > 1.0 or raw_std[1] > 1.0:
        #         print(f"  >>> std 远大于1.0，实锤'熵爆炸'！")
        #     if np.allclose(tanh_mean, 0, atol=0.05):
        #         print(f"  >>> 均值接近0，实锤'均值归零'！")

        # except Exception as e:
        #     print(f"\n  [方法1失败] {e}")

        # # ============ 方法2：同状态重复采样200次（任何版本都能跑） ============
        # print(f"\n  [经验采样] 对当前观测重复采样200次:")
        # sample_actions = []
        # for _ in range(200):
        #     a, _ = model.predict(obs, deterministic=False)
        #     sample_actions.append(a)
        # sample_actions = np.array(sample_actions)
        # det_action, _ = model.predict(obs, deterministic=True)

        # print(f"    deterministic输出:  {det_action}")
        # print(f"    采样均值:            {np.mean(sample_actions, axis=0)}")
        # print(f"    采样标准差:          {np.std(sample_actions, axis=0)}")
        # print(f"    采样最小值:          {np.min(sample_actions, axis=0)}")
        # print(f"    采样最大值:          {np.max(sample_actions, axis=0)}")

        # # 关键指标：有多少比例的采样结果紧贴动作边界？
        # high = env.action_space.high
        # low = env.action_space.low
        # at_limit_ratio = np.mean(
        #     (np.abs(sample_actions - high) < 0.01) |
        #     (np.abs(sample_actions - low) < 0.01),
        #     axis=0
        # )
        # print(f"    线速度触及±{high[0]}边界的比例: {at_limit_ratio[0]*100:.1f}%")
        # print(f"    角速度触及±{high[1]}边界的比例: {at_limit_ratio[1]*100:.1f}%")
        # if at_limit_ratio[0] > 0.3 or at_limit_ratio[1] > 0.3:
        #     print(f"    >>> 超过30%的采样紧贴边界，说明方差大到采样后大部分被裁剪！")

        # print("=" * 60 + "\n")

