# # import sys
# # import os
# # import time
# # import math
# # import numpy as np
# # import yaml
# # import rclpy
# # from pathlib import Path

# # # ========== 1. 你的路径与项目导入 ==========
# # sys.path.append("/home/chendawww/workspace/rl-navibot/src")
# # from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf

# # # ========== 2. 加载配置文件 ==========
# # BASE_DIR = Path(__file__).parent.parent
# # CONFIG_PATH = BASE_DIR / "config" / "rl.config.yaml"
# # with open(CONFIG_PATH, "r") as f:
# #     config = yaml.safe_load(f)
    
# # # ========== 3. 极其关键：初始化 ROS2 ==========
# # rclpy.init()
# # print("ROS2 initialized...")

# # # ========== 4. 实例化环境 (原生的，不要用 DummyVecEnv) ==========
# # my_robot_urdf = fetch_tb3_urdf()
# # env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config)

# # # ========== 5. 导入 SAC 和 Callback ==========
# # from stable_baselines3 import SAC
# # from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# # # ========== 6. 自定义 Callback：记录成功率 + 打印动作分布 ==========
# # class SuccessRateCallback(BaseCallback):
# #     def __init__(self, verbose=0):
# #         super().__init__(verbose)
# #         self.episode_rewards = []
# #         self.episode_successes = []
# #         self.action_means = []  # 存储动作均值
# #         self.action_stds = []   # 存储动作标准差
# #         self._current_rewards = {}
# #         self._current_successes = {}

# #     def _on_step(self) -> bool:
# #         infos = self.locals.get("infos", [])
# #         dones = self.locals.get("dones", [])
# #         rewards = self.locals.get("rewards", [])
# #         actions = self.locals.get("actions", [])  # 获取当前批次的动作

# #         # 计算当前批次的动作均值和标准差
# #         if len(actions) > 0:
# #             action_mean = np.mean(actions, axis=0)
# #             action_std = np.std(actions, axis=0)
# #             self.action_means.append(action_mean)
# #             self.action_stds.append(action_std)
            
# #             # 每100步打印一次动作分布
# #             if len(self.action_means) % 100 == 0:
# #                 print(f"\n[Action Distribution] Mean: {action_mean}, Std: {action_std}")
# #                 print(f"[Recent Episodes: {len(self.episode_rewards)}] "
# #                       f"Success Rate: {np.mean(self.episode_successes[-10:])*100:.0f}% | "
# #                       f"Avg Reward: {np.mean(self.episode_rewards[-10:]):.1f}")

# #         # 遍历所有环境
# #         for i in range(len(dones)):
# #             # 累积当前环境的 reward
# #             self._current_rewards[i] = self._current_rewards.get(i, 0.0) + rewards[i]

# #             # 检查当前环境是否到达目标
# #             if infos[i].get('goal_reached', False):
# #                 self._current_successes[i] = True

# #             # 如果当前环境这一个 Episode 结束了
# #             if dones[i]:
# #                 # 记录到总历史列表中
# #                 self.episode_rewards.append(self._current_rewards[i])
# #                 self.episode_successes.append(float(self._current_successes.get(i, False)))

# #                 # 从字典中删掉该环境的状态
# #                 del self._current_rewards[i]
# #                 self._current_successes.pop(i, None)

# #         return True

# #     def _on_training_end(self) -> None:
# #         # 训练结束时打印最终动作分布
# #         if len(self.action_means) > 0:
# #             final_mean = np.mean(self.action_means, axis=0)
# #             final_std = np.mean(self.action_stds, axis=0)
# #             print(f"\n[Final Action Distribution] Mean: {final_mean}, Std: {final_std}")
# #             print(f"[Total Episodes: {len(self.episode_rewards)}] "
# #                   f"Success Rate: {np.mean(self.episode_successes)*100:.0f}% | "
# #                   f"Avg Reward: {np.mean(self.episode_rewards):.1f}")


# # # ========== 7. 配置 SAC 算法 ==========
# # # 注意：SAC 的 batch_size 通常比 PPO 大很多，256 是标配，4060 跑起来毫无压力
# # model = SAC(
# #     "MlpPolicy",
# #     env,
# #     device="cuda",
# #     learning_rate=3e-4,
# #     buffer_size=100000,
# #     learning_starts=1000,
# #     batch_size=512,
# #     tau=0.005,
# #     gamma=0.99,
# #     ent_coef='auto',
# #     policy_kwargs=dict(net_arch=[256, 256]),
# #     verbose=1,
# #     tensorboard_log=BASE_DIR / "saved_models" / "SAC" / "log",
# # )

# # # ========== 8. 加载checkpoint并继续训练 ==========
# # print("=" * 50)
# # print("Loading checkpoint and continuing training...")
# # print("=" * 50)

# # # 加载最新的checkpoint（假设是148000步）
# # CHECKPOINT_PATH = BASE_DIR / "saved_models" / "SAC" / "sac_nav_model_148000_steps"
# # model.load(CHECKPOINT_PATH)
# # print(f"Model loaded from: {CHECKPOINT_PATH}")

# # # 继续训练的步数（如50000步）
# # CONTINUE_TIMESTEPS = 50000

# # # ========== 9. 开始继续训练 ==========
# # try:
# #     # 断点保存 Callback
# #     checkpoint_callback = CheckpointCallback(
# #         save_freq=4000,
# #         save_path=BASE_DIR / "saved_models" / "SAC",
# #         name_prefix="sac_nav_model",
# #         save_replay_buffer=True,
# #     )

# #     model.learn(
# #         total_timesteps=CONTINUE_TIMESTEPS,
# #         callback=[
# #             checkpoint_callback,
# #             SuccessRateCallback(),
# #         ],
# #         progress_bar=True,
# #     )

# # finally:
# #     # 保存最终模型
# #     model.save(BASE_DIR / "saved_models" / "SAC" / "sac_nav_model_final")
# #     print("Final model saved!")
# #     env.close()
# #     rclpy.shutdown()
# #     print("ROS2 shutdown. Done!")
# #     os.system(f"docker exec gazebo_sim pkill -9 gzserver 2>/dev/null")
# #     print(f"Gazebo sim terminated")


# import sys
# import os
# import time
# import math
# import numpy as np
# import yaml
# import rclpy
# from pathlib import Path

# # ========== 1. 你的路径与项目导入 ==========
# sys.path.append("/home/chendawww/workspace/rl-navibot/src")
# from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf

# # ========== 2. 加载配置文件 ==========
# BASE_DIR = Path(__file__).parent.parent
# CONFIG_PATH = BASE_DIR / "config" / "rl.config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)
    
# # # 重定向所有输出到文件
# # log_file = open(BASE_DIR / "saved_models" / "log" / "training_output.log", "w")
# # sys.stdout = log_file
# # sys.stderr = log_file

# # ========== 3. 极其关键：初始化 ROS2 ==========
# # 你的环境底层用了 rclpy，必须在实例化环境前初始化！
# rclpy.init()
# print("ROS2 initialized...")

# # ========== 4. 实例化环境 (原生的，不要用 DummyVecEnv) ==========
# my_robot_urdf = fetch_tb3_urdf()
# env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config)

# # ========== 5. 导入 SAC 和 Callback ==========
# from stable_baselines3 import SAC, PPO
# from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


# CONTAINER_NAME = "ros2my"  # 换成你 docker ps 里看到的真实名字
# class GazeboShutdownCallback(BaseCallback):
#     """
#     训练结束后关闭 Docker 内的 Gazebo。
#     stop_container=True: 整个容器停掉
#     stop_container=False: 只杀 gzserver 进程
#     """
#     def __init__(self, container_name=CONTAINER_NAME, stop_container=False, verbose=0):
#         super().__init__(verbose)
#         self.container_name = container_name
#         self.stop_container = stop_container
        
#     def _on_step(self) -> bool:
#         return True
    
#     def _on_training_end(self) -> None:
#         if self.stop_container:
#             print(f"\n[Shutdown] 停止容器: {self.container_name}")
#             os.system(f"docker stop {self.container_name}")
#         else:
#             print(f"\n[Shutdown] 杀掉容器内 gzserver")
#             os.system(f"docker exec {self.container_name} pkill -9 gzserver 2>/dev/null")
#         print("[Shutdown] 完成。")

        
# # ========== 6. 自定义 Callback：记录成功率 ==========
# class SuccessRateCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_rewards = []
#         self.episode_successes = []
#         # 用字典存储每个环境(按索引)的独立状态
#         self._current_rewards = {}
#         self._current_successes = {}

#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos", [])
#         dones = self.locals.get("dones", [])
#         rewards = self.locals.get("rewards", [])

#         # 遍历所有环境 (单环境时 len 就是 1，多环境时就是 N)
#         for i in range(len(dones)):
#             # 累积当前环境的 reward
#             self._current_rewards[i] = self._current_rewards.get(i, 0.0) + rewards[i]

#             # 检查当前环境是否到达目标
#             if infos[i].get('goal_reached', False):
#                 self._current_successes[i] = True

#             # 如果当前环境这一个 Episode 结束了
#             if dones[i]:
#                 # 记录到总历史列表中
#                 self.episode_rewards.append(self._current_rewards[i])
#                 self.episode_successes.append(float(self._current_successes.get(i, False)))

#                 # 从字典中删掉该环境的状态，为它下一个 episode 做准备
#                 del self._current_rewards[i]
#                 self._current_successes.pop(i, None)

#         # 打印日志 (每凑齐 10 个 episode 打印一次)
#         if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
#             recent = self.episode_successes[-10:]
#             avg_r = np.mean(self.episode_rewards[-10:])
#             rate = np.mean(recent) * 100
#             print(f"[Total Episodes: {len(self.episode_rewards)}] "
#                   f"Success Rate: {rate:.0f}% | Avg Reward: {avg_r:.1f}")

#         return True


# # # ========== 7. 配置 SAC 算法 ==========
# # # 注意：SAC 的 batch_size 通常比 PPO 大很多，256 是标配，4060 跑起来毫无压力
# # model = SAC(
# #     "MlpPolicy",
# #     env,
# #     device="cuda",
# #     learning_rate=3e-4,
# #     buffer_size=100000,     # 经验回放池大小 (10万步足够初期学习，内存不够可以改小)
# #     learning_starts=1000,   # 前 1000 步纯随机探索，不更新网络 (攒点初始数据)
# #     batch_size=512,         # SAC 标配 256
# #     tau=0.005,              # 软更新系数
# #     gamma=0.99,             # 折扣因子
# #     ent_coef='auto',        # 极其关键！SAC 的灵魂，自动调节探索熵，防止过早收敛
# #     # 网络结构：Actor 和 Critic 都用两层 256 的全连接
# #     policy_kwargs=dict(net_arch=[256, 256]), 
# #     verbose=1,
# #     tensorboard_log=BASE_DIR / "saved_models" / "SAC" / "log",
# # )

# model = PPO(
#     "MlpPolicy",
#     env,
#     device="cuda",
#     learning_rate=3e-4,
#     n_steps=2048,          # 每次更新采集 2048 步
#     batch_size=64,         # 4060 8G 完全够用
#     n_epochs=10,           # 每批数据训练 10 轮
#     gamma=0.99,            # 折扣因子
#     gae_lambda=0.95,       # GAE 参数
#     clip_range=0.2,        # PPO 裁剪范围
#     ent_coef=0.01,         # 熵正则化，鼓励探索（导航任务很重要！）
#     vf_coef=0.5,           # value function 权重
#     max_grad_norm=0.5,     # 梯度裁剪
#     policy_kwargs={
#         "net_arch": [dict(pi=[256, 256], vf=[256, 256])],  # 4060 跑得动
#     },
#     verbose=1,
#     tensorboard_log=BASE_DIR / "saved_models" / "PPO" / "log",
# )

# # ========== 8. 开始训练 ==========
# print("=" * 50)
# print("SAC Training started!")
# print("Monitor: tensorboard --logdir=./tb_logs_sac/")
# print("=" * 50)

# try:
#     # ========== 1. 断点保存 Callback ==========
#     checkpoint_callback = CheckpointCallback(
#         save_freq=4000,
#         save_path=BASE_DIR / "saved_models" / "PPO",
#         name_prefix="ppo_nav_model",
#         save_replay_buffer=True,  # 自动保存经验池
#     )

#     model.learn(
#         # total_timesteps=1,
#         total_timesteps=200_000,  # SAC 收敛比 PPO 慢一点，先跑 20 万步看趋势
#         callback=[
#             checkpoint_callback,
#             SuccessRateCallback(),
#             GazeboShutdownCallback(container_name=CONTAINER_NAME, stop_container=False),
#         ],
#         progress_bar=True,
#     )

# finally:
#     # 无论成功还是报错，务必清理资源
#     model.save(BASE_DIR / "saved_models" / "PPO" / "ppo_nav_model")
#     print("Model saved!")
#     env.close()
#     rclpy.shutdown()
#     print("ROS2 shutdown. Done!")
#     os.system(f"docker exec gazebo_sim pkill -9 gzserver 2>/dev/null")
#     print(f"Gazebo sim in {CONTAINER_NAME} container is terminated")








import sys
import os
import time
import math
import numpy as np
import yaml
import rclpy
from pathlib import Path

# ========== 1. 路径与项目导入 ==========
sys.path.append("/home/chendawww/workspace/rl-navibot/src")
from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf

# ========== 2. 加载配置文件 ==========
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config" / "rl.config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ========== 3. 初始化 ROS2 ==========
rclpy.init()
print("ROS2 initialized...")

# ========== 4. 实例化环境 ==========
my_robot_urdf = fetch_tb3_urdf()
env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config)

# ========== 5. 导入算法与Callback ==========
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ========== 6. 自定义Callback（保持不变）==========
class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        self._current_rewards = {}
        self._current_successes = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i in range(len(dones)):
            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + rewards[i]
            if infos[i].get('goal_reached', False):
                self._current_successes[i] = True

            if dones[i]:
                self.episode_rewards.append(self._current_rewards[i])
                self.episode_successes.append(float(self._current_successes.get(i, False)))
                del self._current_rewards[i]
                self._current_successes.pop(i, None)

        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
            recent = self.episode_successes[-10:]
            avg_r = np.mean(self.episode_rewards[-10:])
            rate = np.mean(recent) * 100
            print(f"[Total Episodes: {len(self.episode_rewards)}] "
                  f"Success Rate: {rate:.0f}% | Avg Reward: {avg_r:.1f}")

        return True

class GazeboShutdownCallback(BaseCallback):
    def __init__(self, container_name="ros2my", stop_container=False, verbose=0):
        super().__init__(verbose)
        self.container_name = container_name
        self.stop_container = stop_container

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.stop_container:
            print(f"\n[Shutdown] 停止容器: {self.container_name}")
            os.system(f"docker stop {self.container_name}")
        else:
            print(f"\n[Shutdown] 杀掉容器内 gzserver")
            os.system(f"docker exec {self.container_name} pkill -9 gzserver 2>/dev/null")
        print("[Shutdown] 完成。")

MODEL_DIR = BASE_DIR / "saved_models" / "PPO"
MODEL_NAME_PREFIX = "ppo_nav_model"
MODEL_NAME_FINAL = "ppo_nav_model" + "_final"

# ========== 7. 配置PPO算法（保持不变）==========
model = PPO(
    "MlpPolicy",
    env,
    device="cuda",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    verbose=1,
    tensorboard_log=MODEL_DIR / "log",
)


# ========== 8. 断点续训逻辑（核心修改）==========
print("=" * 50)
print("Checking for checkpoint...")
print("=" * 50)

import re

checkpoint_files = [f for f in MODEL_DIR.glob(f"{MODEL_NAME_PREFIX}_*.zip") if "final" not in f.stem]
if checkpoint_files:
    # 使用正则表达式提取文件名中的数字（步数）
    def extract_steps(filename):
        match = re.search(r"(\d+)", filename.stem)  # 匹配文件名中的数字
        return int(match.group(1)) if match else 0  # 如果找到数字则返回，否则返回0

    # 找到步数最大的checkpoint（最新的训练进度）
    latest_checkpoint = max(checkpoint_files, key=extract_steps)
    trained_steps = extract_steps(latest_checkpoint)  # 已训练步数
    TOTAL_TIMESTEPS = 200_000  # 总目标步数
    CONTINUE_TIMESTEPS = TOTAL_TIMESTEPS - trained_steps  # 剩余步数
    
    model.load(latest_checkpoint)
    
    print(f"Loaded checkpoint: {latest_checkpoint}")
    print(f"Already trained: {trained_steps} steps, continuing for {CONTINUE_TIMESTEPS} steps.")
else:
    print("No checkpoint found, starting new training.")
    trained_steps = 0
    CONTINUE_TIMESTEPS = 200_000  # 总步数（从头开始）

# ========== 9. 开始训练（保持不变）==========
try:
    checkpoint_callback = CheckpointCallback(
        save_freq=4000,
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME_PREFIX,
        save_replay_buffer=True,
    )

    model.learn(
        total_timesteps=CONTINUE_TIMESTEPS,
        callback=[
            checkpoint_callback,
            SuccessRateCallback(),
            GazeboShutdownCallback(container_name="ros2my", stop_container=False),
        ],
        progress_bar=True,
    )

finally:
    model.save(MODEL_DIR / MODEL_NAME_FINAL)
    print("Final model saved!")
    env.close()
    rclpy.shutdown()
    print("ROS2 shutdown. Done!")
    os.system(f"docker exec ros2my pkill -9 gzserver 2>/dev/null")
    print("Gazebo sim terminated.")
