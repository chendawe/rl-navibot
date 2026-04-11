import sys
import os
import time
import math
import numpy as np
import yaml
import rclpy
from pathlib import Path

# ========== 1. 你的路径与项目导入 ==========
sys.path.append("/home/chendawww/workspace/rl-navibot/src")
from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf

# ========== 2. 加载配置文件 ==========
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config" / "rl.config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
# # 重定向所有输出到文件
# log_file = open(BASE_DIR / "saved_models" / "log" / "training_output.log", "w")
# sys.stdout = log_file
# sys.stderr = log_file

# ========== 3. 极其关键：初始化 ROS2 ==========
# 你的环境底层用了 rclpy，必须在实例化环境前初始化！
rclpy.init()
print("ROS2 initialized...")

# ========== 4. 实例化环境 (原生的，不要用 DummyVecEnv) ==========
my_robot_urdf = fetch_tb3_urdf()
env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, config=config)

# ========== 5. 导入 SAC 和 Callback ==========
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


CONTAINER_NAME = "ros2my"  # 换成你 docker ps 里看到的真实名字
class GazeboShutdownCallback(BaseCallback):
    """
    训练结束后关闭 Docker 内的 Gazebo。
    stop_container=True: 整个容器停掉
    stop_container=False: 只杀 gzserver 进程
    """
    def __init__(self, container_name=CONTAINER_NAME, stop_container=False, verbose=0):
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

        
# ========== 6. 自定义 Callback：记录成功率 ==========
class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        # 用字典存储每个环境(按索引)的独立状态
        self._current_rewards = {}
        self._current_successes = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        # 遍历所有环境 (单环境时 len 就是 1，多环境时就是 N)
        for i in range(len(dones)):
            # 累积当前环境的 reward
            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + rewards[i]

            # 检查当前环境是否到达目标
            if infos[i].get('goal_reached', False):
                self._current_successes[i] = True

            # 如果当前环境这一个 Episode 结束了
            if dones[i]:
                # 记录到总历史列表中
                self.episode_rewards.append(self._current_rewards[i])
                self.episode_successes.append(float(self._current_successes.get(i, False)))

                # 从字典中删掉该环境的状态，为它下一个 episode 做准备
                del self._current_rewards[i]
                self._current_successes.pop(i, None)

        # 打印日志 (每凑齐 10 个 episode 打印一次)
        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
            recent = self.episode_successes[-10:]
            avg_r = np.mean(self.episode_rewards[-10:])
            rate = np.mean(recent) * 100
            print(f"[Total Episodes: {len(self.episode_rewards)}] "
                  f"Success Rate: {rate:.0f}% | Avg Reward: {avg_r:.1f}")

        return True


# ========== 7. 配置 SAC 算法 ==========
# 注意：SAC 的 batch_size 通常比 PPO 大很多，256 是标配，4060 跑起来毫无压力
model = SAC(
    "MlpPolicy",
    env,
    device="cuda",
    learning_rate=3e-4,
    buffer_size=100000,     # 经验回放池大小 (10万步足够初期学习，内存不够可以改小)
    learning_starts=1000,   # 前 1000 步纯随机探索，不更新网络 (攒点初始数据)
    batch_size=512,         # SAC 标配 256
    tau=0.005,              # 软更新系数
    gamma=0.99,             # 折扣因子
    ent_coef='auto',        # 极其关键！SAC 的灵魂，自动调节探索熵，防止过早收敛
    # 网络结构：Actor 和 Critic 都用两层 256 的全连接
    policy_kwargs=dict(net_arch=[256, 256]), 
    verbose=1,
    tensorboard_log=BASE_DIR / "saved_models" / "SAC" / "log",
)

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
#     tensorboard_log="./tb_logs/",
# )

# ========== 8. 开始训练 ==========
print("=" * 50)
print("SAC Training started!")
print("Monitor: tensorboard --logdir=./tb_logs_sac/")
print("=" * 50)

try:
    # ========== 1. 断点保存 Callback ==========
    checkpoint_callback = CheckpointCallback(
        save_freq=4000,
        save_path=BASE_DIR / "saved_models" / "SAC",
        name_prefix="sac_nav_model",
        save_replay_buffer=True,  # 自动保存经验池
    )

    model.learn(
        # total_timesteps=1,
        total_timesteps=200_000,  # SAC 收敛比 PPO 慢一点，先跑 20 万步看趋势
        callback=[
            checkpoint_callback,
            SuccessRateCallback(),
            GazeboShutdownCallback(container_name=CONTAINER_NAME, stop_container=False),
        ],
        progress_bar=True,
    )

finally:
    # 无论成功还是报错，务必清理资源
    model.save(BASE_DIR / "saved_models" / "SAC" / "sac_nav_model")
    print("Model saved!")
    env.close()
    rclpy.shutdown()
    print("ROS2 shutdown. Done!")
    os.system(f"docker exec gazebo_sim pkill -9 gzserver 2>/dev/null")
    print(f"Gazebo sim in {CONTAINER_NAME} container is terminated")
