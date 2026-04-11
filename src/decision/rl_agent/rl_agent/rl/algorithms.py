from __future__ import annotations
import numpy as np
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

# 导入 SB3 相关模块
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

# class RuleBasedPolicy(BasePolicy):
#     """
#     内部类：继承 BasePolicy 以满足 SB3 的类型检查，实际执行规则逻辑。
#     """
#     def __init__(self, observation_space, action_space, config, **kwargs):
#         super().__init__(observation_space, action_space, **kwargs)
#         self.config = config
        
#         self.lin_vel_max = config["environment"].get("lin_vel_max", 0.22)
#         self.ang_vel_max = config["environment"].get("ang_vel_max", 1.5)
#         self.safety_distance = 0.35
#         self.critical_distance = 0.22
#         self.last_action = np.zeros(2, dtype=np.float32)

#     def _predict(self, observation: np.ndarray, deterministic: bool = True):
#         """SB3 标准预测接口"""
#         # if hasattr(observation, 'cpu'):
#         #     observation = observation.cpu().numpy()
            
#         # observation 可能是 batch (1, obs_dim) 或单条
#         if observation.ndim == 1:
#             obs = observation
#         else:
#             obs = observation[0]

#         scan_len = len(obs) - 4 
#         laser_scan = obs[:scan_len]
#         goal_dist = obs[-4]
#         goal_angle = math.atan2(obs[-3], obs[-2])

#         front_count = max(1, int(len(laser_scan) * (30.0 / 180.0)))
#         front_left = laser_scan[:front_count]
#         front_right = laser_scan[-front_count:]
#         front_lasers = np.concatenate([front_left, front_right])
#         min_front_dist = np.min(front_lasers) if len(front_lasers) > 0 else 10.0

#         linear_vel, angular_vel = 0.0, 0.0

#         if min_front_dist < self.critical_distance:
#             linear_vel = 0.0
#             left_dist = np.mean(laser_scan[:len(laser_scan)//2]) if len(laser_scan)>0 else 0
#             right_dist = np.mean(laser_scan[len(laser_scan)//2:]) if len(laser_scan)>0 else 0
#             angular_vel = self.ang_vel_max if left_dist > right_dist else -self.ang_vel_max
#         elif min_front_dist < self.safety_distance:
#             linear_vel = self.lin_vel_max * 0.3
#             left_min = np.min(laser_scan[:len(laser_scan)//2]) if len(laser_scan)>0 else 10
#             right_min = np.min(laser_scan[len(laser_scan)//2:]) if len(laser_scan)>0 else 10
#             repulsive = -1.0 if left_min < right_min else 1.0
#             angular_vel = (2.0 * goal_angle) + repulsive * 2.0
#         else:
#             linear_vel = self.lin_vel_max * max(0.3, min(1.0, goal_dist * 2.0))
#             angular_vel = 2.0 * goal_angle

#         linear_vel = np.clip(linear_vel, 0.0, self.lin_vel_max)
#         angular_vel = np.clip(angular_vel, -self.ang_vel_max, self.ang_vel_max)
        
#         action = np.array([linear_vel, angular_vel], dtype=np.float32)
#         action = 0.8 * action + 0.2 * self.last_action
#         self.last_action = action

#         # 返回格式必须是 (batch_size, action_dim)
#         return action.reshape(1, -1)


# class RuleBasedModel(BaseAlgorithm):
#     """
#     包装类：继承 SB3 的 BaseAlgorithm，让规则策略完美伪装成 SB3 模型。
#     """
#     def __init__(self, config, env, verbose=0, **kwargs):
#         # 必须传 policy 为 None，我们在后面手动覆盖
#         super().__init__(
#             policy=None, 
#             env=env, 
#             learning_rate=0.0, 
#             verbose=verbose, 
#             # _init_setup_model=False, 
#             **kwargs
#         )
#         self.config = config
#         # 手动初始化我们自定义的 Policy
#         self.policy = RuleBasedPolicy(
#             self.observation_space, 
#             self.action_space, 
#             self.config
#         )

#     def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100, tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False) -> RuleBasedModel:
#         """规则策略不需要学习，直接跑完 timesteps"""
#         print("[RuleBasedModel] Running baseline evaluation (No training)...")
        
#         # 我们需要模拟训练循环，以便触发回调函数（比如保存模型、记录 Reward 等）
#         obs = self.env.reset()
#         for _ in range(total_timesteps):
#             action, _ = self.predict(obs, deterministic=True)
#             obs, reward, done, info = self.env.step(action)
#             if done:
#                 obs = self.env.reset()
                
#         return self

#     def _setup_model(self) -> None:
#         pass

#     def set_env(self, env: GymEnv) -> None:
#         super().set_env(env)
#         self.policy = RuleBasedPolicy(self.observation_space, self.action_space, self.config)


def get_algorithm(algo_name: str, env: GymEnv, config: dict, **kwargs) -> BaseAlgorithm:
    """
    算法工厂函数：根据配置字符串返回统一的算法实例。
    
    Args:
        algo_name (str): "ppo", "sac", "rule_baseline"
        env (GymEnv): 实例化好的环境
        config (dict): 全局配置字典
        **kwargs: 传递给 SB3 算法的额外参数 (如 learning_rate, batch_size)
    
    Returns:
        BaseAlgorithm: SB3 的 PPO/SAC 实例，或 RuleBasedModel 实例
    """
    algo_name = algo_name.lower()
    
    if algo_name == "ppo":
        # 从 config 中提取 PPO 专属参数，如果没设置则使用 SB3 默认值
        ppo_params = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": config.get("ppo", {}).get("learning_rate", 3e-4),
            "n_steps": config.get("ppo", {}).get("n_steps", 2048),
            "batch_size": config.get("ppo", {}).get("batch_size", 64),
            "verbose": 1,
            **kwargs
        }
        print("Initializing PPO Algorithm...")
        return PPO(**ppo_params)
        
    elif algo_name == "sac":
        sac_params = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": config.get("sac", {}).get("learning_rate", 3e-4),
            "batch_size": config.get("sac", {}).get("batch_size", 256),
            "buffer_size": config.get("sac", {}).get("buffer_size", 1_000_000),
            "verbose": 1,
            **kwargs
        }
        print("Initializing SAC Algorithm...")
        return SAC(**sac_params)
        
    elif algo_name == "rule_baseline":
        print("Initializing Rule-Based Baseline...")
        return RuleBasedModel(config=config, env=env, verbose=1)
        
    else:
        raise ValueError(f"Unknown algorithm name: {algo_name}. Choose from ['ppo', 'sac', 'rule_baseline'].")


class RuleBasedPolicy:
    """
    规避 SB3 BasePolicy 的类型转换地狱，纯净的 numpy 实现。
    """
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        
        self.lin_vel_max = config["environment"].get("lin_vel_max", 0.22)
        self.ang_vel_max = config["environment"].get("ang_vel_max", 1.5)
        self.laser_range_max = config["environment"].get("laser_range_max", 3.5)
        self.laser_beams_num = config["environment"].get("laser_beams_num", 24)
        
        # 原始物理距离阈值
        self.safety_distance = 0.35
        self.critical_distance = 0.22
        
        self.last_action = np.zeros(2, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        """入口，返回 numpy (1, 2)"""
        action = self._predict(observation, deterministic)
        return action.reshape(1, -1)

    def _predict(self, observation, deterministic=True):
        if observation.ndim == 1:
            obs = observation
        else:
            obs = observation[0]

        # ==========================================
        # 【关键修复】：按 _get_obs() 的拼接顺序严格索引
        # ==========================================
        # obs 布局:
        # [0 : 24]           = laser_norm (归一化, 0~1)
        # [24 : 27]          = imu_acc
        # [27 : 30]          = imu_gyro
        # [30 : 32]          = imu_rp
        # [32]               = goal_angle (归一化, -1~1)
        # [33]               = goal_dist  (归一化, 0~1)
        # [34 : 36]          = odom_vel   (归一化)
        # [36]               = last_act_norm
        
        n = self.laser_beams_num
        laser_norm = obs[:n]
        
        # 反归一化，还原为真实物理距离（米），后续阈值才能直接比较
        laser_scan = laser_norm * self.laser_range_max
        
        goal_angle_norm = obs[n + 8]   # = obs[32], 值域 [-1, 1]
        goal_dist_norm = obs[n + 9]    # = obs[33], 值域 [0, 1]
        
        # 还原真实角度和距离
        goal_angle = goal_angle_norm * math.pi   # [-π, π]
        goal_dist = goal_dist_norm * self.config["environment"].get("dist_to_goal_clip_norm", 5.0)  # [0, 5] 米

        # --- 前方扇区检测 ---
        n_beams = len(laser_scan)
        # 24根激光覆盖360°，每根15°，前方±30° = 前后各2根
        front_count = max(1, int(n_beams * (30.0 / 180.0)))  # = 4
        front_left = laser_scan[:front_count]
        front_right = laser_scan[-front_count:]
        front_lasers = np.concatenate([front_left, front_right])
        min_front_dist = float(np.min(front_lasers))

        # --- 左右半区检测 ---
        left_half = laser_scan[:n_beams // 2]
        right_half = laser_scan[n_beams // 2:]

        linear_vel, angular_vel = 0.0, 0.0

        # 1. 紧急避障：前方 < 0.22m
        if min_front_dist < self.critical_distance:
            linear_vel = 0.0
            left_clear = float(np.min(left_half))
            right_clear = float(np.min(right_half))
            angular_vel = self.ang_vel_max * (1.0 if left_clear > right_clear else -1.0)

        # 2. 安全警戒：前方 < 0.35m
        elif min_front_dist < self.safety_distance:
            linear_vel = self.lin_vel_max * 0.3
            left_min = float(np.min(left_half))
            right_min = float(np.min(right_half))
            repulsive = 1.0 if left_min < right_min else -1.0
            angular_vel = (2.0 * goal_angle) + repulsive * 2.0

        # 3. 安全区域：正常导航
        else:
            linear_vel = self.lin_vel_max * max(0.3, min(1.0, goal_dist * 2.0))
            angular_vel = 2.0 * goal_angle

        # --- 裁剪 ---
        linear_vel = np.clip(linear_vel, 0.0, self.lin_vel_max)
        angular_vel = np.clip(angular_vel, -self.ang_vel_max, self.ang_vel_max)
        
        # --- 平滑 ---
        action = np.array([linear_vel, angular_vel], dtype=np.float32)
        action = 0.8 * action + 0.2 * self.last_action
        self.last_action = action

        return action



class RuleBasedModel:
    """
    鸭子类型：只要会 predict() / learn() / save()，就够用了。
    """
    def __init__(self, config, env, verbose=0, **kwargs):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.config = config
        self.verbose = verbose
        
        self.policy = RuleBasedPolicy(
            self.observation_space, 
            self.action_space, 
            self.config
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return self.policy.predict(observation, deterministic), None

    def learn(self, total_timesteps=0, callback=None, log_interval=100, 
              tb_log_name="run", reset_num_timesteps=True, progress_bar=False):
        print("[RuleBasedModel] Running baseline evaluation (No training)...")
        obs, _ = self.env.reset()
        for _ in range(total_timesteps):
            action, _ = self.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            if done:
                obs, _ = self.env.reset()
        return self

    def save(self, path, exclude=None, include=None):
        if self.verbose > 0:
            print(f"[RuleBasedModel] No weights to save. Skipping {path}.")
