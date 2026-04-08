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

class RuleBasedPolicy(BasePolicy):
    """
    内部类：继承 BasePolicy 以满足 SB3 的类型检查，实际执行规则逻辑。
    """
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.config = config
        
        self.lin_vel_max = config["environment"].get("lin_vel_max", 0.22)
        self.ang_vel_max = config["environment"].get("ang_vel_max", 1.5)
        self.safety_distance = 0.35
        self.critical_distance = 0.22
        self.last_action = np.zeros(2, dtype=np.float32)

    def _predict(self, observation: np.ndarray, deterministic: bool = True):
        """SB3 标准预测接口"""
        # observation 可能是 batch (1, obs_dim) 或单条
        if observation.ndim == 1:
            obs = observation
        else:
            obs = observation[0]

        scan_len = len(obs) - 4 
        laser_scan = obs[:scan_len]
        goal_dist = obs[-4]
        goal_angle = math.atan2(obs[-3], obs[-2])

        front_count = max(1, int(len(laser_scan) * (30.0 / 180.0)))
        front_left = laser_scan[:front_count]
        front_right = laser_scan[-front_count:]
        front_lasers = np.concatenate([front_left, front_right])
        min_front_dist = np.min(front_lasers) if len(front_lasers) > 0 else 10.0

        linear_vel, angular_vel = 0.0, 0.0

        if min_front_dist < self.critical_distance:
            linear_vel = 0.0
            left_dist = np.mean(laser_scan[:len(laser_scan)//2]) if len(laser_scan)>0 else 0
            right_dist = np.mean(laser_scan[len(laser_scan)//2:]) if len(laser_scan)>0 else 0
            angular_vel = self.ang_vel_max if left_dist > right_dist else -self.ang_vel_max
        elif min_front_dist < self.safety_distance:
            linear_vel = self.lin_vel_max * 0.3
            left_min = np.min(laser_scan[:len(laser_scan)//2]) if len(laser_scan)>0 else 10
            right_min = np.min(laser_scan[len(laser_scan)//2:]) if len(laser_scan)>0 else 10
            repulsive = -1.0 if left_min < right_min else 1.0
            angular_vel = (2.0 * goal_angle) + repulsive * 2.0
        else:
            linear_vel = self.lin_vel_max * max(0.3, min(1.0, goal_dist * 2.0))
            angular_vel = 2.0 * goal_angle

        linear_vel = np.clip(linear_vel, 0.0, self.lin_vel_max)
        angular_vel = np.clip(angular_vel, -self.ang_vel_max, self.ang_vel_max)
        
        action = np.array([linear_vel, angular_vel], dtype=np.float32)
        action = 0.8 * action + 0.2 * self.last_action
        self.last_action = action

        # 返回格式必须是 (batch_size, action_dim)
        return action.reshape(1, -1)


class RuleBasedModel(BaseAlgorithm):
    """
    包装类：继承 SB3 的 BaseAlgorithm，让规则策略完美伪装成 SB3 模型。
    """
    def __init__(self, config, env, verbose=0, **kwargs):
        # 必须传 policy 为 None，我们在后面手动覆盖
        super().__init__(
            policy=None, 
            env=env, 
            learning_rate=0.0, 
            verbose=verbose, 
            _init_setup_model=False, 
            **kwargs
        )
        self.config = config
        # 手动初始化我们自定义的 Policy
        self.policy = RuleBasedPolicy(
            self.observation_space, 
            self.action_space, 
            self.config
        )

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100, tb_log_name: str = "run", reset_num_timesteps: bool = True, progress_bar: bool = False) -> RuleBasedModel:
        """规则策略不需要学习，直接跑完 timesteps"""
        print("[RuleBasedModel] Running baseline evaluation (No training)...")
        
        # 我们需要模拟训练循环，以便触发回调函数（比如保存模型、记录 Reward 等）
        obs = self.env.reset()
        for _ in range(total_timesteps):
            action, _ = self.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if done:
                obs = self.env.reset()
                
        return self

    def _setup_model(self) -> None:
        pass

    def set_env(self, env: GymEnv) -> None:
        super().set_env(env)
        self.policy = RuleBasedPolicy(self.observation_space, self.action_space, self.config)


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
