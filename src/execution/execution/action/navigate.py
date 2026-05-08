import asyncio
import time
import math
import numpy as np
import torch
from typing import Dict, Optional

# 【注意】在实际项目中替换为正确的导入路径
from core.ros2.master import Ros2Runtime
from core.ros2.channels.bridges.robot import RobotBridge


class NavigateActioner:
    """基于强化学习模型的导航动作执行器（匹配原始状态空间）"""

    def __init__(self, runtime, model_path: str, target_x: float, target_y: float):
        self.runtime = runtime
        self.target_x = target_x
        self.target_y = target_y

        # 初始化RobotBridge
        self.robot_bridge = RobotBridge(runtime)
        self.robot_bridge.setup(
            laser_topic="/scan",
            imu_topic="/imu",
            odom_topic="/odom",
            cmd_vel_topic="/cmd_vel",
            goal_topic="/goal_pose"
        )
        self.runtime.register_node(self.robot_bridge)

        # 加载模型
        self.model = self._load_model(model_path)

        # 状态管理
        self._current_state = "idle"
        self._start_time = None
        self._elapsed_time = 0.0

        # 导航参数
        self._target_reached_threshold = 0.5  # 到达目标的距离阈值（米）
        self._max_steps = 1000  # 防止无限循环的最大步数

        # 保存上一步动作（用于状态空间）
        self._last_action = np.array([0.0, 0.0])

    def _load_model(self, model_path: str) -> callable:
        """加载模型（支持baseline或强化学习模型）"""
        if model_path == "baseline":
            return self._baseline_model  # 返回baseline函数
        try:
            model = torch.load(model_path)
            model.eval()  # 设置为评估模式
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def _baseline_model(self, obs: np.ndarray) -> tuple:
        """baseline策略：简单规则（朝目标方向移动）"""
        # 从obs中提取目标相对信息
        target_dist = obs[33]  # 原始距离（未归一化）
        target_angle = obs[32]  # 原始角度（未归一化）

        # 简单规则：朝目标方向移动
        lin_x = 0.5  # 固定线速度
        ang_z = target_angle * 0.5  # 角速度与角度偏差成正比
        return lin_x, ang_z

    def _get_obs(self) -> np.ndarray:
        """构建原始状态空间（38维）"""
        # 1. 激光雷达（24维，原始数据）
        laser_data = self.robot_bridge.get_laser_ranges()  # 原始激光数据
        
        # 2. IMU 加速度（3维，原始数据）
        imu_acc = self.robot_bridge.get_imu_data()  # 原始IMU数据
        acc_data = np.array(imu_acc["acc"])
        
        # 3. IMU 角速度（3维，原始数据）
        gyro_data = np.array(imu_acc["gyro"])
        
        # 4. IMU 姿态角 Roll/Pitch（2维，原始数据）
        rpy_data = np.array(imu_acc["rpy"])
        
        # 5. 目标相对偏航角（1维，原始数据）
        goal_relative = self.robot_bridge.get_goal_relative(self.target_x, self.target_y)
        angle_data = goal_relative["angle"]  # 原始角度
        
        # 6. 目标相对距离（1维，原始数据）
        dist_data = goal_relative["dist"]  # 原始距离
        
        # 7. 底盘当前速度（2维，原始数据）
        odom_data = self.robot_bridge.get_odom_data()
        vx = odom_data["vx"]
        vy = odom_data["vy"]
        vel_data = np.array([vx, vy])
        
        # 8. 上一步动作（2维，原始数据）
        last_action = self._last_action.copy()
        
        # 拼接所有数据（38维）
        obs = np.concatenate([
            laser_data,
            acc_data,
            gyro_data,
            rpy_data,
            np.array([angle_data]),
            np.array([dist_data]),
            vel_data,
            last_action
        ])
        return obs

    async def run_episode(self, deterministic=True, verbose=False) -> Dict[str, any]:
        """
        独立的推理/测试循环。
        将模型与导航执行解耦，适合快速验证。
        """
        # 重置状态
        self._current_state = "running"
        self._start_time = time.time()
        step = 0
        done = False

        # 获取初始观测
        obs = self._get_obs()
        ep_reward = 0.0
        final_dist = -1.0

        if verbose:
            print(f"\n[Ep Start] Goal:({self.target_x}, {self.target_y}) | Obs Shape:{obs.shape}")

        while not done:
            # 模型预测动作
            if self.model == self._baseline_model:
                # baseline模式：使用规则策略
                action = self.model(obs)
            else:
                # 强化学习模式：使用模型预测
                with torch.no_grad():
                    action_tensor = self.model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = action_tensor.squeeze().numpy()

            # 保存上一步动作（用于下一轮观测）
            self._last_action = action.copy()

            # 发送速度指令（直接使用模型输出，不归一化）
            lin_x, ang_z = action
            self.robot_bridge.send_velocity(lin_x, ang_z)

            # 更新观测
            obs = self._get_obs()

            # 计算奖励（简化：剩余距离的负值）
            goal_relative = self.robot_bridge.get_goal_relative(self.target_x, self.target_y)
            remaining_dist = goal_relative["dist"]
            reward = -remaining_dist  # 距离越近，奖励越高
            ep_reward += reward

            # 检查终止条件
            if remaining_dist < self._target_reached_threshold:
                done = True
                final_dist = remaining_dist
            elif step >= self._max_steps:
                done = True
                final_dist = remaining_dist

            if verbose:
                print(f"  Step {step}: Act=[{lin_x:.3f}, {ang_z:.3f}], Reward={reward:.3f}, Dist={remaining_dist:.3f}")

            step += 1
            await asyncio.sleep(0.1)  # 控制循环频率

        # 记录执行时间
        self._elapsed_time = time.time() - self._start_time
        self._current_state = "completed" if done else "failed"

        return {
            "reward": ep_reward,
            "success": final_dist < self._target_reached_threshold,
            "steps": step,
            "final_dist": final_dist,
            "elapsed_time": self._elapsed_time
        }

    def get_state(self) -> str:
        """获取当前状态"""
        return self._current_state

    def get_elapsed_time(self) -> float:
        """获取执行时间"""
        return self._elapsed_time if self._start_time else 0.0



# # 1. 初始化ROS2运行时
# runtime = Ros2Runtime()

# # 2. 创建导航执行器（加载模型或baseline）
# navigate_actioner = NavigateActioner(
#     runtime=runtime,
#     model_path="/path/to/model.pth",  # 或 "baseline"
#     target_x=2.0,
#     target_y=3.0
# )

# # 3. 运行测试循环
# result = asyncio.run(navigate_actioner.run_episode(verbose=True))
# print(result)  # 输出：{"reward": -12.34, "success": True, "steps": 50, ...}
