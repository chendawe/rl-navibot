# app/harness/graph/dual_circle/ros2/navi_executor.py
import asyncio
import math
from typing import Dict, Any
from pydantic import BaseModel

from core.ros2.channels.bridges.robot import RobotBridge
from core.ros2.runtime import Ros2Runtime
from core.ros2.simulators.gazebo import GazeboSimulator

class NaviInput(BaseModel):
    target_position: str  # 如 "A点"

class Ros2NaviExecutor:
    def __init__(self):
        self.runtime = Ros2Runtime()  # 单例，自动初始化 ROS2
        self.robot_bridge = RobotBridge(self.runtime)
        self.gazebo_sim = GazeboSimulator(self.runtime)
        
        # 假设你已经配置好了这些服务
        self.gazebo_sim.setup(
            reset_world="/reset_world",
            set_entity="/gazebo/set_entity_state",
            delete_entity="/delete_entity",
            spawn_entity="/spawn_entity",
            entity_name="robot",
            reset_mode="teleport"
        )
        self.robot_bridge.setup(
            laser_topic="/scan",
            imu_topic="/imu",
            odom_topic="/odom",
            cmd_vel_topic="/cmd_vel",
            goal_topic="/goal_pose"
        )

    async def execute(self, input_data: NaviInput) -> Dict[str, Any]:
        target = input_data.target_position
        
        # 1. 字符串转坐标（你需要实现这个映射）
        x, y, yaw = self._resolve_position(target)
        
        # 2. 直接瞬移到目标位置（高性能模式）
        self.gazebo_sim.set_robot_pose(x, y, yaw)
        
        # 3. 等待机器人稳定（可选，根据需要）
        await asyncio.sleep(0.5)
        
        # 4. 检查是否到达（简单距离判断）
        odom = self.robot_bridge.get_odom_data()
        dist = math.hypot(x - odom["x"], y - odom["y"])
        
        if dist < 0.1:  # 10cm 误差内算到达
            return {"status": "success", "actual_position": (odom["x"], odom["y"])}
        else:
            return {"status": "timeout", "reason": f"距离目标还有 {dist:.2f}m"}

    def _resolve_position(self, name: str) -> tuple:
        # 你需要实现位置映射表
        position_map = {
            "A点": (1.0, 2.0, 0.0),
            "B点": (3.0, 4.0, math.pi/2),
            "C点": (5.0, 6.0, math.pi)
        }
        return position_map.get(name, (0.0, 0.0, 0.0))
