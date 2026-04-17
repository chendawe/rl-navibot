import math
import random
import logging
from typing import Dict, Any, Optional
# 明确依赖你的核心实体
from core.ros2.channels.bridges.robot import RobotBridge 

logger = logging.getLogger(__name__)

class RobotService:
    def __init__(self, robot_bridge: Optional[RobotBridge] = None):
        self.robot_bridge = robot_bridge
        self._mock_yaw = 0.0

    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据，底层断开时自动降级为雷达 Mock"""
        if self.robot_bridge:
            try:
                # 调用你 Bridge 里写好的原生 API
                laser_ranges = self.robot_bridge.get_laser_ranges()
                ranges_list = list(laser_ranges[::10]) # 降采样
                return {
                    "laser_ranges": ranges_list, 
                    "min_laser": float(min(ranges_list))
                }
            except Exception as e:
                logger.warning(f"[RobotService] 获取激光失败，启用 Mock: {e}")

        # 兜底 Mock
        mock_ranges = [random.uniform(1.0, 3.5) for _ in range(36)]
        mock_ranges[0] = random.uniform(0.2, 0.5)
        return {"laser_ranges": mock_ranges, "min_laser": min(mock_ranges)}

    def get_odom_data(self) -> Dict[str, float]:
        """获取里程计，底层断开时自动降级为打转 Mock"""
        if self.robot_bridge:
            try:
                return self.robot_bridge.get_odom_data()
            except Exception as e:
                logger.warning(f"[RobotService] 获取里程计失败，启用 Mock: {e}")

        # 兜底 Mock
        self._mock_yaw += 0.05
        return {"x": 0.0, "y": 0.0, "yaw": self._mock_yaw, "vx": 0.0, "wz": 0.0}
