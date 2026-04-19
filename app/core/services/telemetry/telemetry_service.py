import math
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TelemetryService:
    def __init__(self, robot_service, rl_service, rgb_streamer=None, depth_streamer=None, map_streamer=None):
        self.robot_service = robot_service
        self.rl_service = rl_service
        self.rgb_streamer = rgb_streamer
        self.depth_streamer = depth_streamer
        self.map_streamer = map_streamer
        self._last_time = time.time()
        self._fps = -1

    # def get_telemetry(self) -> Dict[str, Any]:
    #     now = time.time()
    #     delta = now - self._last_time if self._last_time else 0.1
    #     self._last_time = now
    #     self._fps = 0.9 * self._fps + 0.1 * (1.0 / delta) if delta > 0 else 10.0

    #     robot_data = self.robot_service.get_odom_data() or {}
    #     sensor_data = self.robot_service.get_sensor_data() or {}
    #     rl_data = self.rl_service.get_state() or {}

    #     # 安全获取值，若缺失则用 None
    #     robot_x = robot_data.get("x")
    #     robot_y = robot_data.get("y")
    #     robot_yaw = robot_data.get("yaw")
    #     min_laser = sensor_data.get("min_laser")

    #     return {
    #         "perf": {"step_fps": self._fps},
    #         "rl": rl_data,
    #         "robot": {
    #             "x": robot_x,
    #             "y": robot_y,
    #             "yaw": robot_yaw,
    #         },
    #         "task": {
    #             "goal_x": 2.0,
    #             "goal_y": 2.0,
    #             "dist": round(math.hypot(2.0 - (robot_x or 0), 2.0 - (robot_y or 0)), 2) if robot_x is not None else None
    #         },
    #         "sensor": {
    #             "laser_ranges": sensor_data.get("laser_ranges"),
    #             "min_laser": min_laser
    #         }
    #     }
    
    def get_telemetry(self) -> Dict[str, Any]:
        now = time.time()
        delta = now - self._last_time if self._last_time else 0.1
        self._last_time = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / delta) if delta > 0 else 10.0

        # 获取底层数据（可能为 None）
        robot_data = self.robot_service.get_odom_data()
        sensor_data = self.robot_service.get_sensor_data()
        rl_data = self.rl_service.get_state()  # 假设 rl_service 已处理 None

        # 安全解包：如果 robot_data 为 None，则提供全 None 占位字典
        if robot_data is None:
            robot_data = {"x": None, "y": None, "yaw": None, "vx": None, "wz": None}
        if sensor_data is None:
            sensor_data = {"laser_ranges": None, "min_laser": None}

        # 计算目标距离（若坐标缺失则设为 None）
        goal_x = 2.0  # 建议从配置或状态管理获取
        goal_y = 2.0
        dist = None
        if robot_data["x"] is not None and robot_data["y"] is not None:
            dist = round(math.hypot(goal_x - robot_data["x"], goal_y - robot_data["y"]), 2)

        # 新增视频流统计
        streams = {}
        for name, streamer in [("rgb", self.rgb_streamer), ("depth", self.depth_streamer), ("map", self.map_streamer)]:
            if streamer:
                streams[name] = {
                    "fps": streamer.get_fps(),          # 实际接收帧率
                    "latency_ms": streamer.get_latency_ms()  # 最新帧延迟(ms)
                }
                
        return {
            "perf": {"step_fps": self._fps},
            "rl": rl_data if rl_data is not None else {},
            "robot": robot_data,
            "task": {
                "goal_x": goal_x,
                "goal_y": goal_y,
                "dist": dist
            },
            "sensor": sensor_data,
            "streams": streams   # 新增
        }

    # def get_telemetry(self) -> Dict[str, Any]:
    #     now = time.time()
    #     delta = now - self._last_time if self._last_time else 0.1
    #     self._last_time = now
    #     self._fps = 0.9 * self._fps + 0.1 * (1.0 / delta) if delta > 0 else 10.0

    #     robot_data = self.robot_service.get_odom_data()
    #     sensor_data = self.robot_service.get_sensor_data()
    #     rl_data = self.rl_service.get_state()

    #     # 严格对齐前端 TelemetryView.draw() 的解构格式
    #     return {
    #         "perf": {"step_fps": self._fps},
    #         "rl": rl_data,
    #         "robot": robot_data,
    #         "task": {
    #             "goal_x": 2.0, "goal_y": 2.0, # 建议后续从 env.state 注入
    #             "dist": round(math.hypot(2.0 - robot_data["x"], 2.0 - robot_data["y"]), 2)
    #         },
    #         "sensor": sensor_data
    #     }
