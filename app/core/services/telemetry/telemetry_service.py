import math
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TelemetryService:
    def __init__(self, robot_service, rl_service):
        self.robot_service = robot_service
        self.rl_service = rl_service
        self._last_time = time.time()
        self._fps = 10.0

    def get_telemetry(self) -> Dict[str, Any]:
        now = time.time()
        delta = now - self._last_time if self._last_time else 0.1
        self._last_time = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / delta) if delta > 0 else 10.0

        robot_data = self.robot_service.get_odom_data()
        sensor_data = self.robot_service.get_sensor_data()
        rl_data = self.rl_service.get_state()

        # 严格对齐前端 TelemetryView.draw() 的解构格式
        return {
            "perf": {"step_fps": self._fps},
            "rl": rl_data,
            "robot": robot_data,
            "task": {
                "goal_x": 2.0, "goal_y": 2.0, # 建议后续从 env.state 注入
                "dist": round(math.hypot(2.0 - robot_data["x"], 2.0 - robot_data["y"]), 2)
            },
            "sensor": sensor_data
        }
