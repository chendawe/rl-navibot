import random
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class RLService:
    def __init__(self, env=None):
        self.env = env

    def get_state(self) -> Dict[str, Any]:
        """
        只读获取 RL 状态，绝不调用 step() 推演。
        前提：建议在你的 TurtleBot3NaviEnv 的 step() 末尾加上缓存：
              self.last_reward = reward
              self.last_info = info
        """
        if self.env:
            try:
                # 尝试读取 Env 缓存的上一步状态
                reward = getattr(self.env, 'last_reward', 0.0)
                info = getattr(self.env, 'last_info', {})
                return {
                    "reward": float(reward),
                    "collision": info.get("collision", False),
                    "goal_reached": info.get("goal_reached", False)
                }
            except Exception as e:
                logger.warning(f"[RLService] 读取 Env 状态失败，启用 Mock: {e}")

        # 兜底 Mock
        return {
            "reward": round(random.uniform(-1.0, 5.0), 2),
            "collision": random.random() < 0.05,
            "goal_reached": random.random() < 0.01
        }

    def step_action(self, action) -> Dict[str, Any]:
        """仅供给 /ws/rl 专用，接收前端下发的动作进行推演"""
        if self.env:
            try:
                action_np = np.array(action, dtype=np.float32)
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                return {
                    "reward": float(reward), 
                    "terminated": terminated, 
                    "truncated": truncated, 
                    "info": info
                }
            except Exception as e:
                return {"error": str(e)}
        return {"reward": 0.0, "terminated": False, "truncated": False, "info": {}}
