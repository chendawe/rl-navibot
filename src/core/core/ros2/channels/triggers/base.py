# core/ros2/channels/triggers/base.py

import time
from abc import ABC, abstractmethod
from typing import Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import logging
logger = logging.getLogger(__name__)


class BaseTrigger(Node, ABC):
    """
    事件触发型节点的抽象基类。
    
    与 Streamer 的区别：
    - Streamer：持续订阅数据流，存储最新帧，供轮询
    - Trigger：订阅轻量级通知，设置触发标志，供消费者检查
    
    典型场景：
    - /slam_toolbox/update → 地图已更新，前端该刷新了
    - /map_saver/transition_event → 地图保存状态变更
    """
    def __init__(self, node_name: str, runtime, default_cooldown: float = 0.5):
        super().__init__(node_name=node_name)
        runtime.register_node(self)
        
        # 冷却时间：防止短时间内重复触发
        self._default_cooldown = default_cooldown
        self._last_trigger_time: float = 0.0
        
        # 触发标志 + 附带数据
        self._triggered: bool = False
        self._payload: Optional[Any] = None
        
        # 互斥回调组
        self.cg = MutuallyExclusiveCallbackGroup()

    def is_triggered(self) -> bool:
        """
        检查是否被触发（带冷却）。
        返回 True 后自动重置标志，确保「一次触发只消费一次」。
        """
        now = time.time()
        if self._triggered and (now - self._last_trigger_time >= self._default_cooldown):
            self._triggered = False
            self._last_trigger_time = now
            return True
        return False

    def peek(self) -> bool:
        """
        只看不消费：检查是否被触发，但不重置标志。
        用于 UI 侧边栏显示状态小圆点等场景。
        """
        return self._triggered

    def get_payload(self) -> Optional[Any]:
        """获取触发时附带的数据（如有），不重置标志"""
        return self._payload

    @abstractmethod
    def _process_msg(self, msg):
        """
        子类核心：收到通知后的处理逻辑。
        设置 self._triggered = True，可选设置 self._payload。
        """
        pass
