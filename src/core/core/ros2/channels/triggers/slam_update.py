# core/ros2/channels/triggers/map_update.py

from std_msgs.msg import Empty
from core.ros2.channels.triggers.base import BaseTrigger

import logging
logger = logging.getLogger(__name__)


class SlamUpdateTrigger(BaseTrigger):
    """
    监听 /slam_toolbox/update，地图有更新时触发。
    
    用法：
        trigger = MapUpdateTrigger(runtime)
        
        # 在 FastAPI 的轮询接口里：
        if trigger.is_triggered():
            # 地图更新了，去拿最新的 /map 数据渲染
            pass
            
        # 在 UI 状态展示里：
        if trigger.peek():
            # 显示 "地图更新中" 的小绿点
            pass
    """
    def __init__(self, runtime, topic: str = '/slam_toolbox/update'):
        super().__init__(
            node_name='web_map_update_trigger',
            runtime=runtime,
            default_cooldown=0.5  # 500ms 冷却，防止地图疯狂刷
        )
        self._trigger_count = 0
        
        self.sub = self.create_subscription(
            Empty, topic, self._process_msg, 10, callback_group=self.cg
        )
        logger.info(f"[MapUpdateTrigger] 已订阅 {topic}")

    def _process_msg(self, msg: Empty):
        self._triggered = True
        self._payload = None  # Empty 消息，无附带数据
        self._trigger_count += 1
        logger.debug(f"[MapUpdateTrigger] 收到更新通知 (累计 {self._trigger_count} 次)")
