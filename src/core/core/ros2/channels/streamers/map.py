import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from core.ros2.channels.streamers.base import BaseStreamer

class DRGStreamer(BaseStreamer):
    """处理 SLAM 栅格地图。1Hz 更新，FPS 限流主要用于防重入"""
    def __init__(self, runtime, topic: str = '/map'):
        super().__init__(node_name='web_map_streamer', runtime=runtime, default_fps=2)
        # 🔥 唯一改动：绑定 callback_group=self.cg
        self.sub = self.create_subscription(
            OccupancyGrid, topic, self._process_msg, 10, callback_group=self.cg
        )

    def _process_msg(self, msg: OccupancyGrid):
        try:
            data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
            map_img = np.zeros_like(data, dtype=np.uint8)
            map_img[data == 0] = 255   # 空闲 -> 白
            map_img[data == -1] = 128  # 未知 -> 灰
            map_img[data == 100] = 0   # 障碍 -> 黑
            
            # 必须用 PNG 无损压缩
            _, png_bytes = cv2.imencode('.png', map_img)
            self._latest_frame = png_bytes.tobytes()
        except Exception:
            pass
