import numpy as np
import cv2
import logging
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from core.ros2.channels.streamers.base import BaseStreamer

logger = logging.getLogger(__name__)

class DRGStreamer(BaseStreamer):
    """处理 SLAM 栅格地图。1Hz 更新，FPS 限流主要用于防重入"""
    
    # ==========================================
    # 🔥 类变量：全局只生成一次，所有实例共享，零消耗
    # ==========================================
    _LUT = np.full(256, 128, dtype=np.uint8)  # 1. 默认全填满 128(未知灰)
    _LUT[0] = 255                             # 2. ROS标准：0(空闲) -> 纯白
    _LUT[100] = 0                             # 3. ROS标准：100(障碍) -> 纯黑
    # 注意：ROS 里的 -1 转成 uint8 就是 255，因为第一步已经填了 128，所以不用再单独写 _LUT[255]=128 了

    def __init__(self, runtime, topic: str = '/map'):
        super().__init__(node_name='web_map_streamer', runtime=runtime, default_fps=2)
        self.qos_profile = QoSProfile(
            depth=10,  # 缓存最近 10 帧
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 关键：改为 BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST
        )
        self.sub = self.create_subscription(
            OccupancyGrid, topic, self._process_msg, self.qos_profile, callback_group=self.cg
        )
        
    def _process_msg(self, msg: OccupancyGrid):
        try:
            # ROS 的 msg.data 本来是 int8 列表，转成 uint8 后，-1 自动变成 255，完美对应上面的 _LUT
            data = np.array(msg.data, dtype=np.uint8).reshape(msg.info.height, msg.info.width)
            
            # 注意这里加 self，调用上面的类变量
            map_img = self._LUT[data]
            
            _, png_bytes = cv2.imencode('.png', map_img)
            self._latest_frame = png_bytes.tobytes()
        except Exception as e:
            logger.warning(f'地图处理失败: {e}')


    # def _process_msg(self, msg: OccupancyGrid):
    #     try:
    #         data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    #         map_img = np.zeros_like(data, dtype=np.uint8)
    #         map_img[data == 0] = 255   # 空闲 -> 白
    #         map_img[data == -1] = 128  # 未知 -> 灰
    #         map_img[data == 100] = 0   # 障碍 -> 黑
            
    #         # 必须用 PNG 无损压缩
    #         _, png_bytes = cv2.imencode('.png', map_img)
    #         self._latest_frame = png_bytes.tobytes()
    #     except Exception:
    #         logger.warning(f'地图处理失败: {e}')
