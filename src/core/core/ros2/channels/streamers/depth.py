
import cv2
import numpy as np
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from core.ros2.channels.streamers.base import BaseStreamer


class DepthStreamer(BaseStreamer):
    """处理深度图，Float32 -> 8bit 灰度的轻量转换"""
    def __init__(self, runtime, topic: str = '/camera/depth/image_raw'):
        super().__init__(node_name='web_depth_streamer', runtime=runtime, default_fps=10)
        self.qos_profile = QoSProfile(
            depth=10,  # 缓存最近 10 帧
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 关键：改为 BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST
        )
        # 🔥 唯一改动：绑定 callback_group=self.cg
        self.sub = self.create_subscription(
            Image, topic, self._process_msg, self.qos_profile, callback_group=self.cg
        )

    def _process_msg(self, msg: Image):
        try:
            depth_arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            depth_8u = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            _, jpg_bytes = cv2.imencode('.jpg', depth_8u, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._latest_frame = jpg_bytes.tobytes()
        except Exception:
            logger.warning(f'深度图处理失败: {e}')
