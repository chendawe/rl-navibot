from sensor_msgs.msg import CompressedImage, Image
from core.ros2.channels.streamers.base import BaseStreamer
# from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)

class RGBStreamer(BaseStreamer):
    """专门处理 RGB 摄像头，零 CPU 消耗神仙操作"""
    def __init__(self, runtime, topic: str = '/camera/image_raw/compressed'):
        super().__init__(node_name='web_rgb_streamer', runtime=runtime, default_fps=15)
        # 🌟 新增：设置 QoS 策略为 BEST_EFFORT
        self.qos_profile = QoSProfile(
            depth=10,  # 缓存最近 10 帧
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 关键：改为 BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST
        )
        
        # 🔥 唯一改动：绑定 callback_group=self.cg
        image_type = None
        if topic == '/camera/image_raw/compressed' :
            image_type = CompressedImage
        if topic == '/camera/image_raw' :
            image_type = Image
            # self._bridge = CvBridge()
        self.sub = self.create_subscription(
            image_type, topic, self._process_msg, self.qos_profile, callback_group=self.cg
        )

    def _process_msg(self, msg: CompressedImage):
        # 🌟 神仙操作：ROS2 驱动层已经压好 JPEG 了，直接拿字节，零拷贝！
        
        # try:
        #     if isinstance(msg, CompressedImage):
        #         # 如果是压缩图，直接拿字节（零拷贝）
        #         self._latest_frame = msg.data
        #     else:
        #         # 🌟 如果是原始图，用 cv_bridge 转成 numpy，再压成 JPEG
        #         cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #         success, jpeg_bytes = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        #         if success:
        #             self._latest_frame = jpeg_bytes.tobytes()  # 存的是真 JPEG 了！
        # except Exception as e:
        #     print(f"[RGBStreamer] 图像处理失败: {e}")
        self._record_frame(msg)   # ← 新增
        try:
            if isinstance(msg, CompressedImage):
                # 压缩图：直接拿 JPEG 字节
                self._latest_frame = msg.data
            else:
                # 🌟 手动解析：不用 cv_bridge！
                # msg.data 是一维 bytes，按 (height, width, 3) reshape
                img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 3)
                )
                # 检查编码并转换到 BGR
                if msg.encoding == "rgb8":
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif msg.encoding == "bgr8":
                    pass  # 已经是 BGR，无需转换
                else:
                    logger.warning(f"未知图像编码: {msg.encoding}，按 BGR 处理")
                # 压成 JPEG
                success, jpeg_array = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    # 1. tobytes() 在内存里生成了一个【全新的】 bytes 对象
                    jpeg_bytes = jpeg_array.tobytes() 
                    
                    # 2. 下面这一行，在 CPython 中仅仅是一条字节码指令 (STORE_ATTR)
                    # 它的本质是把 self._latest_frame 这个“指针”，指向新对象的内存地址
                    self._latest_frame = jpeg_bytes
                    logger.debug(f"[DEBUG] JPEG 头: {jpeg_bytes[:3].hex()}")
                    
                    # 应该是 ff d8 ff
        except Exception as e:
            print(f"[RGBStreamer] 图像处理失败: {e}")

