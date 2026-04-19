# core/ros2/channels/streamers/map.py

import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from core.ros2.channels.streamers.base import BaseStreamer

import logging
logger = logging.getLogger(__name__)


class MapProvider(BaseStreamer):
    """
    寄人篱下：名义上继承 Streamer 借用底层基建，语义上是低频状态提供者。
    """
    def __init__(self, runtime, topic: str = '/map'):
        # 借用 Streamer 的初始化，FPS 限流设为 1
        super().__init__(node_name='web_map_provider', runtime=runtime, default_fps=1)
        
        # /map 的专属 QoS（不设这个绝对收不到数据）
        self.qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            # durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # 预分配 LUT
        self.color_lut = np.zeros((256, 3), dtype=np.uint8)
        self.color_lut[0] = [254, 254, 254]     
        self.color_lut[100] = [0, 0, 0]          
        self.color_lut[255] = [205, 205, 205]     
        
        logger.info(f"[MapProvider] 待寄生 {topic}")
        self.sub = self.create_subscription(
            OccupancyGrid, topic, self._process_msg, self.qos_profile, callback_group=self.cg
        )
        logger.info(f"[MapProvider] 已寄生 {topic}")

    def _process_msg(self, msg: OccupancyGrid):
        self._record_frame(msg)   # ← 新增
        try:
            logger.debug(f"[MapProvider] 收到 /map 消息，分辨率: {msg.info.resolution}")
            grid = np.frombuffer(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
            grid_uint8 = np.where(grid == -1, 255, grid.astype(np.uint8))
            # cv2.LUT 要求输入图像和查找表的通道数一致。你的 grid_uint8 是单通道（灰度），而 self.color_lut 是 3 通道（RGB），导致通道数不匹配的断言失败。
            # rgb_array = cv2.LUT(grid_uint8, self.color_lut)
            rgb_array = self.color_lut[grid_uint8]
            success, png_bytes = cv2.imencode('.png', rgb_array, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if success:
                # 把处理好的 PNG 塞进 Streamer 父类的口袋里
                self._latest_frame = png_bytes.tobytes()
                logger.debug(f"[MapProvider] PNG 编码成功，大小: {len(self._latest_frame)}")
        except Exception as e:
            logger.warning(f"[MapProvider] 地图转换失败: {e}")

    # def _process_msg(self, msg: OccupancyGrid):
    #     try:
    #         grid = np.frombuffer(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    #         grid_uint8 = np.where(grid == -1, 255, grid.astype(np.uint8))
    #         rgb_array = self.color_lut[grid_uint8]
    #         success, png_bytes = cv2.imencode('.png', rgb_array, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    #         if success:
    #             self._latest_frame = png_bytes.tobytes()
    #             logger.info(f"[MapProvider] 地图已更新，大小: {len(self._latest_frame)} bytes")
    #         else:
    #             logger.warning("[MapProvider] PNG 编码失败")
    #     except Exception as e:
    #         logger.warning(f"[MapProvider] 地图转换失败: {e}")