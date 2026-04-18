# core/ros2/channels/services/depth.py

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class DepthService:
    """深度图镜像：帧去重 + 空值兜底 + 异常隔离"""
    def __init__(self, streamer=None):
        self.streamer = streamer
        # 兜底：生成一张 320x240 的纯灰 JPEG（深度图的"无数据"语义）
        _, img_array = cv2.imencode('.jpg', np.full((240, 320), 128, dtype=np.uint8))
        self._mock_frame = img_array.tobytes()
        self._last_frame_id = None

    def get_frame(self) -> bytes:
        if self.streamer:
            try:
                current_frame = getattr(self.streamer, '_latest_frame', None)
                if current_frame is not None:
                    current_id = id(current_frame)
                    if current_id != self._last_frame_id:
                        self._last_frame_id = current_id
                        return current_frame
                    return None
                return None
            except Exception as e:
                logger.warning(f"[DepthService] 获取深度帧失败，启用灰屏兜底: {e}")
        logger.debug("[DepthService] 使用 Mock 深度图")
        self._last_frame_id = None
        return self._mock_frame
