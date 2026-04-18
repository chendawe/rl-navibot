# core/ros2/channels/services/map.py

import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)

class MapService:
    def __init__(self, trigger=None, provider=None):
        self.trigger = trigger
        self.provider = provider
        
        # 兜底：一张小灰图 PNG
        _, img_array = cv2.imencode('.png', np.full((240, 320, 3), 205, dtype=np.uint8),
                                     [cv2.IMWRITE_PNG_COMPRESSION, 9])
        self._mock_frame = img_array.tobytes()
        
        # 🌟 修复：用 bytes 真实内容去重，而不是用 id()
        self._last_frame_bytes = None

    def get_frame(self) -> bytes:
        # # 1. Trigger 没触发 → 不拉取
        # if self.trigger and not self.trigger.is_triggered():
        #     return None
        
        # 2. 拿到标准化的 bytes 帧
        frame_bytes = self._extract_frame()
        if frame_bytes is None:
            return None
            
        # 3. 🌟 修复：如果是同一张图，就不发了（防止 SLAM 地图没变化时狂发死循环）
        if frame_bytes == self._last_frame_bytes:
            return None
            
        self._last_frame_bytes = frame_bytes
        return frame_bytes

    def _extract_frame(self) -> bytes:
        if self.provider:
            try:
                current_frame = getattr(self.provider, '_latest_frame', None)
                if current_frame is not None:
                    # 🌟 核心修复：无论是 ndarray 还是 bytes，统一无脑转成 PNG bytes！
                    if isinstance(current_frame, np.ndarray):
                        # 如果是 numpy 数组，压缩成 PNG
                        _, encoded_img = cv2.imencode('.png', current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                        return encoded_img.tobytes()
                    elif isinstance(current_frame, bytes):
                        # 如果已经是 bytes，直接返回
                        return current_frame
            except Exception as e:
                logger.warning(f"[MapService] 获取或编码地图帧失败: {e}")
        
        # 拿不到就返回兜底灰图
        return self._mock_frame
