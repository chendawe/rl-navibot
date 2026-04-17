import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

class RGBService:
    def __init__(self, streamer=None):
        self.streamer = streamer
        
        # 🛡️ 兜底策略：生成一张 320x240 的纯黑 JPEG 图，避免前端 <img> 标签报错
        _, img_array = cv2.imencode('.jpg', np.zeros((240, 320, 3), dtype=np.uint8))
        self._mock_frame = img_array.tobytes()
        
        self._last_frame_id = None  # 用于判断画面是否真的更新了

    def get_frame(self) -> bytes:
        if self.streamer:
            try:
                current_frame = getattr(self.streamer, '_latest_frame', None)
                if current_frame is not None:
                    current_id = id(current_frame)
                    if current_id != self._last_frame_id:
                        self._last_frame_id = current_id
                        return current_frame
                    else:
                        # 画面没变，跳过发送
                        return None
                else:
                    # 🌟 明确处理：ROS2 还没推数据过来，直接返回 None，别发黑屏！
                    return None 
            except Exception as e:
                logger.warning(f"[RGBService] 获取画面失败，启用黑屏兜底: {e}")
                
        # 只有在 streamer 为 None 或者异常时，才走黑屏
        print("[DEBUG] 使用 Mock 图像")
        self._last_frame_id = None
        return self._mock_frame

