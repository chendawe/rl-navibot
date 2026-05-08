import cv2
import numpy as np
from typing import Optional, Dict, Any

# 从 perception 层导入接口
from perception.yolo.detector import YOLODetector
from perception.groundingDINO.detector import GroundingDINODetector

class SnapShotVisioner:
    """视觉快照调度器（解耦版）"""
    
    def __init__(self, rgb_bridge, yolo_detector, groundingdino_detector):
        """
        :param rgb_bridge: RGBBridge 实例，负责取图
        :param yolo_detector: YOLODetector 实例
        :param groundingdino_detector: GroundingDINODetector 实例
        """
        self.rgb_bridge = rgb_bridge
        self.yolo_detector = yolo_detector
        self.groundingdino_detector = groundingdino_detector

    def _get_safe_frame(self) -> np.ndarray:
        """
        底层取图兜底，统一输出 RGB 格式!!!
        如果仿真没开，Bridge 取不到图，直接返回一张纯黑 numpy 图，
        保证 perception 的 baseline 逻辑能顺利走通不报错。
        """
        frame_bytes = self.rgb_bridge.get_latest_frame()
        if frame_bytes is not None:
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            # cv2 解码出来默认是 BGR
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                # 🔥 强制转成 RGB 对外输出，保证和相机驱动/ROS的RGB约定一致
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
        # 没拿到图，生成一张假图保命 (全黑图 RGB 和 BGR 一样，但规范起见按 RGB 约定)
        print("[Warning] RGBBridge 取图失败，使用全黑假图!")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def yolo_detect(self) -> Dict[str, Any]:
        """调用 YOLO 检测图中的实体"""
        img_rgb = self._get_safe_frame()
        return self.yolo_detector.detect(img_rgb)

    def groundingdino_detect(self, text_prompt: str, 
                             box_threshold: float = 0.35, 
                             text_threshold: float = 0.25) -> Dict[str, Any]:
        """调用 GroundingDINO 查找帧图中想要的实体"""
        img_rgb = self._get_safe_frame()
        return self.groundingdino_detector.detect(
            image_np=img_rgb, 
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
