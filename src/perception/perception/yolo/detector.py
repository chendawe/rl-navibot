import time
import cv2
import numpy as np
from typing import Optional, Dict, Any

class YOLODetector:
    """YOLO 目标检测器（输入 RGB，输出 RGB 标注图）"""
    
    def __init__(self, model_path: str):
        from ultralytics import YOLO
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        
        print(f"[YOLO] 正在热身...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.model(dummy_img, verbose=False)
        print(f"[YOLO] 模型加载并热身完成 (设备: {self.device})")

    @staticmethod
    def show_image(image_np: np.ndarray, title: str = "Image"):
        """纯图片展示器，接收 RGB 格式"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(image_np)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def detect(self, image_np: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        输入: RGB numpy (H, W, 3) —— 统一约定
        输出: annotated_image 也是 RGB
        """
        if image_np is None:
            return {"status": "failed", "detail": "输入图像为空"}

        # 🔥 1. 强制把输入转成 BGR 喂给 YOLO (因为 YOLO 默认吃 BGR)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        results = self.model(image_bgr, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000
        
        boxes = results[0].boxes if len(results) > 0 else None
        
        # 🔥 2. YOLO 的 plot() 输出的必定是 BGR
        annotated_bgr = results[0].plot() if len(results) > 0 else image_bgr.copy()
        
        # 🔥 3. 强制把输出转回 RGB 返回给外部
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
        return {
            "status": "success",
            "boxes": boxes,
            "annotated_image": annotated_rgb,
            "inference_time_ms": infer_ms
        }
