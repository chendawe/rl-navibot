import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Dict, Any, Tuple

from groundingdino.util.inference import load_model, predict, annotate as gd_annotate
import groundingdino.datasets.transforms as T

class GroundingDINODetector:
    """GroundingDINO 接口（输入 RGB，输出 RGB 标注图）"""

    def __init__(self, config_path: str, checkpoint_path: str):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.backends.cudnn.benchmark = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(config_path, checkpoint_path, device=self.device)
        self.model.eval()
        print(f"[GroundingDINODetector] 模型加载完成 (设备: {self.device})")

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def show_image(image_np: np.ndarray, title: str = "Image"):
        """纯图片展示器，默认接收 RGB 格式"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(image_np)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def _load_image_from_numpy(self, image_np_rgb: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        if image_np_rgb.ndim != 3 or image_np_rgb.shape[2] != 3:
            raise ValueError("需要 (H, W, 3) 的 RGB numpy")
        image_pil = Image.fromarray(image_np_rgb).convert("RGB")
        image_transformed, _ = self.transform(image_pil, None)
        image_source = np.asarray(image_pil)
        return image_source, image_transformed

    def _warmup(self, image_np_rgb: np.ndarray):
        _, image_tensor = self._load_image_from_numpy(image_np_rgb)
        t0 = time.time()
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    predict(model=self.model, image=image_tensor, caption="warmup .",
                            box_threshold=0.35, text_threshold=0.25, device=self.device)
            else:
                predict(model=self.model, image=image_tensor, caption="warmup .",
                            box_threshold=0.35, text_threshold=0.25, device=self.device)
        del image_tensor
        torch.cuda.empty_cache()
        print(f"[GroundingDINODetector] 预热完成，耗时 {time.time() - t0:.2f}s")

    def detect(self, image_np_rgb: Optional[np.ndarray], text_prompt: str,
               box_threshold: float = 0.35, text_threshold: float = 0.25) -> Dict[str, Any]:
        """
        输入: RGB numpy (H, W, 3)
        输出: annotated_image 统一转成 RGB
        """
        if image_np_rgb is None:
            return {"status": "failed", "detail": "输入图像为空"}

        image_source_rgb, image_tensor = self._load_image_from_numpy(image_np_rgb)

        t0 = time.time()
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    boxes, logits, phrases = predict(
                        model=self.model, image=image_tensor, caption=text_prompt,
                        box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
            else:
                boxes, logits, phrases = predict(
                    model=self.model, image=image_tensor, caption=text_prompt,
                    box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
        elapsed = time.time() - t0

        # gd_annotate 输入 RGB，输出 BGR
        annotated_bgr = gd_annotate(
            image_source=image_source_rgb, boxes=boxes, logits=logits, phrases=phrases
        )

        # 🔥 统一转成 RGB 对外输出
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return {
            "status": "success",
            "boxes": boxes,
            "logits": logits,
            "phrases": phrases,
            "annotated_image": annotated_rgb,
            "inference_time_s": elapsed,
        }




# # 示例：从 RGBBridge JPEG bytes -> RGB numpy
# jpeg_bytes = rgb_bridge.get_latest_frame()
# nparr = np.frombuffer(jpeg_bytes, np.uint8)
# img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 统一用 RGB 进模型

# result = groundingdino_detector.detect(
#     image_np_rgb=img_rgb,
#     text_prompt="cup . chair . bag .",
#     box_threshold=0.35,
#     text_threshold=0.25,
# )
