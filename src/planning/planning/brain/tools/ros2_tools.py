# import json
# from langchain_core.tools import tool
# from execution.vision.snapshot import SnapShotVisioner

# def create_vision_tools(visioner: SnapShotVisioner):
#     """工厂函数：注入 Visioner 实例，返回绑定的 Tool 列表"""

#     @tool
#     def yolo_detect() -> str:
#         """使用YOLO模型检测当前视野中的常见物体(如人、杯子、椅子等)。无需传入参数。"""
#         result = visioner.yolo_detect()
        
#         if result["status"] != "success":
#             return f"YOLO检测失败: {result.get('detail', '未知错误')}"
        
#         boxes = result["boxes"]
#         if boxes is None or len(boxes) == 0:
#             return "当前视野中未检测到任何常见物体。"

#         infos = []
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             cls_name = visioner.yolo_detector.model.names[cls_id]
#             conf = float(box.conf[0])
#             xyxy = box.xyxy[0].cpu().numpy().tolist()
#             infos.append({
#                 "object": cls_name,
#                 "confidence": round(conf, 2),
#                 "bbox": [round(x, 1) for x in xyxy]
#             })

#         return f"检测到 {len(infos)} 个物体: {json.dumps(infos, ensure_ascii=False)}"


#     @tool
#     def groundingdino_detect(text_prompt: str, box_threshold: float = 0.35) -> str:
#         """使用GroundingDINO根据文本描述查找特定物体。
        
#         Args:
#             text_prompt: 想要查找的物体名称或文本描述 (例如: 'red cup', 'chair')
#             box_threshold: 检测的置信度阈值，默认0.35。低于此阈值的物体将被过滤。
#         """
#         result = visioner.groundingdino_detect(
#             text_prompt=text_prompt,
#             box_threshold=box_threshold
#         )
        
#         if result["status"] != "success":
#             return f"GroundingDINO检测失败: {result.get('detail', '未知错误')}"
        
#         boxes = result["boxes"]
#         phrases = result["phrases"]
#         logits = result["logits"]

#         if len(boxes) == 0:
#             return f"未找到与 '{text_prompt}' 匹配的物体。"

#         infos = []
#         for phrase, logit, box in zip(phrases, logits, boxes):
#             infos.append({
#                 "object": phrase,
#                 "confidence": round(float(logit), 2),
#                 "bbox_normalized": [round(x, 3) for x in box.cpu().numpy().tolist()]
#             })

#         return f"找到 {len(infos)} 个 '{text_prompt}': {json.dumps(infos, ensure_ascii=False)}"

#     # 返回绑定好实例的 tool 函数列表
#     return [yolo_detect, groundingdino_detect]


# def create_navigate_tool(navigator):
#     """工厂函数：注入 Navigator 实例，返回绑定的导航 Tool"""

#     @tool
#     def navigate_to(x: float, y: float, theta: float = 0.0) -> str:
#         """控制机器人导航到指定的坐标点。
        
#         Args:
#             x: 目标位置的 X 坐标 (浮点数)
#             y: 目标位置的 Y 坐标 (浮点数)
#             theta: 目标位置的朝向角度，单位为弧度 (浮点数，默认0.0)
#         """
#         try:
#             # 假设你的 navigator 有 go_to 方法
#             # navigator.go_to(x, y, theta)
#             return f"已发送导航指令，目标位置: x={x}, y={y}, theta={theta}"
#         except Exception as e:
#             return f"导航失败: {str(e)}"

#     return [navigate_to]



# app/harness/tools/ros_tools.py
import logging
from langchain_core.tools import tool

logger = logging.getLogger("ROSTool")

@tool
def ros_navigate(target: str) -> dict:
    """导航到指定目标点"""
    import time, random
    time.sleep(0.5)
    success = random.random() > 0.5
    if success: return {"status": "success", "detail": f"已到达 {target}"}
    else: return {"status": "failed", "detail": f"{target} 不可达"}

@tool
def ros_observe(target: str) -> dict:
    """观察指定目标物体"""
    import time, random
    time.sleep(0.3)
    success = random.random() > 0.5
    if success: return {"status": "success", "detail": f"已识别 {target}"}
    else: return {"status": "timeout", "detail": f"等待 {target} 超时"}
