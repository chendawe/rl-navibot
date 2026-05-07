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
