import logging
from langgraph.types import interrupt

logger = logging.getLogger("BrainNode.Input")

def node_user_input(state: dict) -> dict:
    logger.info("系统挂起，等待用户输入...")
    user_msg = interrupt("等待用户指令...")
    logger.info(f"收到用户输入: {user_msg}")
    return {"user_input": user_msg}
