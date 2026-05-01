import logging
from app.harness.clients.llm_client import get_planner_llm
from app.harness.prompts.brain_prompts import SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("BrainNode.Planner")

def node_planner(state: dict) -> dict:
    logger.info(f"[Planner] 开始解析: {state['user_input']}")
    structured_llm = get_planner_llm()
    
    try:
        parsed = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["user_input"])
        ])
        result_dict = parsed.model_dump()
        logger.info(f"[Planner] 解析成功 -> 类型: {result_dict['mission_type']}")
        # ---- 关键：如果是 mission，补上循环变量 ----
        if result_dict["mission_type"] == "mission":
            result_dict["current_block_idx"] = 0
            result_dict["block_results"] = []
        return result_dict
    except Exception as e:
        logger.error(f"[Planner] 兜底触发: {e}")
        return {
            "mission_type": "chat",
            "reasoning": "error",
            "response": "大脑短路，请重试。",
            "mission_blocks": []
        }
