import logging
from planning.brain.clients.llm_client import get_intent_router_llm, get_mission_planner_llm
from planning.brain.prompts.brain_prompts import SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("BrainNode.Planner")

# 初始化 LLM 实例（模块级别，避免重复初始化）
intent_router = get_intent_router_llm()
mission_planner = get_mission_planner_llm()

def node_planner(state: dict) -> dict:
    user_input = state.get("user_input", "")
    logger.info(f"[Planner] 开始解析: {user_input}")

    try:
        # ---- 第一步：意图识别 ----
        intent_dict = intent_router.invoke([HumanMessage(content=user_input)])
        mission_type = intent_dict.get("mission_type", "chat")  # 用 "intent" 而不是 "mission_type"
        logger.info(f"[Planner] 意图识别完成: {mission_type}")

        # ---- 第二步：根据意图分发 ----
        if mission_type == "mission":
            goal = intent_dict.get("goal", user_input)
            
            # 调用任务规划器生成 blocks
            blocks = mission_planner.plan(goal=goal)
            blocks_dict = [b.model_dump() for b in blocks]
            
            logger.info(f"[Planner] 任务拆解成功 -> 生成 {len(blocks_dict)} 个 blocks")
            
            return {
                "mission_type": "mission",
                "user_goal": goal,
                "mission_blocks": blocks_dict,
                "current_block_idx": 0,
                "block_results": [],
                "block_retry_counts": {}
            }
            
        elif mission_type == "shutdown":
            return {
                "mission_type": "shutdown",
                "is_running": False,
                "response": "正在关机..."
            }
            
        else:  # chat
            response = intent_dict.get("response", "你好呀")
            return {
                "mission_type": "chat",
                "response": response
            }

    except Exception as e:
        logger.error(f"[Planner] 兜底触发: {e}", exc_info=True)
        return {
            "mission_type": "chat",
            "reasoning": "error",
            "response": "大脑短路，请重试。",
            "mission_blocks": [],
            "user_goal": "",
        }
