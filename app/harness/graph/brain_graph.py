from langgraph.graph import StateGraph, END, START
from app.harness.schemas.brain_schema import BrainState
from app.harness.nodes.input_node import node_user_input
from app.harness.nodes.planner_node import node_planner
from app.harness.nodes.action_node import node_chat, node_execute_block, node_check_block, node_replan, node_shutdown

import logging
logger = logging.getLogger("buildBrainGraph")

def route_task(state: BrainState) -> str:
    return state.get("mission_type", "chat")

# def build_brain_graph(checkpointer=None):
#     workflow = StateGraph(BrainState)

#     workflow.add_node("user_input", node_user_input)
#     workflow.add_node("planner",    node_planner)
#     workflow.add_node("chat",       node_chat)
#     workflow.add_node("mission",    node_mission)
#     workflow.add_node("shutdown",   node_shutdown)

#     workflow.add_edge(START, "user_input")
#     workflow.add_edge("user_input", "planner")
    
#     workflow.add_conditional_edges(
#         "planner", route_task,
#         {"chat": "chat", "mission": "mission", "shutdown": "shutdown"}
#     )
    
#     workflow.add_edge("chat",     "user_input")
#     workflow.add_edge("mission",  "user_input")
#     # workflow.add_edge("shutdown", END)
#     workflow.add_edge("shutdown", "user_input")

#     return workflow.compile(checkpointer=checkpointer)

# ============================================================
# 路由 2：Block 执行后分派（核心路由）
# ============================================================
def route_after_check(state: BrainState) -> str:
    idx = state["current_block_idx"]
    total = len(state["mission_blocks"])

    # 情况 A：所有 block 跑完了
    if idx >= total:
        logger.info(f"[Route] 所有 {total} 个 block 完成 -> user_input")
        return "all_done"

    # 情况 B：检查最后一个 block 的结果
    last_result = state["block_results"][-1]

    # 情况 C：成功 -> 继续执行下一个
    if last_result["status"] == "success":
        logger.info(f"[Route] Block[{idx-1}] 成功 -> 继续执行 Block[{idx}]")
        return "continue"

    # 情况 D：失败 -> 去 replan
    logger.info(f"[Route] Block[{idx-1}] 失败 -> replan")
    return "replan"


def build_brain_graph(checkpointer=None):
    workflow = StateGraph(BrainState)

    # ---- 注册所有节点 ----
    workflow.add_node("user_input",     node_user_input)
    workflow.add_node("planner",        node_planner)
    workflow.add_node("chat",           node_chat)
    workflow.add_node("shutdown",       node_shutdown)
    workflow.add_node("execute_block",  node_execute_block)
    workflow.add_node("check_block",    node_check_block)
    workflow.add_node("replan",         node_replan)

    # ---- 入口 ----
    workflow.add_edge(START, "user_input")
    workflow.add_edge("user_input", "planner")

    # ---- Planner 输出路由 ----
    workflow.add_conditional_edges(
        "planner", route_task,
        {
            "chat":     "chat",
            "mission":  "execute_block",   # 直接进入第一个 block 的执行
            "shutdown": "shutdown"
        }
    )

    # ---- Block 执行循环（核心！） ----
    workflow.add_edge("execute_block", "check_block")

    workflow.add_conditional_edges(
        "check_block", route_after_check,
        {
            "continue":  "execute_block",  # 成功：继续下一个 block
            "replan":    "replan",          # 失败：去 replan
            "all_done":  "user_input"       # 全部完成：回用户输入
        }
    )

    # ---- Replan 后重新执行当前 block ----
    workflow.add_edge("replan", "execute_block")

    # ---- 闭环 ----
    workflow.add_edge("chat",     "user_input")
    workflow.add_edge("shutdown", "user_input")

    return workflow.compile(checkpointer=checkpointer)