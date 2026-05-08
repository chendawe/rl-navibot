from langgraph.graph import StateGraph, END, START
from planning.brain.schemas.brain_schema import BrainState
from planning.brain.nodes.input_node import node_user_input
from planning.brain.nodes.planner_node import node_planner

# 👇 改成模块级导入，不把具体函数散落出来
import planning.brain.nodes.action_node as action_node

import logging
logger = logging.getLogger("buildBrainGraph")

def route_task(state: BrainState) -> str:
    return state.get("mission_type", "chat")

def route_after_check(state: BrainState) -> str:
    idx = state["current_block_idx"]
    total = len(state["mission_blocks"])

    if idx >= total:
        logger.info(f"[Route] 所有 {total} 个 block 完成 -> user_input")
        return "all_done"

    last_result = state["block_results"][-1]

    if last_result["status"] == "success":
        logger.info(f"[Route] Block[{idx-1}] 成功 -> 继续执行 Block[{idx}]")
        return "continue"

    # 判断是否达到熔断阈值
    retry_count = state.get("block_retry_counts", {}).get(idx, 0)
    if retry_count >= action_node.MAX_RETRIES:  # 👈 常量也跟着加前缀
        logger.error(f"[Route] Block[{idx}] 重试达 {retry_count} 次上限 -> abort_block")
        return "abort"

    logger.warning(f"[Route] Block[{idx}] 失败(重试 {retry_count}/{action_node.MAX_RETRIES}) -> replan")
    return "replan"


def build_brain_graph(checkpointer=None):
    workflow = StateGraph(BrainState)

    # ---- 注册所有节点 ----
    workflow.add_node("user_input",     node_user_input)
    workflow.add_node("planner",        node_planner)
    
    # 👇 全部加上 action_node. 前缀
    workflow.add_node("chat",           action_node.node_chat)
    workflow.add_node("shutdown",       action_node.node_shutdown)
    workflow.add_node("execute_block",  action_node.node_execute_block)
    workflow.add_node("check_block",    action_node.node_check_block)
    workflow.add_node("replan",         action_node.node_replan)
    workflow.add_node("abort_block",    action_node.node_abort_block)

    # ---- 入口 ----
    workflow.add_edge(START, "user_input")
    workflow.add_edge("user_input", "planner")

    # ---- Planner 输出路由 ----
    workflow.add_conditional_edges(
        "planner", route_task,
        {
            "chat":     "chat",
            "mission":  "execute_block",   
            "shutdown": "shutdown"
        }
    )

    # ---- Block 执行循环 ----
    workflow.add_edge("execute_block", "check_block")

    workflow.add_conditional_edges(
        "check_block", route_after_check,
        {
            "continue":  "execute_block",  
            "replan":    "replan",          
            "abort":     "abort_block",     
            "all_done":  "user_input"       
        }
    )

    # ---- Replan / Abort 后重新执行 ----
    workflow.add_edge("replan", "execute_block")
    workflow.add_edge("abort_block", "user_input")

    # ---- 闭环 ----
    workflow.add_edge("chat",     "user_input")
    workflow.add_edge("shutdown", "user_input")

    return workflow.compile(checkpointer=checkpointer)
