import logging
from planning.brain.clients.llm_client import get_executor_llm

logger = logging.getLogger("BrainNode.Action")

MAX_RETRIES = 3  # 硬件级熔断阈值

def node_chat(state: dict) -> dict:
    return {"response": state["response"], "is_running": True}

def node_shutdown(state: dict) -> dict:
    return {"response": "系统休眠。", "is_running": False}

# ============================================================
# 核心改造：把死代码，换成 Mission LLM 的调用
# ============================================================
def node_execute_block(state: dict) -> dict:
    idx = state["current_block_idx"]
    blocks = state["mission_blocks"]

    if idx >= len(blocks):
        return {"response": f"全部任务完成！", "current_block_idx": idx}

    block = blocks[idx]
    logger.info(f"[ExecuteBlock] ▶ 准备执行 [{idx+1}/{len(blocks)}] {block}")

    executor_llm = get_executor_llm()
    
    logger.info(f"[ExecuteBlock] 调用底层 -> type={block['block_type']}, target={block['target']}")

    # 传 block_type + target，让 LLM 路由到对应工具
    tool_result = executor_llm.invoke(
        messages=[],
        block_type=block["block_type"],
        target=block["target"],
    )

    # 拿到的是 {"status": "success"/"failed"/"timeout", "detail": "..."}
    block_result = {
        "idx": idx,
        "block": block,
        "status": tool_result["status"],       # 👈 旧代码写的是 llm_decision["success"]
        "detail": tool_result["detail"],
    }

    icon = {"success": "✅", "failed": "❌", "timeout": "⏱️"}.get(block_result["status"], "❓")
    logger.info(f"[ExecuteBlock] {icon} Block[{idx}] 结果: {block_result['detail']}")

    return {
        "block_results": state.get("block_results", []) + [block_result],
        "current_block_idx": idx,
    }


# ============================================================
# 检查 Block 结果 (累加重试次数)
# ============================================================
def node_check_block(state: dict) -> dict:
    idx = state["current_block_idx"]
    blocks = state["mission_blocks"]
    
    if idx >= len(blocks):
        return {"current_block_idx": idx}

    last_result = state["block_results"][-1]
    status = last_result["status"]

    # ---- 成功：清零该 block 的重试次数，游标 +1 ----
    if status == "success":
        logger.info(f"[CheckBlock] ✅ Block[{idx}] 成功，游标推进: {idx} -> {idx+1}")
        counts = state.get("block_retry_counts", {}).copy()
        counts.pop(idx, None) # 清除重试记录
        return {"current_block_idx": idx + 1, "block_retry_counts": counts}

    # ---- 失败或超时：累加重试次数 ----
    counts = state.get("block_retry_counts", {}).copy()
    current_retry = counts.get(idx, 0) + 1
    counts[idx] = current_retry
    
    logger.warning(f"[CheckBlock] ❌ Block[{idx}] 状态[{status}]，重试计数: {current_retry}/{MAX_RETRIES}")
    return {"block_retry_counts": counts}


# def node_replan(state: dict) -> dict:
#     prompt = f"""
#     用户最终目标：{state['user_goal']}
#     当前卡点：Block[{idx}] {state['block_results'][-1]['detail']}
#     请重新规划后续步骤。
#     """
#     # 直接把目标喂给 LLM，LLM 天然知道还要找水杯，根本不需要代码去“记”
#     new_tail = llm.invoke(prompt) 


def node_replan(state: dict) -> dict:
    idx = state["current_block_idx"]
    failed_result = state["block_results"][-1]
    error_detail = failed_result["detail"]

    logger.info(f"[Replan] 🔄 从 Block[{idx}] 开始重新规划（原方案失败: {error_detail}）")
    
    from planning.brain.schemas.brain_schema import BlockPlan
    # ✅ 直接拿任务级目标，不用管 block 级的历史怎么变的
    goal = state.get("user_goal", "未知目标")
    
    # Mock：根据任务级目标，硬编码生成绕行 + 重试
    new_tail_blocks = [
        BlockPlan(block_type="navi", description="Replan: 绕行到替代路线", target=f"备用入口-{idx}"),
        BlockPlan(block_type="observe", description=f"Replan: 重新尝试完成[{goal}]", target=goal),
    ]

    head = state["mission_blocks"][:idx]
    tail = [b.model_dump() for b in new_tail_blocks]

    updated_blocks = head + tail

    logger.info(f"[Replan] 📋 保留前 {idx} 个 | 替换后共 {len(updated_blocks)} 个 | 原始目标: {goal}")

    return {
        "mission_blocks": updated_blocks,
        "current_block_idx": idx,
        "response": f"Block[{idx}] 失败，已从断点重新规划后续 {len(new_tail_blocks)} 个步骤。",
    }



# ============================================================
# Abort Block (达到熔断阈值，强制跳过)
# ============================================================
def node_abort_block(state: dict) -> dict:
    idx = state["current_block_idx"]
    block = state["mission_blocks"][idx]
    total = len(state["mission_blocks"])

    logger.error(f"[AbortBlock] 🛑 Block[{idx}] 已达 {MAX_RETRIES} 次上限，放弃剩余任务！目标: {block['target']}")

    return {
        "current_block_idx": total,     # 👈 直接跳到末尾，结束任务
        "block_retry_counts": {},
        "response": f"任务 {block['description']} 多次失败已放弃，整个任务终止。",
    }
