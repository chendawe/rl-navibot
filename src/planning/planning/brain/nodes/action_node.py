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
from planning.brain.schemas.brain_schema import BlockResult, BlockPlan

def node_execute_block(state: dict) -> dict:
    idx = state["current_block_idx"]
    blocks = state["mission_blocks"]

    if idx >= len(blocks):
        return {"response": f"全部任务完成！", "current_block_idx": idx}

    block = blocks[idx]
    logger.info(f"[ExecuteBlock] ▶ 准备执行 [{idx+1}/{len(blocks)}] {block}")

    executor_llm = get_executor_llm()
    logger.info(f"[ExecuteBlock] 调用底层 -> type={block['block_type']}, target={block['target']}")

    tool_result = executor_llm.invoke(
        messages=[],
        block_type=block["block_type"],
        target=block["target"],
    )

    # ✅ 用强类型 Schema 构建，而不是手动拼字典
    current_retry_count = state.get("block_retry_counts", {}).get(idx, 0)

    block_result = BlockResult(
        idx=idx,
        plan=BlockPlan(block_type=block["block_type"], target=block["target"]),
        status=tool_result["status"],
        detail=tool_result["detail"],
        retry_count=current_retry_count,
    )

    icon = {"success": "✅", "failed": "❌", "timeout": "⏱️"}.get(block_result.status, "❓")
    logger.info(f"[ExecuteBlock] {icon} Block[{idx}] 结果: {block_result.detail}")

    return {
        "block_results": state.get("block_results", []) + [block_result.model_dump()],
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

    # ---- 成功：游标 +1 ----
    if status == "success":
        logger.info(f"[CheckBlock] ✅ Block[{idx}] 成功，游标推进: {idx} -> {idx+1}")
        return {"current_block_idx": idx + 1}

    # ---- 失败或超时：只打日志，不做额外计数 ----
    block_key = f"{last_result['plan']['block_type']}:{last_result['plan']['target']}"
    fail_count = sum(
        1 for r in state["block_results"]
        if f"{r['plan']['block_type']}:{r['plan']['target']}" == block_key
        and r["status"] in ("failed", "timeout")
    )
    logger.warning(f"[CheckBlock] ❌ Block[{idx}] 状态[{status}]，目标[{block_key}]累计失败: {fail_count}/{MAX_RETRIES}")
    return {}



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
    goal = state.get("user_goal", "未知目标")

    logger.info(f"[Replan] 🔄 从 Block[{idx}] 开始重新规划（原方案失败: {error_detail}）")

    from planning.brain.clients.llm_client import get_mission_planner_llm
    planner = get_mission_planner_llm()

    # 构造一个“从失败中恢复”的提示
    replan_prompt = (
        f"【最终目标】: {goal}\n"
        f"【当前状况】: Block[{idx}] 执行失败，原因: {error_detail}\n"
        f"【已完成步骤】: {state['mission_blocks'][:idx]}\n"
        f"请重新规划从当前位置开始的后续步骤，只能使用真实存在的地点（如客厅、卧室、走廊），不要编造'备用入口-X'这种名字。"
    )

    new_tail_blocks = planner.plan(goal=replan_prompt)
    new_tail_dicts = [b.model_dump() for b in new_tail_blocks]

    head = state["mission_blocks"][:idx]
    updated_blocks = head + new_tail_dicts

    logger.info(f"[Replan] 📋 保留前 {idx} 个 | 替换后共 {len(updated_blocks)} 个")

    return {
        "mission_blocks": updated_blocks,
        "current_block_idx": idx,
        "response": f"Block[{idx}] 失败，已从断点重新规划后续 {len(new_tail_dicts)} 个步骤。",
    }




# ============================================================
# Abort Block (达到熔断阈值，强制跳过)
# ============================================================
def node_abort_block(state: dict) -> dict:
    idx = state["current_block_idx"]
    block = state["mission_blocks"][idx]
    total = len(state["mission_blocks"])

    logger.error(
        f"[AbortBlock] 🛑 Block[{idx}] 已达 {MAX_RETRIES} 次上限，"
        f"放弃剩余任务！类型={block['block_type']} 目标={block['target']}"
    )

    return {
        "current_block_idx": total,
        "block_retry_counts": {},
        "response": f"任务 {block['block_type']}({block['target']}) 多次失败已放弃，整个任务终止。",
    }

