import logging
from app.harness.clients.llm_client import get_executor_llm
from langchain_core.messages import HumanMessage

logger = logging.getLogger("BrainNode.Action")

def node_chat(state: dict) -> dict:
    return {"response": state["response"], "is_running": True}

def node_shutdown(state: dict) -> dict:
    return {"response": "系统休眠。", "is_running": True}

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

    # 【关键点】：拿到单例的 Executor LLM
    executor_llm = get_executor_llm()
    
    # 【关键点】：构造只属于当前 Block 的微上下文（不包含宏观闲聊！）
    # 真实环境：这里会带上上一个 Block 的工具返回值（如果有的话）
    micro_prompt = f"当前任务: {block['description']}，目标: {block['target']}。请执行。"
    
    # 让微观 LLM 去做决策并调用底层工具
    llm_decision = executor_llm.invoke([HumanMessage(content=micro_prompt)])

    # 根据 LLM 的决策记录结果
    block_result = {
        "idx": idx,
        "block": block,
        "status": "success" if llm_decision["success"] else "failed",
        "detail": llm_decision["detail"]
    }

    icon = "✅" if block_result["status"] == "success" else "❌"
    logger.info(f"[ExecuteBlock] {icon} LLM决策结果: {block_result['detail']}")

    return {
        "block_results": state.get("block_results", []) + [block_result],
        "current_block_idx": idx, # 游标不动，交给 check_block
    }

def node_check_block(state: dict) -> dict:
    idx = state["current_block_idx"]
    if idx >= len(state["mission_blocks"]): return {"current_block_idx": idx}
    
    if state["block_results"][-1]["status"] == "success":
        return {"current_block_idx": idx + 1} # 成功，游标推进一步
    return {"current_block_idx": idx}         # 失败，游标保持，等 replan

def node_replan(state: dict) -> dict:
    idx = state["current_block_idx"]
    logger.info(f"[Replan] 🔄 Block[{idx}] 失败，尝试原地重规划")
    
    # 真实环境：这里也可以调用 Executor LLM，让它根据报错信息生成替代方案
    # 现在用 Mock 替代
    from app.harness.schemas.brain_schema import BlockPlan
    new_block = BlockPlan(block_type="navi", description=f"Replan-绕行", target=f"备用路线-{idx}")
    
    updated_blocks = state["mission_blocks"].copy()
    updated_blocks[idx] = new_block.model_dump()
    
    return {"mission_blocks": updated_blocks, "response": f"Block[{idx}] 失败，微观层已替换方案。"}


# import logging
# from app.harness.subgraphs.mission_executor import build_mission_subgraph

# logger = logging.getLogger("BrainNode.Action")

# def node_chat(state: dict) -> dict:
#     logger.info("[Chat] 直接回复")
#     return {"response": state["response"], "is_running": True}

# # def node_shutdown(state: dict) -> dict:
# #     logger.warning("[Shutdown] 系统终止")
# #     return {"response": "系统正在关机...", "is_running": False}

# def node_shutdown(state: dict) -> dict:
#     logger.warning("[Shutdown] 系统进入休眠挂起")
#     return {
#         "response": "系统已休眠，输入任意内容唤醒。",
#         "is_running": True   # 不退出了，保持挂起
#     }

# # def node_mission(state: dict) -> dict:
# #     """Wrapper 节点：状态翻译官，彻底隔离父子图"""
# #     logger.info(f"[Mission] 宏观调度：收到 {len(state['mission_blocks'])} 个 Block")
    
# #     # 1. 初始化子图
# #     subgraph = build_mission_subgraph()
    
# #     # 2. 状态翻译：父状态 -> 子状态（过滤掉所有无关数据）
# #     blocks_str = "\n".join([f"- {b['block_type']}: {b['target']}" for b in state["mission_blocks"]])
# #     micro_init_state = {
# #         "task_desc": blocks_str,
# #         "internal_scratchpad": "",
# #         "hardware_error_code": 0,
# #         "final_mock_result": ""
# #     }
    
# #     # 3. 隔离执行：子图内部怎么折腾都不影响外界
# #     final_micro_state = subgraph.invoke(micro_init_state)
    
# #     # 4. 出参提取：只把父图关心的结论拿出来，垃圾数据全部丢弃
# #     result = final_micro_state["final_mock_result"]
# #     logger.info(f"[Mission] 子图执行完毕，提取结论: {result}。丢弃了子图内部的脏数据。")
    
# #     return {
# #         "response": f"任务已下发并执行完毕，最终状态: {result}",
# #         "mission_result": result,  # 存入父图状态，供未来 Replan 路由使用
# #         "is_running": True
# #     }

# # ============================================================
# # 核心：一次只执行一个 block 的节点
# # ============================================================
# def node_execute_block(state: dict) -> dict:
#     idx = state["current_block_idx"]
#     blocks = state["mission_blocks"]

#     # 安全兜底：所有 block 跑完了
#     if idx >= len(blocks):
#         logger.info(f"[ExecuteBlock] 所有 {len(blocks)} 个 block 已执行完毕")
#         return {
#             "response": f"全部任务完成！共执行 {len(blocks)} 个步骤。",
#             "current_block_idx": idx,
#         }

#     block = blocks[idx]
#     logger.info(f"[ExecuteBlock] ▶ 正在执行 [{idx+1}/{len(blocks)}] {block['block_type']}: {block['description']} -> {block['target']}")

#     # ============ 真实环境：这里换成 ROS2 调用 ============
#     # 例如：
#     # if block["block_type"] == "navi":
#     #     result = ros_navi_client.send_goal(block["target"])
#     # elif block["block_type"] == "observe":
#     #     result = vision_client.detect(block["target"])
#     # ========================================================

#     # Mock：模拟执行结果（70% 成功率，随机失败用来测试 replan）
#     import random
#     success = random.random() > 0.3

#     if success:
#         block_result = {
#             "idx": idx,
#             "block": block,
#             "status": "success",
#             "detail": f"成功完成 {block['description']}"
#         }
#         logger.info(f"[ExecuteBlock] ✅ Block[{idx}] 成功")
#     else:
#         block_result = {
#             "idx": idx,
#             "block": block,
#             "status": "failed",
#             "detail": f"执行失败：{block['target']} 不可达（模拟）"
#         }
#         logger.warning(f"[ExecuteBlock] ❌ Block[{idx}] 失败 -> 即将触发 Replan")

#     # 累加结果到 block_results（手动 merge，因为默认是覆盖）
#     updated_results = state.get("block_results", []) + [block_result]

#     return {
#         "block_results": updated_results,
#         "current_block_idx": idx,  # idx 不变，等 check_block 来决定要不要 +1
#     }


# # ============================================================
# # 核心：检查当前 block 结果，决定继续还是 replan
# # ============================================================
# def node_check_block(state: dict) -> dict:
#     idx = state["current_block_idx"]
#     blocks = state["mission_blocks"]

#     # 所有 block 跑完，直接结束
#     if idx >= len(blocks):
#         return {"current_block_idx": idx}

#     last_result = state["block_results"][-1]

#     if last_result["status"] == "success":
#         # 成功：idx + 1，下一个 block
#         new_idx = idx + 1
#         logger.info(f"[CheckBlock] ✅ 通过，推进到 Block[{new_idx}]")
#         return {"current_block_idx": new_idx}
#     else:
#         # 失败：idx 不动，触发 replan
#         logger.info(f"[CheckBlock] ❌ Block[{idx}] 失败，保持 idx={idx}，等待 Replan")
#         return {"current_block_idx": idx}


# # ============================================================
# # 核心：Replan 节点（只针对失败的 block 重新规划）
# # ============================================================
# def node_replan(state: dict) -> dict:
#     idx = state["current_block_idx"]
#     failed_block = state["block_results"][-1]
#     error_detail = failed_block["detail"]

#     logger.info(f"[Replan] 🔄 Block[{idx}] 执行失败: {error_detail}")
#     logger.info(f"[Replan] 📋 已完成的 blocks: {[r['status'] for r in state['block_results']]}")

#     # ============ 真实环境：调用 LLM 重新规划 ============
#     # 你可以给 LLM 看失败的上下文，让它生成替代方案
#     # parsed = structured_llm.invoke([
#     #     SystemMessage(content=f"原任务 Block[{idx}] 失败: {error_detail}，请生成替代 block"),
#     #     HumanMessage(content=f"已完成的记录: {state['block_results']}")
#     # ])
#     # new_block = parsed.mission_blocks[0]
#     # ========================================================

#     # Mock：生成一个替代 block
#     from app.harness.schemas.brain_schema import BlockPlan
#     new_block = BlockPlan(
#         block_type="navi",
#         description="Replan: 换条路走",
#         target=f"替代路径-{failed_block['block']['target']}"
#     )
#     logger.info(f"[Replan] 🆕 新方案: {new_block.description} -> {new_block.target}")

#     # 关键操作：把失败的 block 替换掉，后面的 block 保持不变
#     updated_blocks = state["mission_blocks"].copy()
#     updated_blocks[idx] = new_block.model_dump()

#     return {
#         "mission_blocks": updated_blocks,
#         "response": f"Block[{idx}] 失败，已重新规划替代方案。",
#     }