import logging
from app.harness.subgraphs.mission_executor import build_mission_subgraph

logger = logging.getLogger("BrainNode.Action")

def node_chat(state: dict) -> dict:
    logger.info("[Chat] 直接回复")
    return {"response": state["response"], "is_running": True}

# def node_shutdown(state: dict) -> dict:
#     logger.warning("[Shutdown] 系统终止")
#     return {"response": "系统正在关机...", "is_running": False}

def node_shutdown(state: dict) -> dict:
    logger.warning("[Shutdown] 系统进入休眠挂起")
    return {
        "response": "系统已休眠，输入任意内容唤醒。",
        "is_running": True   # 不退出了，保持挂起
    }

def node_mission(state: dict) -> dict:
    """Wrapper 节点：状态翻译官，彻底隔离父子图"""
    logger.info(f"[Mission] 宏观调度：收到 {len(state['mission_blocks'])} 个 Block")
    
    # 1. 初始化子图
    subgraph = build_mission_subgraph()
    
    # 2. 状态翻译：父状态 -> 子状态（过滤掉所有无关数据）
    blocks_str = "\n".join([f"- {b['block_type']}: {b['target']}" for b in state["mission_blocks"]])
    micro_init_state = {
        "task_desc": blocks_str,
        "internal_scratchpad": "",
        "hardware_error_code": 0,
        "final_mock_result": ""
    }
    
    # 3. 隔离执行：子图内部怎么折腾都不影响外界
    final_micro_state = subgraph.invoke(micro_init_state)
    
    # 4. 出参提取：只把父图关心的结论拿出来，垃圾数据全部丢弃
    result = final_micro_state["final_mock_result"]
    logger.info(f"[Mission] 子图执行完毕，提取结论: {result}。丢弃了子图内部的脏数据。")
    
    return {
        "response": f"任务已下发并执行完毕，最终状态: {result}",
        "mission_result": result,  # 存入父图状态，供未来 Replan 路由使用
        "is_running": True
    }
