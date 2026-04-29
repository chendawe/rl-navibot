
# class BlockStateEnum(str, Enum):
#     PENDING = "pending"
#     RUNNING = "running"
#     SUCCESS = "success"
#     FAILED = "failed"
#     TIMEOUT = "timeout"

# class MissionStateEnum(str, Enum):
#     PLANNING = "planning"
#     EXECUTING = "executing"
#     COMPLETED = "completed"
#     ABORTED = "aborted"

import logging
import random
import asyncio
from typing import TypedDict, Dict, Any, Literal
from pydantic import BaseModel

# ==========================================
# 0. 基础设施层
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Harness")

# ==========================================
# 1. Schema 层 (仅用于 Component 边界校验，不进 State)
# ==========================================
class NaviInput(BaseModel):
    target_position: str

class NaviOutput(BaseModel):
    actual_path: list[str]

class ObserveInput(BaseModel):
    target_object: str

class ObserveOutput(BaseModel):
    detected_objects: list[str]

# BlockPlan 保留，方便未来给 LLM 做 Output Structured 解析用
class BlockPlan(BaseModel):
    block_type: Literal["navi", "observe"]
    params: Dict[str, Any]

# ==========================================
# 2. Component 层 (纯执行器，不吃状态，不碰持久化)
# ==========================================
class NaviExecutor:
    async def execute(self, input_data: NaviInput) -> NaviOutput:
        logger.info(f"[NaviExecutor] 执行寻路到: {input_data.target_position}")
        await asyncio.sleep(1)
        if random.random() < 0.3:
            logger.error("[NaviExecutor] 底层报错：路径规划超时！")
            raise TimeoutError("A* Algorithm Timeout")
        return NaviOutput(actual_path=["start", input_data.target_position])

class ObserveExecutor:
    async def execute(self, input_data: ObserveInput) -> ObserveOutput:
        logger.info(f"[ObserveExecutor] 执行观测: {input_data.target_object}")
        await asyncio.sleep(1)
        return ObserveOutput(detected_objects=[input_data.target_object, "extra_box"])

# ==========================================
# 3. LangGraph 编排层 (全内置类型，对 msgpack 友好)
# ==========================================
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

class MissionManagerGraphState(TypedDict):
    mission_schedule: str
    block_plans: list[dict]       # ✅ 纯字典
    current_block_idx: int
    block_states: Dict[int, str]  # ✅ 存 "success"/"timeout" 字符串
    retry_counts: Dict[int, int]
    mission_status: str           # ✅ 存 "executing"/"completed" 字符串

async def execute_block_node(state: MissionManagerGraphState) -> dict:
    idx = state["current_block_idx"]
    
    if idx >= len(state["block_plans"]):
        return {"mission_status": "completed"}
        
    plan = state["block_plans"][idx]
    block_type = plan["block_type"]
    logger.info(f"--- 开始执行 Block [{idx}]: {block_type} ---")
    
    executors = {"navi": NaviExecutor(), "observe": ObserveExecutor()}
    executor = executors.get(block_type)
    
    new_state = {
        "block_states": state["block_states"].copy(),
        "retry_counts": state["retry_counts"].copy(),
    }
    
    try:
        # ✅ 在节点边界重建 Pydantic，做严格校验
        if block_type == "navi":
            await executor.execute(NaviInput(**plan["params"]))
        else:
            await executor.execute(ObserveInput(**plan["params"]))
            
        new_state["block_states"][idx] = "success"
        new_state["current_block_idx"] = idx + 1
    except TimeoutError:
        new_state["block_states"][idx] = "timeout"
        new_state["retry_counts"][idx] = new_state["retry_counts"].get(idx, 0) + 1
    except Exception as e:
        new_state["block_states"][idx] = "failed"
        new_state["retry_counts"][idx] = new_state["retry_counts"].get(idx, 0) + 1
        logger.error(f"Block [{idx}] 执行异常: {e}")
        
    return new_state

def route_after_execution(state: MissionManagerGraphState) -> str:
    if state["mission_status"] == "completed":
        return "end"
    
    idx = state["current_block_idx"]
    if idx >= len(state["block_plans"]):
        return "execute_block"
    
    current_status = state["block_states"].get(idx)
    
    if current_status is None or current_status == "success":
        return "execute_block"
        
    if current_status in ["failed", "timeout"]:
        retry_count = state["retry_counts"].get(idx, 0)
        if retry_count < 3:
            logger.warning(f"Block [{idx}] 失败，触发第 {retry_count + 1} 次重试")
            return "execute_block"
        else:
            logger.error(f"Block [{idx}] 重试达 3 次上限，任务中止！")
            return "abort_mission"
            
    return "end"

def abort_mission_node(state: MissionManagerGraphState) -> dict:
    logger.error("!!! 任务流程触发 Abort，系统锁定 !!!")
    return {"mission_status": "aborted"}

def build_mission_graph(checkpointer):
    workflow = StateGraph(MissionManagerGraphState)
    workflow.add_node("execute_block", execute_block_node)
    workflow.add_node("abort_mission", abort_mission_node)
    workflow.set_entry_point("execute_block")
    workflow.add_conditional_edges(
        "execute_block", route_after_execution,
        {"execute_block": "execute_block", "abort_mission": "abort_mission", "end": END}
    )
    workflow.add_edge("abort_mission", END)
    return workflow.compile(checkpointer=checkpointer)

# ==========================================
# 4. 运行测试
# ==========================================
async def main():
    # ✅ 纯 dict 初始化，没有 Pydantic，没有 Enum
    initial_state = {
        "mission_schedule": "用户指令：前往A点，观察箱子",
        "block_plans": [
            {"block_type": "navi", "params": {"target_position": "A点"}},
            {"block_type": "observe", "params": {"target_object": "箱子"}},
        ],
        "current_block_idx": 0,
        "block_states": {},
        "retry_counts": {},
        "mission_status": "executing",
    }

    config = {"configurable": {"thread_id": "mission_001"}}
    
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        app = build_mission_graph(checkpointer)
        
        logger.info("========== 启动任务引擎 ==========")
        async for event in app.astream(initial_state, config):
            pass
            
        final_state = await app.aget_state(config)
        logger.info(f"========== 任务结束: {final_state.values.get('mission_status')} ==========")

if __name__ == "__main__":
    asyncio.run(main())
