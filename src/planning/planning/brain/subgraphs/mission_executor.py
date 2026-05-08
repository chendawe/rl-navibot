# # 如果用封装，只需一行，看起来很爽
# micro_agent = create_react_agent(llm, tools=[ros_navi_tool])

# def node_mission(state):
#     # 黑盒调用
#     result = micro_agent.invoke({"messages": [HumanMessage(content="去导航")]})
#     # 痛点：result 里面是一大坨 MessagesState 垃圾，你很难干净地提取出 "success" 
#     # 如果它内部死循环了，你在这里没有任何手段可以强杀它（没有熔断计数器的控制点）

# 官方模板通常强制要求你的 Plan 是一段纯文本或者特定的 Plan 对象。
# 但在你的架构里，你的 Plan 是结构极其严谨的 list[BlockPlan]
# （包含 block_type, target 等强类型字段）。如果套用官方模板，
# 要么你被迫改掉你精心设计的 BrainState 去迁就它，
# 要么你需要写极其恶心的适配器代码。

import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END, START

logger = logging.getLogger("MissionSubGraph")

# ==========================================
# 子图独有状态（绝对污染不到父图）
# ==========================================
class MissionMicroState(TypedDict):
    task_desc: str
    internal_scratchpad: str  # 脏数据：内部思考过程
    hardware_error_code: int # 脏数据：模拟硬件错误
    final_mock_result: str

# ==========================================
# 子图内部死代码节点（模拟 ReAct 循环）
# ==========================================
def micro_step_1_think(state: MissionMicroState) -> dict:
    logger.info(f"  [子图-思考] 准备执行: {state['task_desc']}")
    return {"internal_scratchpad": "正在思考怎么走...", "hardware_error_code": 0}

def micro_step_2_act(state: MissionMicroState) -> dict:
    logger.info(f"  [子图-行动] 调用底盘ROS接口... 模拟遇到小障碍... 重试成功")
    return {
        "internal_scratchpad": state["internal_scratchpad"] + "\n执行成功，虽然有点颠簸。",
        "hardware_error_code": 404 # 模拟的脏数据
    }

def micro_step_3_finish(state: MissionMicroState) -> dict:
    logger.info(f"  [子图-结束] 任务闭环")
    return {"final_mock_result": "success"}

# ==========================================
# 子图组装
# ==========================================
def build_mission_subgraph():
    workflow = StateGraph(MissionMicroState)
    workflow.add_node("think", micro_step_1_think)
    workflow.add_node("act", micro_step_2_act)
    workflow.add_node("finish", micro_step_3_finish)
    
    workflow.add_edge(START, "think")
    workflow.add_edge("think", "act")
    workflow.add_edge("act", "finish")
    workflow.add_edge("finish", END)
    
    return workflow.compile()
