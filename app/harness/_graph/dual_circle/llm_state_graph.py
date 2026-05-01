# app/harness/graph/dual_circle/llm/llm__state_graph.py

import json
import logging
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 0. Log 回转：配置标准 Logger
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BrainGraph")

# ==========================================
# 1. 严格格式化：用 Pydantic 强制约束 LLM 输出
# ==========================================
class BlockPlan(BaseModel):
    block_type: Literal["navi", "observe"] = Field(..., description="必须为 navi 或 observe")
    description: str = Field(..., description="执行什么动作")
    target: str = Field(..., description="去哪里 / 看什么")

class BrainParsedResult(BaseModel):
    mission_type: Literal["chat", "mission", "shutdown"] = Field(..., description="任务主类型")
    reasoning: str = Field(default="", description="简要分析过程")
    response: str = Field(default="", description="如果类型是 chat，直接写回复内容；否则为空")
    mission_blocks: list[BlockPlan] = Field(default_factory=list, description="如果类型是 mission，填写 block 列表；否则为空列表")

SYSTEM_PROMPT = """你是一个机器人任务调度器。严格分析用户指令。
1. 如果是闲聊/问答，mission_type 设为 "chat"，response 写上回复，mission_blocks 留空。
2. 如果是机器人执行任务，mission_type 设为 "mission"，response 留空，将任务拆解为 navi(导航) 和 observe(观察) 的 block 组合。
3. 如果用户明确要求关机/退出，mission_type 设为 "shutdown"。"""

# ==========================================
# 2. 状态定义：清理冗余，结构清晰
# ==========================================
class BrainState(TypedDict):
    user_input: str
    # 以下字段由 LLM 填充
    mission_type: str
    reasoning: str
    response: str
    mission_blocks: list
    # 系统控制
    is_running: bool

# ==========================================
# 3. LLM 实例
# ==========================================
# 结构化输出推荐用支持 tool calling 的模型 (如 gpt-4o-mini / qwen-plus 等)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(BrainParsedResult)

# ==========================================
# 4. 节点实现（内含兜底机制）
# ==========================================

def node_user_input(state: BrainState) -> dict:
    logger.info("系统挂起，等待用户输入...")
    user_msg = interrupt("等待用户指令...")
    logger.info(f"收到用户输入: {user_msg}")
    return {"user_input": user_msg}


def node_planner(state: BrainState) -> dict:
    """LLM 解析 + 极致兜底：无论 LLM 怎么抽风，图都不能崩"""
    logger.info(f"[Planner] 开始解析: {state['user_input']}")
    
    # ----- 兜底逻辑开始 -----
    try:
        # with_structured_output 保证 99% 情况返回合法对象
        parsed: BrainParsedResult = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["user_input"])
        ])
        result_dict = parsed.model_dump()
        logger.info(f"[Planner] 解析成功 -> 类型: {result_dict['mission_type']}, Blocks数量: {len(result_dict['mission_blocks'])}")
        return result_dict

    except Exception as e:
        # 极端情况：网络错误、模型拒绝回答、Pydantic 校验彻底失败
        logger.error(f"[Planner] ⚠️ LLM解析异常: {e}，启动兜底机制！")
        return {
            "mission_type": "chat",
            "reasoning": "parser_failed",
            "response": "抱歉，我刚才大脑短路了一下，请再说一次。",
            "mission_blocks": []
        }
    # ----- 兜底逻辑结束 -----


def node_chat(state: BrainState) -> dict:
    logger.info(f"[Chat] 直接回复用户")
    return {"response": state["response"], "is_running": True}


def node_mission(state: BrainState) -> dict:
    logger.info(f"[Mission] 接收到任务，共 {len(state['mission_blocks'])} 个 block")
    # TODO: 对接你的 mission_executor 子图
    # from app.harness.graph.dual_circle.llm.mission_executor import build_mission_graph
    # mission_app = build_mission_graph(...)
    # ...
    
    block_summary = "\n".join([f"  - [{b['block_type']}] {b['description']} -> {b['target']}" for b in state["mission_blocks"]])
    return {"response": f"任务开始执行:\n{block_summary}", "is_running": True}


def node_shutdown(state: BrainState) -> dict:
    logger.warning("[Shutdown] 收到关机指令，系统即将终止")
    return {"response": "系统正在关机...", "is_running": False}


# ==========================================
# 5. 路由逻辑
# ==========================================
def route_task(state: BrainState) -> str:
    return state.get("mission_type", "chat")

# ==========================================
# 6. 构建图
# ==========================================
def build_brain_graph(checkpointer=None):
    workflow = StateGraph(BrainState)

    # 注册节点
    workflow.add_node("user_input", node_user_input)
    workflow.add_node("planner",    node_planner)
    workflow.add_node("chat",       node_chat)
    workflow.add_node("mission",    node_mission)
    workflow.add_node("shutdown",   node_shutdown)

    # 连边
    workflow.add_edge(START, "user_input")
    workflow.add_edge("user_input", "planner")
    
    # 条件路由
    workflow.add_conditional_edges(
        "planner",
        route_task,
        {
            "chat":     "chat",
            "mission":  "mission",
            "shutdown": "shutdown"
        }
    )
    
    # 闭环与终结
    workflow.add_edge("chat",     "user_input")  # 聊完继续等
    workflow.add_edge("mission",  "user_input")  # 任务完继续等
    workflow.add_edge("shutdown", END)           # 关机结束

    return workflow.compile(checkpointer=checkpointer)

# ==========================================
# 7. 本地测试主循环
# ==========================================
if __name__ == "__main__":
    app = build_brain_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-001"}}

    logger.info("🚀 机器人系统已启动")
    app.invoke(None, config=config)

    while True:
        current_state = app.get_state(config)
        
        # 判断是否关机
        if not current_state.values.get("is_running", True):
            logger.info("系统已安全关机，主循环退出。")
            break

        # 打印回复
        response = current_state.values.get("response", "")
        if response:
            print(f"\n🤖 {response}")

        user_msg = input("👤 你: ").strip()
        if not user_msg:
            continue

        app.invoke(Command(resume=user_msg), config=config)



"""
====================================================================================================
🔥 HARNESS 架构拆解指南 (解耦规范)
====================================================================================================

当前的单文件属于 MVP 阶段的 "上帝文件"（God Object），随着业务增加（比如接入真实 ROS2 导航、
多模态视觉、更复杂的 block 重试），必须按职责拆解。

以下是基于 Harness 思想的严格拆解映射表，请按此结构将代码迁移到对应目录：

📁 app/harness/graph/dual_circle/
│
├── 📁 schemas/                     # 【纯数据结构层】绝对不依赖任何业务逻辑或外部库
│   └── brain_schema.py
│       ├── class BlockPlan(BaseModel)      # 从这里搬
│       ├── class BrainParsedResult(...)    # 从这里搬
│       └── class BrainState(TypedDict)     # 从这里搬
│
├── 📁 prompts/                     # 【纯文本规范层】管理所有 Prompt，方便做 A/B 测试和版本控制
│   └── brain_prompts.py
│       └── SYSTEM_PROMPT = "..."           # 从这里搬
│
├── 📁 clients/                     # 【外部通信层】封装所有与大模型、API 的网络交互
│   └── llm_client.py
│       └── def get_structured_llm():       # 从这里搬 (返回 structured_llm 实例)
│
├── 📁 nodes/                       # 【纯行为逻辑层】每个文件是一个节点，只做状态计算，不关心图怎么连
│   ├── input_node.py
│   │   └── def node_user_input(...)       # 从这里搬
│   │
│   ├── planner_node.py
│   │   └── def node_planner(...)          # 从这里搬 (包含 try-except 兜底)
│   │
│   └── action_nodes.py
│       ├── def node_chat(...)             # 从这里搬
│       ├── def node_mission(...)          # 从这里搬 (内部调用 mission 子图的入口)
│       └── def node_shutdown(...)         # 从这里搬
│
├── 📁 subgraphs/                   # 【子图层】复杂的独立状态机 (如 mission 执行链)
│   └── mission_executor.py
│       ├── class MissionState(...)        # 子图自己的状态
│       └── def build_mission_graph(...)   # 子图的组装逻辑
│
├── 📁 graph/                       # 【拓扑组装层】只负责 "把节点连起来"，不写任何业务代码
│   └── brain_graph.py
│       ├── def route_task(...)            # 从这里搬
│       └── def build_brain_graph(...)     # 从这里搬 (只剩下 add_node 和 add_edge)
│
└── 📁 entrypoints/                 # 【驱动层】系统的启动入口，负责实例化图和跑 while 循环
    └── cli_main.py
        └── if __name__ == "__main__":     # 从这里搬 (MemorySaver 实例化、invoke、while True)

====================================================================================================
💡 核心心法（拆解原则）：
1. schemas/ 里的类：可以跨节点、跨子图复用，改了这里全系统报警。
2. nodes/ 里的函数：输入是 State，输出是 dict，实现类似纯函数，方便写单元测试。
3. graph/ 里：绝对不放 if/else 业务逻辑，只放 add_conditional_edges 的路由映射。
4. clients/ 里：如果以后把 GPT 换成本地 Qwen，只改这一个文件，nodes 和 graph 完全不动。
====================================================================================================
"""
