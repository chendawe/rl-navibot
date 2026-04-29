"""
main_graph.py — LangGraph 主状态图
"""
import json
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# 1. 状态定义
class MainState(TypedDict):
    user_input: str
    task_json: dict
    task_type: Literal["chat", "mission", "shutdown"]
    mission_blocks: list
    response: str
    is_running: bool


# 2. LLM System Prompt
SYSTEM_PROMPT = """你是一个机器人任务调度器。分析用户指令，输出结构化 JSON。

## 任务类型
- chat    : 闲聊、问答
- mission : 机器人执行任务，拆解为 mission_block
- shutdown: 用户要求关机/停止/退出

## Mission Block 类型
- navi    : 导航到某位置
- observe : 在某位置观察/检测

## 严格输出 JSON，不要输出其他内容
{
  "task_type": "chat | mission | shutdown",
  "reasoning": "简要分析",
  "response": "chat时直接写回复",
  "mission_blocks": [
    {"block_type": "navi|observe", "description": "做什么", "target": "去哪里/看什么"}
  ]
}"""


# 3. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 4. 节点
def node_user_input(state: MainState) -> dict:
    """interrupt() = 阻塞点，图挂起直到外部 resume"""
    user_msg = interrupt("等待用户指令...")
    return {"user_input": user_msg}


def node_llm_parse(state: MainState) -> dict:
    """LLM 解析为结构化任务 JSON"""
    result = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_input"]),
    ])

    content = result.content.strip()
    # 兼容 LLM 返回的 markdown 代码块格式
    backticks = chr(96) * 3
    if content.startswith(backticks):
        content = content.split("\n", 1)[1].rsplit(backticks, 1)[0].strip()

    task_json = json.loads(content)
    return {
        "task_json": task_json,
        "task_type": task_json["task_type"],
    }


def node_chat(state: MainState) -> dict:
    return {"response": state["task_json"].get("response", "收到。")}


def node_mission(state: MainState) -> dict:
    """任务执行占位节点，后续展开为 block 子图"""
    blocks = state["task_json"].get("mission_blocks", [])
    
    block_summary = ""
    for i, b in enumerate(blocks):
        block_summary += f"  Block {i+1}: [{b['block_type']}] {b['description']} -> {b['target']}\n"
        
    return {
        "mission_blocks": blocks,
        "response": f"任务已接收，共 {len(blocks)} 个 block:\n{block_summary}",
    }


def node_shutdown(state: MainState) -> dict:
    return {"response": "系统正在关机...", "is_running": False}


# 5. 路由
def route_task(state: MainState) -> str:
    return state.get("task_type", "chat")


# 6. 构建图
def build_graph(checkpointer=None):
    builder = StateGraph(MainState)

    builder.add_node("user_input", node_user_input)
    builder.add_node("llm_parse",  node_llm_parse)
    builder.add_node("chat",       node_chat)
    builder.add_node("mission",    node_mission)
    builder.add_node("shutdown",   node_shutdown)

    builder.add_edge(START, "user_input")
    builder.add_edge("user_input", "llm_parse")
    builder.add_conditional_edges(
        "llm_parse",
        route_task,
        {"chat": "chat", "mission": "mission", "shutdown": "shutdown"},
    )
    builder.add_edge("chat",     "user_input")
    builder.add_edge("mission",  "user_input")
    builder.add_edge("shutdown", END)

    return builder.compile(checkpointer=checkpointer)


# 7. 主循环
if __name__ == "__main__":
    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "session-001"}}

    print("机器人系统已启动")
    app.invoke(None, config=config)

    while True:
        current_state = app.get_state(config)
        if not current_state.values.get("is_running", True):
            print("\n系统已关机，再见！")
            break

        response = current_state.values.get("response", "")
        if response:
            print(f"\n机器人: {response}")

        user_msg = input("用户: ").strip()
        if not user_msg:
            continue

        app.invoke(Command(resume=user_msg), config=config)
