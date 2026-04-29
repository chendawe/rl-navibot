from langgraph.graph import StateGraph, END, START
from app.harness.schemas.brain_schema import BrainState
from app.harness.nodes.input_node import node_user_input
from app.harness.nodes.planner_node import node_planner
from app.harness.nodes.action_node import node_chat, node_mission, node_shutdown

def route_task(state: BrainState) -> str:
    return state.get("mission_type", "chat")

def build_brain_graph(checkpointer=None):
    workflow = StateGraph(BrainState)

    workflow.add_node("user_input", node_user_input)
    workflow.add_node("planner",    node_planner)
    workflow.add_node("chat",       node_chat)
    workflow.add_node("mission",    node_mission)
    workflow.add_node("shutdown",   node_shutdown)

    workflow.add_edge(START, "user_input")
    workflow.add_edge("user_input", "planner")
    
    workflow.add_conditional_edges(
        "planner", route_task,
        {"chat": "chat", "mission": "mission", "shutdown": "shutdown"}
    )
    
    workflow.add_edge("chat",     "user_input")
    workflow.add_edge("mission",  "user_input")
    # workflow.add_edge("shutdown", END)
    workflow.add_edge("shutdown", "user_input")

    return workflow.compile(checkpointer=checkpointer)
