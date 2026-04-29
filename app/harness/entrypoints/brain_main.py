import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from app.harness.graph.brain_graph import build_brain_graph

# 配置日志以观察解耦效果
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S"
)

if __name__ == "__main__":
    app = build_brain_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-harness-001"}}

    print("🚀 Harness 架构可行性测试已启动 (无需联网，纯内存运转)")
    # app.invoke(None, config=config)
    # langgraph.errors.EmptyInputError: Received no input for __start__
    app.invoke({}, config=config)

    while True:
        current_state = app.get_state(config)
        
        if not current_state.values.get("is_running", True):
            print("系统已安全关机，主循环退出。")
            break

        response = current_state.values.get("response", "")
        # 观察点：看看这里拿到的 response 有没有被子图的脏数据污染
        if response:
            print(f"\n🤖 【最终输出】 {response}")

        user_msg = input("👤 你 (试试输入: 你好 / 去拿个杯子 / 关机): ").strip()
        if not user_msg:
            continue

        app.invoke(Command(resume=user_msg), config=config)
