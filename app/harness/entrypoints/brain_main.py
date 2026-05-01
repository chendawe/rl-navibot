import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from app.harness.graph.brain_graph import build_brain_graph

# 配置日志以观察解耦效果
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S"
    # 1. 输出到文件（不打印控制台）
    # filename="run.log",
    # 2. 写入模式 a=追加 w=覆盖
    # filemode="a",
    # 3. 编码（解决中文乱码！超级重要）
    # encoding="utf-8",
)

# if __name__ == "__main__":
#     app = build_brain_graph(checkpointer=MemorySaver())
#     config = {"configurable": {"thread_id": "test-harness-001"}}

#     print("🚀 Harness 架构可行性测试已启动 (无需联网，纯内存运转)")
#     # app.invoke(None, config=config)
#     # langgraph.errors.EmptyInputError: Received no input for __start__
#     app.invoke({}, config=config)

#     while True:
#         current_state = app.get_state(config)
        
#         if not current_state.values.get("is_running", True):
#             print("系统已安全关机，主循环退出。")
#             break

#         response = current_state.values.get("response", "")
#         # 观察点：看看这里拿到的 response 有没有被子图的脏数据污染
#         if response:
#             print(f"\n🤖 【最终输出】 {response}")

#         user_msg = input("👤 你 (试试输入: 你好 / 去拿个杯子 / 关机): ").strip()
#         if not user_msg:
#             continue

#         app.invoke(Command(resume=user_msg), config=config)

if __name__ == "__main__":
    app = build_brain_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-replan-001"}}

    print("🚀 Replan 架构可行性测试")
    app.invoke({}, config=config)

    while True:
        current_state = app.get_state(config)

        response = current_state.values.get("response", "")
        if response:
            print(f"\n🤖 {response}")

        # 打印当前 block 执行进度（新增）
        results = current_state.values.get("block_results", [])
        if results:
            print("📋 执行记录:")
            for r in results:
                icon = "✅" if r["status"] == "success" else "❌"
                print(f"   {icon} Block[{r['idx']}] {r['block']['description']}: {r['detail']}")

        user_msg = input("\n👤 你 (试试: 去厨房拿水 / 关机): ").strip()
        if not user_msg:
            continue

        app.invoke(Command(resume=user_msg), config=config)