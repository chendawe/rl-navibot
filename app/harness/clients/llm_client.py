import logging
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
# 注意：这里不 import ChatOpenAI，用纯 Python 模拟结构化输出，确保没网也能跑

logger = logging.getLogger("MockLLM")

# class MockStructuredLLM:
#     """纯内存假 LLM，根据输入死代码返回 Pydantic 对象，用于可行性测试"""
#     def invoke(self, messages: List[BaseMessage]):
#         user_msg = "未知"
#         for m in reversed(messages):
#             if isinstance(m, HumanMessage):
#                 user_msg = m.content
#                 break
        
#         logger.info(f"[MockLLM] 接收到解析请求: {user_msg}")
        
#         # 简单的规则匹配模拟 LLM 拆解
#         if "关机" in user_msg or "退出" in user_msg:
#             from app.harness.schemas.brain_schema import BrainParsedResult
#             return BrainParsedResult(mission_type="shutdown", reasoning="用户要关机")
#         elif "任务" in user_msg or "去" in user_msg or "拿" in user_msg:
#             from app.harness.schemas.brain_schema import BrainParsedResult, BlockPlan
#             return BrainParsedResult(
#                 mission_type="mission",
#                 reasoning="检测到物理任务",
#                 mission_blocks=[
#                     BlockPlan(block_type="navi", description="导航", target="目标点A"),
#                     BlockPlan(block_type="observe", description="观察", target="目标物体")
#                 ]
#             )
#         else:
#             from app.harness.schemas.brain_schema import BrainParsedResult
#             return BrainParsedResult(mission_type="chat", response=f"这是针对'{user_msg}'的模拟回复。")

# def get_structured_llm():
#     """工厂函数：未来换成真实 LLM 只改这里"""
#     # 真实环境: return ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(BrainParsedResult)
#     return MockStructuredLLM()


# ==========================================
# 1. 宏观大脑 LLM (Mock)
# ==========================================
class MockPlannerLLM:
    def invoke(self, messages):
        user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        logger.info(f"[PlannerLLM] 解析意图: {user_msg}")
        from app.harness.schemas.brain_schema import BrainParsedResult, BlockPlan
        if "关机" in user_msg: return BrainParsedResult(mission_type="shutdown")
        if "去" in user_msg or "拿" in user_msg:
            return BrainParsedResult(mission_type="mission", mission_blocks=[
                BlockPlan(block_type="navi", description="导航", target="桌子"),
                BlockPlan(block_type="observe", description="观察", target="水杯")
            ])
        return BrainParsedResult(mission_type="chat", response=f"宏观回复：{user_msg}")

# ==========================================
# 2. 微观小脑 LLM (Mock) - 核心：它是跟着循环走的单例
# ==========================================
class MockExecutorLLM:
    def invoke(self, messages):
        # 在真实环境中，这里接收的是当前 block 指令 + 上一轮的工具返回结果
        # 它负责决定：是调用工具？还是报错？
        logger.info("[ExecutorLLM] 正在决策当前 Block 的动作...")
        import random
        success = random.random() > 0.3
        # 真实环境这里应该返回 ToolCall，我们用简单的 dict 模拟它的输出
        return {"action": "call_tool", "success": success, "detail": "执行完毕" if success else "轮子卡住了"}

# ==========================================
# 3. 工厂函数
# ==========================================
def get_planner_llm():
    return MockPlannerLLM()

def get_executor_llm():
    # 注意：如果是真实环境，这里可以注入不同的 model_name 或 temperature
    return MockExecutorLLM()