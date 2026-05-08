import logging
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
import random
from planning.brain.tools import ros2_tools

# 注意：这里不 import ChatOpenAI，用纯 Python 模拟结构化输出，确保没网也能跑

logger = logging.getLogger("MockLLM")


# ==========================================
# 1. 宏观大脑 LLM (Mock)
# ==========================================
class MockPlannerLLM:
    def invoke(self, messages):
        user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        logger.info(f"[PlannerLLM] 解析意图: {user_msg}")
        from planning.brain.schemas.brain_schema import BrainParsedResult, BlockPlan
        if "关机" in user_msg: return BrainParsedResult(mission_type="shutdown")
        if "去" in user_msg or "拿" in user_msg:
            return BrainParsedResult(
                mission_type="mission",
                user_goal="拿到水杯",  # 👈 【加在这里】未来换成 LLM 自动提取
                mission_blocks=[
                    BlockPlan(block_type="navi", description="导航", target="桌子"),
                    BlockPlan(block_type="observe", description="观察", target="水杯")
            ])
        return BrainParsedResult(mission_type="chat", response=f"宏观回复：{user_msg}")

# ============================================================
# 微观小脑 LLM (模拟真实工具的三种返回)
# ============================================================
class MockExecutorLLM:
    def __init__(self):
        # 2. 注册工具
        self.tools = {
            "navi": ros2_tools.ros_navigate,
            "observe": ros2_tools.ros_observe,
        }

    def invoke(self, messages, block_type=None, target=None):
        logger.info(f"[ExecutorLLM] 📦 调用底层工具 -> type={block_type}, target={target}")
        
        tool = self.tools.get(block_type)
        if not tool:
            return {"status": "failed", "detail": f"未知工具类型: {block_type}"}
            
        # 3. 真正调用工具，拿到返回值
        result = tool.invoke(target)
        
        # 4. 根据 status 打不同级别的 log
        status = result["status"]
        if status == "success":
            logger.info(f"[ExecutorLLM] ✅ 工具返回成功")
        elif status == "failed":
            logger.warning(f"[ExecutorLLM] ❌ 工具返回失败 -> {result['detail']}")
        else:
            logger.error(f"[ExecutorLLM] ⏱️ 工具返回超时 -> {result['detail']}")
            
        return result

# ==========================================
# 3. 工厂函数
# ==========================================
def get_planner_llm():
    return MockPlannerLLM()

def get_executor_llm():
    # 注意：如果是真实环境，这里可以注入不同的 model_name 或 temperature
    return MockExecutorLLM()