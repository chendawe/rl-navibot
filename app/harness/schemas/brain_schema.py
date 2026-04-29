from typing import TypedDict, Literal
from pydantic import BaseModel, Field

class BlockPlan(BaseModel):
    block_type: Literal["navi", "observe"] = Field(..., description="必须为 navi 或 observe")
    description: str = Field(..., description="执行什么动作")
    target: str = Field(..., description="去哪里 / 看什么")

class BrainParsedResult(BaseModel):
    mission_type: Literal["chat", "mission", "shutdown"] = Field(..., description="任务主类型")
    reasoning: str = Field(default="", description="简要分析过程")
    response: str = Field(default="", description="如果类型是 chat，直接写回复内容；否则为空")
    mission_blocks: list[BlockPlan] = Field(default_factory=list, description="如果类型是 mission，填写 block 列表；否则为空列表")

class BrainState(TypedDict):
    user_input: str
    mission_type: str
    reasoning: str
    response: str
    mission_blocks: list
    mission_result: str  # 新增：用于接收子图执行后的解耦结果
    is_running: bool
