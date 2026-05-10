from typing import TypedDict, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ==========================================
# 基础数据结构
# ==========================================
class BlockPlan(BaseModel):
    block_type: Literal["navi", "observe", "standby"] = Field(..., description="动作类型")
    # 🚨 删掉了 description，因为 MissionPlannerLLM 不生成它，强制必填会导致解析报错
    target: str = Field(..., description="去哪里 / 看什么")

class BrainParsedResult(BaseModel):
    mission_type: Literal["chat", "mission", "shutdown"] = Field(..., description="任务主类型")
    user_goal: Optional[str] = None  
    reasoning: str = Field(default="", description="简要分析过程")
    response: str = Field(default="", description="chat时的回复内容")
    mission_blocks: List[BlockPlan] = Field(default_factory=list, description="mission时的block列表")

# ==========================================
# 状态定义 (概念分层，物理扁平)
# ==========================================
# 🆕 强类型执行结果约束
class BlockResult(BaseModel):
    idx: int = Field(..., description="对应的 mission_blocks 索引")
    plan: BlockPlan = Field(..., description="执行的快照")  # 👈 完美复用！
    status: Literal["success", "failed", "timeout"] = Field(..., description="执行状态")
    detail: str = Field(default="", description="状态详情或失败原因")
    retry_count: int = Field(default=0, description="当前块已经重试的次数")


class BrainState(TypedDict):
    user_input: str
    mission_type: str
    response: str
    # ---- 任务专属层 ----
    user_goal: str        
    mission_blocks: List[BlockPlan]     
    current_block_idx: int              
    block_results: List[Dict[str, Any]] # 内部严格遵循 BlockResult 结构
    # ---- 系统控制层 ----
    is_running: bool

# {
#     "idx": 0,                      # 当前执行的是 mission_blocks 的哪个索引
#     "block_type": "navi",          # 动作类型
#     "target": "厨房",              # 目标
#     "status": "failed",            # 执行结果
#     "detail": "门口被挡住",        # 结果详情
#     "attempt": 2                   # 🔄 这是第几次尝试 (相当于之前的 retry_count + 1)
# }

    
# class BrainState(TypedDict):
#     user_input: str
#     mission_type: str
#     reasoning: str
#     response: str
#     # ---- 任务执行相关 ----
#     user_goal: str        # 🆕 比如："拿到水杯"（任务级，永不改变）
#     mission_blocks: list           # 规划出来的完整 block 列表
#     current_block_idx: int         # 当前执行到第几个 block（核心！）
#     block_results: list            # 每个 block 的执行结果记录
#     block_retry_counts: dict   # 🆕 记录每个 idx 的重试次数，如 {0: 2, 1: 1}
#     # ---- 系统控制 ----
#     is_running: bool
    
    
# BrainState，主进程和用户交互，路由LLM意图识别：
# -> BrainParsedResult：
# 如果chat那就chat，shutdown就shutdown，mission：
# -> MissionState
# => MissionPlannerLLM -> mission_blocks -> BlockPlan
# => ExecutorLLM -> BlockResult
