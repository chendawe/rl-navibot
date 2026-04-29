from pydantic import BaseModel, Field

class TaskPlan(BaseModel):
    # navi, 
    task_id: str = Field(..., description="任务唯一ID")
    task_type: str = Field(..., description="任务类型：navigation/observation/grasp")
    target: str = Field(..., description="目标节点/物体ID")
    steps: list[str] = Field(..., description="阶段规划点位序列")
