from ...schema.llm_goals import TaskPlan
from ...domain.task_lifecycle import TaskLifecycleManager
import datetime

class TaskManager:
    def __init__(self):
        self.lifecycle_manager = TaskLifecycleManager()

    def create_task(self, user_command: str) -> TaskPlan:
        """通过LLM解析任务，创建TaskPlan对象"""
        # 调用LLM识别任务类型、目标等，返回TaskPlan
        task_plan = TaskPlan(
            task_id=f"task_{datetime.now().timestamp()}",
            task_type="navigation",
            target="shelf_A1",
            steps=["A1", "B2", "C3"]
        )
        return task_plan

    def update_task(self, task: TaskPlan, feedback: str):
        """根据ROS2反馈更新任务状态"""
        if "到达" in feedback:
            task.status = "success"
        else:
            task = self.lifecycle_manager.handle_timeout(task)
        return task
