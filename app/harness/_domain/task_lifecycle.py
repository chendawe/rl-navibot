from schema import TaskPlan
from datetime import datetime, timedelta

class TaskLifecycleManager:
    def __init__(self, max_retries: int = 3, timeout_seconds: int = 30):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def handle_timeout(self, task: TaskPlan) -> TaskPlan:
        """处理任务超时，增加重试次数"""
        if task.retry_count < self.max_retries:
            task.retry_count += 1
            task.status = "created"  # 重置为待执行状态
        else:
            task.status = "failed"  # 超过重试次数，标记为失败
        return task
