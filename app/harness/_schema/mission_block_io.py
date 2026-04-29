# schemas/components.py
from pydantic import BaseModel, Field

class NavigationInput(BaseModel):
    target_position: str = Field(..., description="目标节点ID或坐标")
    timeout_seconds: int = Field(default=30, description="超时时间")

class NavigationOutput(BaseModel):
    status: str = Field(..., description="success/failed/timeout")
    path: list[str] = Field(default=[], description="实际路径节点序列")

class ObservationInput(BaseModel):
    target: str = Field(..., description="观测目标（如节点ID、物体类别）")
    sensor_type: str = Field(default="camera", description="传感器类型")

class ObservationOutput(BaseModel):
    detected_objects: list[str] = Field(default=[], description="识别的物体列表")
    confidence: float = Field(default=0.0, description="置信度")


class NaviInput(BaseModel):
    position: tuple

    
class NaviOutput(BaseModel):
    is_accomplished: bool
    is_failed: bool
