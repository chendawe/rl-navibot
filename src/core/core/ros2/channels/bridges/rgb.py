# /home/chendawww/workspace/rl-navibot/src/core/core/ros2/channels/bridges/rgb.py
from typing import Optional
from core.ros2.channels.streamers.rgb import RGBStreamer
from core.ros2.master import Ros2Runtime

class RGBBridge:
    """RGB 相机 Bridge，提供图像读取 API"""
    def __init__(self, runtime, topic: str = '/camera/image_raw/compressed'):
        self.streamer = RGBStreamer(runtime, topic)
        runtime.register_node(self.streamer)
    
    def get_latest_frame(self) -> Optional[bytes]:
        """获取最新的 JPEG 帧"""
        return self.streamer.get_latest_frame()
