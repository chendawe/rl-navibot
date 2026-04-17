import time
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

class BaseStreamer(Node, ABC):
    """
    Web 流媒体节点的抽象基类。
    
    架构升级点：
    1. 彻底废弃内部 _spin_loop 线程！转为入驻 Ros2Runtime 的多线程池。
    2. 使用 ReentrantCallbackGroup 声明自身为“可重入/可并发”节点。
       Runtime 会自动把图像处理调度到独立的线程，绝不卡死 RL 的互斥组。
    3. 保留极其优秀的 Web 端 FPS 限流机制。
    """
    def __init__(self, node_name: str, runtime, default_fps: int = 15):
        # 绝不调 rclpy.init()，向 Runtime 借用生命
        super().__init__(node_name=node_name)
        runtime.register_node(self)  # 入驻线程池
        
        self._default_fps = default_fps
        self._frame_interval = 1.0 / default_fps
        
        # 存储最新处理好的二进制帧 (Python GIL 保证了赋值的原子性，无需加锁)
        self._latest_frame: Optional[bytes] = None
        self._last_send_time: float = 0.0
        
        # 🔥 核心：向 Runtime 申请一个“允许并发执行”的通行证
        # 有了这个组，哪怕 _process_msg 里的 cv2.imencode 卡了 20ms，
        # 也不会阻塞 Runtime 处理其他 Streamer 或 RL Bridge 的消息
        self.cg = ReentrantCallbackGroup()

    # ----------------------------------------
    # 对外暴露的统一接口 (供 FastAPI 调用)
    # ----------------------------------------
    def get_frame(self) -> Optional[bytes]:
        """
        带限流的帧获取器。
        返回：None 表示还没到下一帧的时间；bytes 表示可直接通过 WebSocket 发送的裸二进制。
        """
        now = time.time()
        if now - self._last_send_time < self._frame_interval:
            return None
            
        if self._latest_frame:
            self._last_send_time = now
            return self._latest_frame
        return None

    # ----------------------------------------
    # 强制子类实现的模板方法
    # ----------------------------------------
    @abstractmethod
    def _process_msg(self, msg):
        """
        子类核心：在这里写轻量的预处理逻辑！
        约束：必须将 ROS msg 转换为 bytes，并赋值给 self._latest_frame。
        """
        pass
