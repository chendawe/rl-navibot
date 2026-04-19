import time
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
# from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup 
from collections import deque

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
        
        # # 🔥 核心：向 Runtime 申请一个“允许并发执行”的通行证
        # # 有了这个组，哪怕 _process_msg 里的 cv2.imencode 卡了 20ms，
        # # 也不会阻塞 Runtime 处理其他 Streamer 或 RL Bridge 的消息
        # self.cg = ReentrantCallbackGroup()
        
        # 2. 这里：实例化改为互斥组
        # 效果：如果上一张图片的 cv2.imencode 还没处理完，新来的图片会在队列里等待
        # 配合 ROS 2 的 QoS depth=1，新图片甚至会直接覆盖旧图片，永远只处理最新帧
        self.cg = MutuallyExclusiveCallbackGroup()
        
        self._msg_times = deque(maxlen=30)  # 最近30帧到达时间戳
        self._last_latency = 0.0

    # ----------------------------------------
    # 计算fps和延迟用
    # ----------------------------------------
    def _record_frame(self, msg):
        """子类在 _process_msg 中调用，记录到达时间并计算延迟"""
        now = time.time()
        self._msg_times.append(now)
        # 计算延迟：若消息包含时间戳（如 sensor_msgs/Image 的 header.stamp），使用它
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._last_latency = (now - msg_time) * 1000.0  # 毫秒
        else:
            self._last_latency = 0.0

    def get_fps(self) -> float:
        """计算最近30帧的平均接收帧率"""
        if len(self._msg_times) < 2:
            return 0.0
        duration = self._msg_times[-1] - self._msg_times[0]
        if duration <= 0:
            return 0.0
        return (len(self._msg_times) - 1) / duration

    def get_latency_ms(self) -> float:
        return self._last_latency

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
