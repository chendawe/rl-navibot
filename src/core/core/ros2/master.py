"""
ROS2 全局运行时环境（单例）。

职责：
1. 唯一拥有 rclpy.init() 和 rclpy.shutdown() 的权限。
2. 持有 MultiThreadedExecutor，统一调度所有业务 Node。
3. 后台静默 Spin，业务代码永远不需要关心事件循环。

所有业务 Node（Bridge / Streamer）必须通过 register_node() 入驻才能工作。
"""

import threading
import logging

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

logger = logging.getLogger(__name__)


class Ros2Runtime:
    """
    全局唯一的 ROS2 运行环境。

    生命周期：
        runtime = Ros2Runtime()          # 自动 init + 启动 executor 线程
        bridge = GazeboBridge(runtime)   # Bridge 入驻
        streamer = RGBStreamer(runtime)  # Streamer 入驻
        runtime.shutdown()               # 关闭时统一销毁
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if not rclpy.ok():
            rclpy.init()

        # MultiThreadedExecutor：自动为不同的 CallbackGroup 分配线程
        # Bridge 的控制回调在自己的线程里串行，Streamer 的编码回调在各自的线程里并发
        # 互不阻塞，物理级隔离
        self.executor = MultiThreadedExecutor()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self._initialized = True
        logger.info("✅ [Ros2Runtime] 全局环境已启动，等待 Node 租户入驻...")

    def _spin_loop(self):
        """后台守护线程，0.01s 一次 tick，处理所有 Node 的回调队列"""
        while rclpy.ok():
            self.executor.spin_once(timeout_sec=0.01)

    def register_node(self, node: Node):
        """
        业务 Node 入驻。
        入驻后该 Node 的所有订阅、发布、服务才真正生效。
        """
        self.executor.add_node(node)
        logger.info(f"🏠 [{node.get_name()}] 已入驻线程池")

    def shutdown(self):
        """统一关闭：销毁所有 Node + 终止 ROS2 上下文"""
        self.executor.shutdown()
        rclpy.shutdown()
        self._initialized = False
        logger.info("[Ros2Runtime] 已关闭")
