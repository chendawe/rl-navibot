import rclpy
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile
from typing import Any, Optional, Callable

import logging


logger = logging.getLogger(__name__)

class Ros2Bridge:
    """
    统一的 ROS2 通信桥。
    解决痛点：全局只能 init 一次、多线程安全读写、后台静默 spin。
    """
    _instance = None
    DEFAULT_NODE_NAME = "ros2_bridge_node" 
    
    def __new__(cls, node_name: str = "web_backend_node"):
        # 单例模式：保证整个 Python 进程只有一个 rclpy.init()
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, node_name: str = "web_backend_node"):
        if self._initialized:
            # ====== 🔥 采用你的建议：保留参数，打印警告 ======
            requested_name = node_name or self._DEFAULT_NODE_NAME
            if requested_name != self._actual_node_name:
                logger.warning(
                    f"[Ros2Bridge] 单例已存在！请求创建的节点名 '{requested_name}' 被忽略，"
                    f"当前实际使用的节点名为 '{self._actual_node_name}'。"
                )
            return
        
        if not rclpy.ok():
            rclpy.init()
            
        self._actual_node_name = node_name or self._DEFAULT_NODE_NAME
        self.node = Node(self._actual_node_name)
        self._lock = threading.Lock()
        self._data_store = {}  # 核心：所有话题的最新数据都存在这里
        
        # 启动后台守护线程（和你 Gym 里写的一模一样）
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        
        self._initialized = True
        print(f"✅ Ros2Bridge 初始化完成，节点名: {node_name}")

    def _spin_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    # 在 ros2_bridge.py 里修改这个方法：
    def create_subscriber(self, topic: str, msg_type, preprocess_cb: Optional[Callable] = None, qos=None):
        def _internal_cb(msg):
            with self._lock:
                if preprocess_cb:
                    preprocess_cb(msg)
                self._data_store[topic] = msg
                
        # ★ 改这里：如果传了 qos 就用 qos，没传就用默认的 depth=10
        actual_qos = qos if qos is not None else QoSProfile(depth=10)
        self.node.create_subscription(msg_type, topic, _internal_cb, actual_qos)

    def get_data(self, topic: str) -> Optional[Any]:
        """线程安全地获取最新一帧数据"""
        with self._lock:
            return self._data_store.get(topic)

    def create_publisher(self, topic: str, msg_type, qos_depth: int = 10):
        """暴露 publisher 接口备用（比如以后 Web 端要发 /goal_pose）"""
        pub = self.node.create_publisher(msg_type, topic, qos_depth)
        self._data_store[f'pub_{topic}'] = pub
        return pub
        
    def publish(self, topic: str, msg):
        """线程安全的发布"""
        with self._lock:
            pub = self._data_store.get(f'pub_{topic}')
        if pub:
            pub.publish(msg)
