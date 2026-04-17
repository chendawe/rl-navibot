"""
所有仿真 Bridge 的抽象基类。

封装与具体仿真引擎无关的通用通信基础设施：
- 线程安全的数据存取（_data_store + _lock）
- 发布器注册与缓存
- 用完即焚的动态服务调用
- 节点发现

设计原则：
1. 不持有 ROS2 生命周期（init / shutdown 由 Ros2Runtime 统一管理）。
2. 对外只暴露 Python 原生类型 和 Numpy 数组，坚决不暴露 ROS 消息体。
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile


class BaseBridge(Node, ABC):
    """
    仿真通信防腐层抽象基类。

    子类必须实现 setup() 方法来完成特定仿真引擎的初始化。
    """

    def __init__(self, node_name: str, runtime):
        """
        Args:
            node_name: ROS2 节点名
            runtime: Ros2Runtime 实例，本节点将入驻其线程池
        """
        # 绝不调 rclpy.init()，由 runtime 统一管理
        super().__init__(node_name=node_name)

        # 向运行时注册自身（此时 ROS2 回调才开始生效）
        runtime.register_node(self)

        # ---- 通用数据存储 ----
        # 所有 Topic 的最新消息快照，key = topic 名
        self._lock = threading.Lock()
        self._data_store: Dict[str, Any] = {}

        # ---- 语义化映射 ----
        # 允许外部用 "laser" / "imu" 等业务名代替 "/scan" 等 ROS Topic 字符串
        self._topic_map: Dict[str, str] = {}

    # ================================================================
    #  一、底层通信基础设施（子类可直接使用，也可完全覆盖）
    # ================================================================

    def _create_subscriber(self, topic: str, msg_type,
                           preprocess_cb: Optional[Callable] = None,
                           qos: Optional[QoSProfile] = None):
        """
        注册订阅，支持可选的预处理回调。
        回调内自动加锁写入 _data_store，保证线程安全。
        """
        def _internal_cb(msg):
            with self._lock:
                if preprocess_cb:
                    preprocess_cb(msg)
                self._data_store[topic] = msg

        actual_qos = qos if qos is not None else QoSProfile(depth=10)
        self.create_subscription(msg_type, topic, _internal_cb, actual_qos)

    def _get_data(self, topic: str) -> Optional[Any]:
        """线程安全地获取原始 ROS 消息快照"""
        with self._lock:
            return self._data_store.get(topic)

    def _create_publisher(self, topic: str, msg_type, qos_depth: int = 10):
        """注册发布器并缓存引用（供 _publish 使用）"""
        pub = self.create_publisher(msg_type, topic, qos_depth)
        with self._lock:
            self._data_store[f'pub_{topic}'] = pub
        return pub

    def _publish(self, topic: str, msg):
        """
        线程安全的发布。
        ROS2 的 pub.publish() 本身是线程安全的，
        这里只需保证获取 pub 对象时的原子性，不需要把 publish 动作也锁住。
        """
        with self._lock:
            pub = self._data_store.get(f'pub_{topic}')
        if pub:
            pub.publish(msg)

    def _generic_cb(self, topic: str) -> Callable:
        """内部工具：生成一个带锁的通用回调闭包"""
        def callback(msg):
            with self._lock:
                self._data_store[topic] = msg
        return callback

    # ================================================================
    #  二、系统级工具方法（节点发现、动态服务调用）
    # ================================================================

    def _find_node(self, name_fragment: str, timeout_sec: float = 5.0) -> Optional[str]:
        """在 ROS2 网络中按名称片段查找节点"""
        start = time.time()
        while time.time() - start < timeout_sec:
            nodes = self.get_node_names()
            for n in nodes:
                if name_fragment in n:
                    return n
            time.sleep(0.1)
        return None

    def _call_service(self, service_name: str, srv_type,
                      request, timeout_sec: float = 5.0):
        """
        用完即焚的动态服务调用。
        适用场景：偶尔调用的服务（如获取 URDF）。
        对于高频调用（如 reset_world），请在子类中创建长连接客户端。
        """
        client = self.create_client(srv_type, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.destroy_client(client)
            raise RuntimeError(f"Service '{service_name}' 未在 {timeout_sec}s 内上线!")

        future = client.call_async(request)
        start = time.time()
        while not future.done() and time.time() - start < timeout_sec:
            time.sleep(0.05)

        self.destroy_client(client)

        if future.done():
            if future.exception() is not None:
                raise RuntimeError(
                    f"调用服务 '{service_name}' 异常: {future.exception()}"
                )
            return future.result()
        raise TimeoutError(f"调用服务 '{service_name}' 超时!")

    # ================================================================
    #  三、子类必须实现的模板方法
    # ================================================================

    @abstractmethod
    def setup(self, **kwargs):
        """
        子类核心：注册传感器、执行器、仿真服务。
        由上层业务代码（如 Gym Env 的 __init__）在合适时机调用。
        """
        pass
