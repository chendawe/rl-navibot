import math
import time
import threading
from typing import Any, Optional, Callable, Dict
import logging

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ================================================================
# 🔥 优化点 1：将所有 ROS2 消息类型统一提至顶部导入
# 原因：原代码在函数内部 (如 send_velocity, reset_world) 频繁 import。
# 在高频调用的 RL step 循环中，虽然 Python 有缓存，但顶层导入能彻底消除字典查找开销。
# ================================================================
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity

import numpy as np

logger = logging.getLogger(__name__)


class GazeboRos2Bridge:
    """
    ROS2/Gazebo 通信防腐层。
    
    核心设计哲学：
    1. 单例模式：保证整个 Python 进程 (包含 Web 端、Gym 端) 只有一个 rclpy.init() 和底层 Node。
    2. 后台静默 Spin：将 ROS2 的异步事件循环藏进守护线程，Gym 的 step() 绝不被阻塞。
    3. 数据降维：对外只暴露 Python 原生类型 和 Numpy 数组，坚决不暴露 ROS 消息体。
    """
    _instance = None
    DEFAULT_NODE_NAME = "gazebo_ros2_bridge_node"

    def __new__(cls, node_name: str = DEFAULT_NODE_NAME):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, node_name: str = DEFAULT_NODE_NAME):
        if self._initialized:
            requested_name = node_name or self._DEFAULT_NODE_NAME
            if requested_name != self._actual_node_name:
                logger.warning(
                    f"[GazeboRos2Bridge] 单例已存在！请求节点名 '{requested_name}' 被忽略，"
                    f"当前实际使用: '{self._actual_node_name}'。"
                )
            return

        if not rclpy.ok():
            rclpy.init()

        self._actual_node_name = node_name or self._DEFAULT_NODE_NAME
        self.node = Node(self._actual_node_name)
        
        # 数据总线：所有 topic 的最新快照
        self._lock = threading.Lock()
        self._data_store: Dict[str, Any] = {}
        
        # Topic 名称映射表 (供后续通过语义调用，如 get_laser_ranges())
        self._topic_map: Dict[str, str] = {}

        # 后台线程：持续为 ROS2 的回调队列泵血
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self._initialized = True
        logger.info(f"✅ GazeboRos2Bridge 初始化完成，节点名: {self._actual_node_name}")

    def _spin_loop(self):
        """后台守护线程，0.01s 一次 tick，处理所有 incoming 的消息和 service 响应"""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    # ================================================================
    #  一、底层通信基础设施 (底层 API，尽量只供 Bridge 内部使用)
    # ================================================================

    def create_subscriber(self, topic: str, msg_type, preprocess_cb: Optional[Callable] = None, qos=None):
        def _internal_cb(msg):
            with self._lock:
                if preprocess_cb:
                    preprocess_cb(msg)
                self._data_store[topic] = msg

        actual_qos = qos if qos is not None else QoSProfile(depth=10)
        self.node.create_subscription(msg_type, topic, _internal_cb, actual_qos)

    def get_data(self, topic: str) -> Optional[Any]:
        """线程安全地获取原始 ROS 消息快照 (仅供特殊场景使用，常规请使用下面的 get_xxx 系列方法)"""
        with self._lock:
            return self._data_store.get(topic)

    def create_publisher(self, topic: str, msg_type, qos_depth: int = 10):
        pub = self.node.create_publisher(msg_type, topic, qos_depth)
        with self._lock:
            self._data_store[f'pub_{topic}'] = pub
        return pub

    def publish(self, topic: str, msg):
        """线程安全的发布"""
        # 🔥 优化点 2：缩小锁粒度。ROS2 的 pub.publish() 本身是线程安全的，
        # 我们只需要保证获取 pub 对象时的原子性，不需要把 publish 动作也锁住。
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
    #  二、系统级工具方法 (节点发现、动态服务调用)
    # ================================================================

    def find_node(self, name_fragment: str, timeout_sec: float = 5.0) -> Optional[str]:
        start = time.time()
        while time.time() - start < timeout_sec:
            nodes = self.node.get_node_names()
            for n in nodes:
                if name_fragment in n:
                    return n
            time.sleep(0.1)
        return None

    def call_service(self, service_name: str, srv_type, request, timeout_sec: float = 5.0):
        """
        用完即焚的动态服务调用。
        适用场景：偶尔调用的服务 (如获取 URDF)。
        对于高频调用 (如 reset_world)，请使用 setup_sim_services 创建的长连接。
        """
        client = self.node.create_client(srv_type, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.node.destroy_client(client)
            raise RuntimeError(f"Service '{service_name}' 未在 {timeout_sec}s 内上线!")

        future = client.call_async(request)
        start = time.time()
        while not future.done() and time.time() - start < timeout_sec:
            time.sleep(0.05)

        self.node.destroy_client(client)

        if future.done():
            if future.exception() is not None:
                raise RuntimeError(f"调用服务 '{service_name}' 发生异常: {future.exception()}")
            return future.result()
        raise TimeoutError(f"调用服务 '{service_name}' 超时!")

    def get_remote_parameter(self, node_fragment: str, param_name: str, timeout_sec: float = 10.0) -> Optional[str]:
        """高阶封装：从目标节点获取指定参数 (专用于拉取 robot_description)"""
        target_node = self.find_node(node_fragment, timeout_sec=timeout_sec / 2)
        if not target_node:
            raise RuntimeError(f"未找到包含 '{node_fragment}' 的节点!")

        service_name = f"{target_node}/get_parameters"
        req = GetParameters.Request()
        req.names = [param_name]

        result = self.call_service(service_name, GetParameters, req, timeout_sec=timeout_sec / 2)

        if result.values and hasattr(result.values[0], 'string_value'):
            return result.values[0].string_value
        return None

    # ================================================================
    #  三、通信初始化组 (供 Gym Env 的 __init__ 调用)
    # ================================================================

    def setup_sensors(self, laser_topic: str, imu_topic: str, odom_topic: str, laser_noise_threshold: float = 0.0):
        """一站式注册传感器。将严格的 QoS 和底层去噪逻辑彻底封装"""
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

        def _laser_cb(msg: LaserScan):
            if laser_noise_threshold > 0:
                for i in range(len(msg.ranges)):
                    if msg.ranges[i] <= laser_noise_threshold:
                        msg.ranges[i] = msg.range_max
            with self._lock:
                self._data_store[laser_topic] = msg

        self.node.create_subscription(LaserScan, laser_topic, _laser_cb, qos)
        self.node.create_subscription(Imu, imu_topic, self._generic_cb(imu_topic), qos)
        self.node.create_subscription(Odometry, odom_topic, self._generic_cb(odom_topic), qos)

        self._topic_map.update({"laser": laser_topic, "imu": imu_topic, "odom": odom_topic})

    def setup_actuators(self, cmd_vel_topic: str, goal_topic: str):
        self.create_publisher(cmd_vel_topic, Twist, 10)
        self.create_publisher(goal_topic, PoseStamped, 10)
        self._topic_map.update({"cmd_vel": cmd_vel_topic, "goal": goal_topic})

    def setup_sim_services(self, reset_world: str, set_entity: str, delete_entity: str, spawn_entity: str, entity_name: str):
        """创建长期存活的 Gazebo 服务客户端 (避免高频 reset 时反复创建/销毁的开销)"""
        self._sim_clients = {
            "reset_world": self.node.create_client(Empty, reset_world),
            "set_entity":  self.node.create_client(SetEntityState, set_entity),
            "delete":      self.node.create_client(DeleteEntity, delete_entity),
            "spawn":       self.node.create_client(SpawnEntity, spawn_entity),
        }
        self._entity_name = entity_name

    # ================================================================
    #  四、数据读取 API (纯粹的数据出口，返回 Numpy / Dict)
    # ================================================================

    def get_laser_ranges(self) -> Optional[np.ndarray]:
        topic = self._topic_map.get("laser")
        if not topic: return None
        msg = self.get_data(topic)
        return np.array(msg.ranges, dtype=np.float32) if msg else None

    def get_odom_data(self) -> Dict[str, float]:
        topic = self._topic_map.get("odom")
        if not topic or not self.get_data(topic):
            return {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0}
        
        msg = self.get_data(topic)
        p, o, t = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        
        siny_cosp = 2 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
        
        return {"x": p.x, "y": p.y, "yaw": math.atan2(siny_cosp, cosy_cosp),
                "vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z}

    def get_imu_data(self) -> Dict[str, list]:
        topic = self._topic_map.get("imu")
        if not topic or not self.get_data(topic):
            return {"acc": [0.0, 0.0, 0.0], "gyro": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0]}
            
        msg = self.get_data(topic)
        a, g, o = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        
        sinr_cosp = 2 * (o.w * o.x + o.y * o.z)
        cosr_cosp = 1 - 2 * (o.x * o.x + o.y * o.y)
        sinp = 2 * (o.w * o.y - o.z * o.x)

        return {
            "acc": [a.x, a.y, a.z],
            "gyro": [g.x, g.y, g.z],
            "rpy": [math.atan2(sinr_cosp, cosr_cosp), math.asin(np.clip(sinp, -1.0, 1.0))]
        }

    # ================================================================
    #  五、执行与仿真控制 API
    # ================================================================

    def send_velocity(self, lin_x: float, ang_z: float):
        topic = self._topic_map.get("cmd_vel")
        if not topic: return
        msg = Twist()
        msg.linear.x, msg.angular.z = float(lin_x), float(ang_z)
        self.publish(topic, msg)

    def wait_for_sim_services(self, timeout_sec: float = 5.0):
        if not hasattr(self, '_sim_clients'):
            raise RuntimeError("仿真服务未初始化！请先调用 setup_sim_services()")
        
        # 🔥 优化点 3：修复超时逻辑。
        # 以前是 if not client.wait_for_service(timeout_sec=0.5)，这会导致总超时变成 N * 0.5s。
        start = time.time()
        for key, client in self._sim_clients.items():
            while time.time() - start < timeout_sec:
                if client.service_is_ready():
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError(f"仿真服务 '{key}' ({client.srv_name}) 在 {timeout_sec}s 内未上线!")

    def reset_world(self, timeout_sec: float = 5.0):
        future = self._sim_clients["reset_world"].call_async(Empty.Request())
        start = time.time()
        while not future.done() and time.time() - start < timeout_sec:
            time.sleep(0.05)
        if not future.done(): raise TimeoutError(f"reset_world 超时 ({timeout_sec}s)!")
        if future.exception(): raise RuntimeError(f"reset_world 失败: {future.exception()}")

    def set_robot_pose(self, x: float, y: float, yaw: float = 0.0, frame: str = "world"):
        req = SetEntityState.Request()
        req.state.name = self._entity_name
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = float(x), float(y), 0.0
        req.state.pose.orientation.z, req.state.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        req.state.reference_frame = frame

        future = self._sim_clients["set_entity"].call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < 3.0:
            time.sleep(0.05)
        if not future.done(): raise TimeoutError("set_robot_pose 超时 (3s)!")
        if future.exception(): raise RuntimeError(f"set_robot_pose 失败: {future.exception()}")
