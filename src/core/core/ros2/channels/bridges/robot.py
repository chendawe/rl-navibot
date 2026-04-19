"""
通用 ROS2 机器人通信桥（仿真无关）。

职责：
- 传感器数据订阅与最新状态缓存 (快照模式)
- 执行器指令发布
- 数据降维：对外只暴露 Python 原生类型 和 Numpy 数组

适用场景：Gazebo 仿真、Isaac Sim 仿真、真机部署。
只要 ROS2 Topic 名和消息格式一致，此类无需任何修改。
"""

import math
import logging
from typing import Dict, Optional

import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

# ==============================================================
# 🧪 旧版依赖：仅作为注释存档，证明我们曾经用过 ATS
# import message_filters
# from message_filters import ApproximateTimeSynchronizer
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# ==============================================================

from core.ros2.channels.bridges.base import BaseBridge

logger = logging.getLogger(__name__)


class RobotBridge(BaseBridge):
    """
    仿真无关的机器人通信桥。

    🔥 架构升级：从 ATS(时间戳对齐) 降级为 LFS(最新快照)。
    原因：RL 的 0.1s 离散步长根本不在乎传感器间几毫秒的物理延迟，
         强行对齐反而容易在 Gazebo 掉帧时导致 RL step() 假死。
    现在的行为：各传感器独立缓存，到点直接拿，0 等待，0 报错。
    """

    def __init__(self, runtime, node_name: str = "robot_bridge"):
        super().__init__(node_name=node_name, runtime=runtime)

        self._laser_noise_threshold = 0.0

        # RL 安全兜底数据：开机第一帧尚未收到数据时使用
        self._fallback_data = {
            "laser": np.zeros(360, dtype=np.float32),
            "odom": {"x": 0.0, "y": 0.0, "yaw": 0.0,
                     "vx": 0.0, "vy": 0.0, "wz": 0.0},
            "imu": {"acc": [0.0, 0.0, 0.0],
                    "gyro": [0.0, 0.0, 0.0],
                    "rpy": [0.0, 0.0]}
        }
        
        # 🔥 核心变更：从单一同步字典，拆分为三个独立缓存
        # 这样即使激光丢了，里程计依然能读到最新值，不会互相拖后腿
        self._latest_laser: Optional[LaserScan] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None

    # ================================================================
    #  一、模板方法实现
    # ================================================================

    def setup(self, *,
              laser_topic: str, imu_topic: str, odom_topic: str,
              cmd_vel_topic: str, goal_topic: str,
              laser_noise_threshold: float = 0.0):
        """一站式初始化传感器 + 执行器"""
        self._laser_noise_threshold = laser_noise_threshold
        self._setup_sensors(laser_topic, imu_topic, odom_topic)
        self._setup_actuators(cmd_vel_topic, goal_topic)

    # ================================================================
    #  二、传感器（内部）
    # ================================================================

    def _setup_sensors(self, laser_topic: str, imu_topic: str, odom_topic: str):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==============================================================
        # 🧪 旧版 ATS 方案（已注释）：
        # 通过时间戳强行捆绑三个传感器，slop=0.05s (50ms)。
        # 缺点：Gazebo 稍微卡顿一下，就凑不齐包，导致 RL 拿不到数据。
        # ==============================================================
        # self._sensor_cg = MutuallyExclusiveCallbackGroup()
        # laser_sub = message_filters.Subscriber(self, LaserScan, laser_topic, qos_profile=qos, callback_group=self._sensor_cg)
        # imu_sub = message_filters.Subscriber(self, Imu, imu_topic, qos_profile=qos, callback_group=self._sensor_cg)
        # odom_sub = message_filters.Subscriber(self, Odometry, odom_topic, qos_profile=qos, callback_group=self._sensor_cg)
        # ats = ApproximateTimeSynchronizer([laser_sub, imu_sub, odom_sub], queue_size=10, slop=0.05)
        # ats.registerCallback(self._ats_sync_callback)
        # ==============================================================


        # 🔥 新版 LFS 方案（当前生效）：
        # 各自独立订阅，谁来了谁更新自己的坑位，互不干涉。
        self.create_subscription(LaserScan, laser_topic, self._laser_cb, qos)
        self.create_subscription(Imu, imu_topic, self._imu_cb, qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, qos)

        self._topic_map.update({
            "laser": laser_topic, "imu": imu_topic, "odom": odom_topic
        })
        logger.info("[RobotBridge] 传感器已切换为 LFS (Latest-First-Snapshot) 模式")

    # --- 独立回调函数：只管覆盖自己那份缓存，极简极快 ---
    
    def _laser_cb(self, msg: LaserScan):
        """激光回调：过滤近距离噪点后直接覆盖"""
        if self._laser_noise_threshold > 0:
            # 使用列表推导比 for 循环快一点点
            msg.ranges = [r if r > self._laser_noise_threshold else msg.range_max for r in msg.ranges]
        with self._lock:
            self._latest_laser = msg

    def _imu_cb(self, msg: Imu):
        """IMU 回调：无处理，直接覆盖"""
        with self._lock:
            self._latest_imu = msg

    def _odom_cb(self, msg: Odometry):
        """里程计回调：无处理，直接覆盖"""
        with self._lock:
            self._latest_odom = msg

    # 🧪 旧版 ATS 联合回调（已注释存档）
    # def _ats_sync_callback(self, laser_msg: LaserScan, imu_msg: Imu, odom_msg: Odometry):
    #     """物理时间戳对齐后触发"""
    #     if self._laser_noise_threshold > 0:
    #         for i in range(len(laser_msg.ranges)):
    #             if laser_msg.ranges[i] <= self._laser_noise_threshold:
    #                 laser_msg.ranges[i] = laser_msg.range_max
    #     with self._lock:
    #         self._latest_synced_data = {"laser": laser_msg, "imu": imu_msg, "odom": odom_msg}

    # ================================================================
    #  三、执行器（内部）
    # ================================================================

    def _setup_actuators(self, cmd_vel_topic: str, goal_topic: str):
        self._create_publisher(cmd_vel_topic, Twist, 10)
        self._create_publisher(goal_topic, PoseStamped, 10)
        self._topic_map.update({"cmd_vel": cmd_vel_topic, "goal": goal_topic})

    # ================================================================
    #  四、数据读取 API
    # ================================================================

    def _safe_get_latest(self, key: str):
        """
        底层读取器。
        无论外部怎么调，这里永远只做一件事：打开锁 -> 读对应坑位 -> 没有就给兜底。
        """
        with self._lock:
            if key == "laser": data = self._latest_laser
            elif key == "imu": data = self._latest_imu
            elif key == "odom": data = self._latest_odom
            else: data = None
            
        if data is not None:
            return data
        return self._fallback_data.get(key)

    def get_laser_ranges(self) -> np.ndarray:
        return np.array(self._safe_get_latest("laser").ranges, dtype=np.float32)

    def get_odom_data(self) -> Dict[str, float]:
        msg = self._safe_get_latest("odom")
        p, o, t = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        siny_cosp = 2 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
        return {
            "x": p.x, "y": p.y, "yaw": math.atan2(siny_cosp, cosy_cosp),
            "vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z
        }

    def get_imu_data(self) -> Dict[str, list]:
        msg = self._safe_get_latest("imu")
        a, g, o = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        sinr_cosp = 2 * (o.w * o.x + o.y * o.z)
        cosr_cosp = 1 - 2 * (o.x * o.x + o.y * o.y)
        sinp = 2 * (o.w * o.y - o.z * o.x)
        return {
            "acc": [a.x, a.y, a.z],
            "gyro": [g.x, g.y, g.z],
            "rpy": [
                math.atan2(sinr_cosp, cosr_cosp),
                math.asin(np.clip(sinp, -1.0, 1.0))
            ]
        }

    # ================================================================
    #  五、执行 API
    # ================================================================

    def send_velocity(self, lin_x: float, ang_z: float):
        topic = self._topic_map.get("cmd_vel")
        if not topic:
            return
        msg = Twist()
        msg.linear.x, msg.angular.z = float(lin_x), float(ang_z)
        self._publish(topic, msg)

    # ==================== 新增：RL 语义化便利方法 ====================
    # 注意：以下方法的内部逻辑完全复制自 Env 中的同名/同功能代码段，仅将数据源替换为 self

    def get_laser_normalized(self, beams: int, max_range: float) -> np.ndarray:
        """
        获取降维、归一化后的激光雷达状态。
        这是 RL 环境中最容易炸裂的地方（NaN 传播、维度不匹配），这里的防御逻辑是保命符。
        （逻辑与 Env._get_laser_data() 完全一致）
        """
        ranges = self.get_laser_ranges()
        # 1. 防御性兜底：如果仿真器第一帧还没来得及发数据，用满量程填充（代表周围绝对安全）
        if ranges is None:
            return np.ones(beams, dtype=np.float32)
            
        # ======= 2. 数据清洗 (必做！Gazebo 的 Ray 传感器偶尔会抽风) =======
        # 🛡️ 作用：把 NaN (非数字)、Inf (无穷大) 替换为合法值，防止它们进入神经网络导致梯度爆炸
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        # 🛡️ 作用：硬性截断，防止物理引擎穿透时出现负数距离
        ranges = np.clip(ranges, 0, max_range)
        
        # ======= 3. RL 状态空间适配 (核心逻辑) =======
        n = len(ranges)
        step = max(1, n // beams)
        truncated_len = (n // step) * step
        
        # 🎯 算法解释：为什么用 .min(axis=1) 而不是 .mean()？
        # 因为对于避障任务，漏掉一个近处障碍物是致命的。按区域取最小值，能最大程度保留危险特征。
        sampled = ranges[:truncated_len].reshape(-1, step).min(axis=1)
        
        # 🛡️ 维度强一致性校验：无论底层雷达发了多少束光，出去的必须是 beams 维
        if len(sampled) > beams:
            sampled = sampled[:beams]
        elif len(sampled) < beams:
            sampled = np.pad(sampled, (0, beams - len(sampled)), 'constant', constant_values=max_range)
            
        # 归一化到 [0, 1]，神经网络最喜欢的输入区间
        return sampled / max_range

    def get_laser_min_dist(self, max_range: float) -> float:
        """
        获取原始激光点中的最小距离。
        用途：主要供给 Reward 函数计算接近障碍物的惩罚项。
        （逻辑与 Env._min_laser() 完全一致）
        """
        ranges = self.get_laser_ranges()
        if ranges is None:
            return max_range

        # 必须过滤 NaN 和 Inf，否则 np.min 会抛出异常或返回异常值导致奖励崩溃
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        return float(np.min(ranges))

    def get_goal_relative(self, goal_x: float, goal_y: float) -> dict:
        """
        计算目标点相对于机器人当前位姿的本体坐标、距离和偏航角。
        （逻辑与 Env._get_obs() 中目标处理部分完全一致）
        """
        odom = self.get_odom_data()
        if not odom:
            return {"local_x": 0.0, "local_y": 0.0, "angle": 0.0, "dist": 1.0}
        
        # 坐标系转换：将世界坐标系下的目标偏移量，转换为机器人本体坐标系下的偏移量
        dx = goal_x - odom['x']
        dy = goal_y - odom['y']
        yaw = odom['yaw']
        # 旋转矩阵二维展开：[cos(yaw), sin(yaw); -sin(yaw), cos(yaw)] * [dx, dy]^T
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        dist = math.hypot(dx, dy)
        angle = math.atan2(local_y, local_x)
        return {"local_x": local_x, "local_y": local_y, "angle": angle, "dist": dist}

    def get_imu_normalized(self, acc_max: float, gyro_max: float) -> dict:
        """
        返回归一化到 [-1,1] 的 IMU 数据。
        （逻辑与 Env._get_obs() 中 IMU 处理部分完全一致）
        """
        imu = self.get_imu_data()
        if not imu:
            return {
                "acc": np.zeros(3, dtype=np.float32),
                "gyro": np.zeros(3, dtype=np.float32),
                "rpy": np.zeros(2, dtype=np.float32)
            }
        acc = np.clip(np.array(imu['acc']) / acc_max, -1.0, 1.0)
        gyro = np.clip(np.array(imu['gyro']) / gyro_max, -1.0, 1.0)
        rpy = np.clip(np.array(imu['rpy'][:2]) / (math.pi/4), -1.0, 1.0)
        return {"acc": acc, "gyro": gyro, "rpy": rpy}
    
    def is_collision(self, threshold: float, max_range: float) -> bool:
        """判断当前激光最小距离是否小于碰撞阈值。"""
        min_dist = self.get_laser_min_dist(max_range)
        return min_dist < threshold

### 上面主要是改了不用ATS实现，
# 这样改完之后，不管激光雷达是 10Hz 还是 5Hz，不管 Gazebo 掉不掉帧，get_laser_ranges() 永远是 0 延迟返回当前最新值，彻底告别 50ms 容差带来的假死隐患。
# 砍掉了 message_filters：去掉了 MutuallyExclusiveCallbackGroup 和 ApproximateTimeSynchronizer 的实例化。
# 三权分立：_latest_synced_data 字典被拆成了 _latest_laser、_latest_imu、_latest_odom 三个独立变量。这意味着哪怕某一帧激光雷达丢了，你的里程计依然能正常返回最新的位置，绝不会被激光雷达“绑架”。
# 锁的粒度更细：以前是拿到大字典才加锁，现在是各自回调里加锁写各自的数据，读取时也只锁极短的瞬间。
# API 向下完全兼容：外层的 get_laser_ranges() 等方法签名一个没变，你的 Env 代码完全不需要做任何修改，直接无缝切换。


# """
# 通用 ROS2 机器人通信桥（仿真无关）。

# 职责：
# - 传感器数据订阅与时间戳对齐
# - 执行器指令发布
# - 数据降维：对外只暴露 Python 原生类型 和 Numpy 数组

# 适用场景：Gazebo 仿真、Isaac Sim 仿真、真机部署。
# 只要 ROS2 Topic 名和消息格式一致，此类无需任何修改。
# """

# import math
# import logging
# from typing import Dict, Optional

# import numpy as np
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# from sensor_msgs.msg import LaserScan, Imu
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist, PoseStamped

# import message_filters
# from message_filters import ApproximateTimeSynchronizer

# from core.ros2.channels.bridges.base import BaseBridge

# logger = logging.getLogger(__name__)


# class RobotBridge(BaseBridge):
#     """
#     仿真无关的机器人通信桥。

#     通过 ApproximateTimeSynchronizer 保证传感器数据的物理时间戳一致性，
#     通过 MutuallyExclusiveCallbackGroup 让 Runtime 保证串行调度，
#     通过 _lock + fallback 保证 RL 的 step() 绝不因数据缺失而崩溃。
#     """

#     def __init__(self, runtime, node_name: str = "robot_bridge"):
#         super().__init__(node_name=node_name, runtime=runtime)

#         self._laser_noise_threshold = 0.0

#         # RL 安全兜底数据：开机第一帧 ATS 尚未凑齐时使用
#         self._fallback_data = {
#             "laser": np.zeros(360, dtype=np.float32),
#             "odom": {"x": 0.0, "y": 0.0, "yaw": 0.0,
#                      "vx": 0.0, "vy": 0.0, "wz": 0.0},
#             "imu": {"acc": [0.0, 0.0, 0.0],
#                     "gyro": [0.0, 0.0, 0.0],
#                     "rpy": [0.0, 0.0]}
#         }
#         self._latest_synced_data: Optional[Dict] = None

#     # ================================================================
#     #  一、模板方法实现
#     # ================================================================

#     def setup(self, *,
#               laser_topic: str, imu_topic: str, odom_topic: str,
#               cmd_vel_topic: str, goal_topic: str,
#               laser_noise_threshold: float = 0.0):
#         """一站式初始化传感器 + 执行器"""
#         self._laser_noise_threshold = laser_noise_threshold
#         self._setup_sensors(laser_topic, imu_topic, odom_topic)
#         self._setup_actuators(cmd_vel_topic, goal_topic)

#     # ================================================================
#     #  二、传感器（内部）
#     # ================================================================

#     def _setup_sensors(self, laser_topic: str, imu_topic: str, odom_topic: str):
#         qos = QoSProfile(
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=10
#         )

#         # 向 Runtime 申请互斥回调组：三个传感器回调串行执行，ATS 内部安全
#         self._sensor_cg = MutuallyExclusiveCallbackGroup()

#         laser_sub = message_filters.Subscriber(
#             self, LaserScan, laser_topic,
#             qos_profile=qos, callback_group=self._sensor_cg
#         )
#         imu_sub = message_filters.Subscriber(
#             self, Imu, imu_topic,
#             qos_profile=qos, callback_group=self._sensor_cg
#         )
#         odom_sub = message_filters.Subscriber(
#             self, Odometry, odom_topic,
#             qos_profile=qos, callback_group=self._sensor_cg
#         )

#         ats = ApproximateTimeSynchronizer(
#             [laser_sub, imu_sub, odom_sub],
#             queue_size=10,
#             slop=0.05  # 允许 50ms 的时间差
#         )
#         ats.registerCallback(self._ats_sync_callback)

#         self._topic_map.update({
#             "laser": laser_topic, "imu": imu_topic, "odom": odom_topic
#         })
#         logger.info("[RobotBridge] 传感器已通过 ATS 绑定")

#     def _ats_sync_callback(self, laser_msg: LaserScan, imu_msg: Imu, odom_msg: Odometry):
#         """物理时间戳对齐后触发，受 _sensor_cg 保护，绝对不会并发"""
#         if self._laser_noise_threshold > 0:
#             for i in range(len(laser_msg.ranges)):
#                 if laser_msg.ranges[i] <= self._laser_noise_threshold:
#                     laser_msg.ranges[i] = laser_msg.range_max

#         with self._lock:
#             self._latest_synced_data = {
#                 "laser": laser_msg, "imu": imu_msg, "odom": odom_msg
#             }

#     # ================================================================
#     #  三、执行器（内部）
#     # ================================================================

#     def _setup_actuators(self, cmd_vel_topic: str, goal_topic: str):
#         self._create_publisher(cmd_vel_topic, Twist, 10)
#         self._create_publisher(goal_topic, PoseStamped, 10)
#         self._topic_map.update({"cmd_vel": cmd_vel_topic, "goal": goal_topic})

#     # ================================================================
#     #  四、数据读取 API
#     # ================================================================

#     def _safe_get_synced(self, key: str):
#         with self._lock:
#             data = self._latest_synced_data
#         if data and data.get(key) is not None:
#             return data[key]
#         return self._fallback_data.get(key)

#     def get_laser_ranges(self) -> np.ndarray:
#         return np.array(self._safe_get_synced("laser").ranges, dtype=np.float32)

#     def get_odom_data(self) -> Dict[str, float]:
#         msg = self._safe_get_synced("odom")
#         p, o, t = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
#         siny_cosp = 2 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
#         return {
#             "x": p.x, "y": p.y, "yaw": math.atan2(siny_cosp, cosy_cosp),
#             "vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z
#         }

#     def get_imu_data(self) -> Dict[str, list]:
#         msg = self._safe_get_synced("imu")
#         a, g, o = msg.linear_acceleration, msg.angular_velocity, msg.orientation
#         sinr_cosp = 2 * (o.w * o.x + o.y * o.z)
#         cosr_cosp = 1 - 2 * (o.x * o.x + o.y * o.y)
#         sinp = 2 * (o.w * o.y - o.z * o.x)
#         return {
#             "acc": [a.x, a.y, a.z],
#             "gyro": [g.x, g.y, g.z],
#             "rpy": [
#                 math.atan2(sinr_cosp, cosr_cosp),
#                 math.asin(np.clip(sinp, -1.0, 1.0))
#             ]
#         }

#     # ================================================================
#     #  五、执行 API
#     # ================================================================

#     def send_velocity(self, lin_x: float, ang_z: float):
#         topic = self._topic_map.get("cmd_vel")
#         if not topic:
#             return
#         msg = Twist()
#         msg.linear.x, msg.angular.z = float(lin_x), float(ang_z)
#         self._publish(topic, msg)
