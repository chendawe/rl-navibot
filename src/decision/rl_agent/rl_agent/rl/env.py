"""
TurtleBot3 Waffle 导航 Gymnasium 环境 (ROS2 原生版)

依赖: rclpy, sensor_msgs, nav_msgs, geometry_msgs
观测空间 (37 维):
    [0:24]  LiDAR        (降采样, [0,1])
    [24:27] IMU 线加速度
    [27:30] IMU 角速度
    [30:32] IMU 姿态     [Roll, Pitch]
    [32]    目标相对角度 (归一化 [-1,1]) -> dyaw
    [33]    目标距离     (归一化 [0,1]) -> distance
    [34]    里程计线速度
    [35]    里程计角速度
    [36]    上一步动作线速度
"""

"""
TurtleBot3 Waffle 导航 Gymnasium 环境 (ROS2 原生版 - 最终优化版)

改进点:
1. 参数命名严格对应 config 文件。
2. Reset 目标生成逻辑增加兜底方案 (默认 0,0)。
3. 增加车身稳定性惩罚 (基于 Roll/Pitch)。
4. 明确 Action Clip 的安全性意义。
"""
"""
TurtleBot3 Waffle 导航 Gymnasium 环境 (ROS2 原生版 - 动作平滑优化版)

改进点:
1. 新增动作平滑性惩罚 (Action Smoothness Penalty)。
2. 优化卡住惩罚逻辑：改为检测“指令速度”与“实际速度”的差值。
3. 确保初始化状态符合物理静止状态。
"""

import math
import time
import threading
import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ROS2 相关导入
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped


from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose


logger = logging.getLogger(__name__)

class TurtleBot3HouseEnv(gym.Env, Node):
    """
    直接继承 rclpy.Node，去除外部接口依赖。
    所有配置参数均严格对应 config 键名。
    """

    metadata = {"render_modes": []}

    # 默认 House 地图安全区域 (XML 坐标)
    DEFAULT_SAFE_ZONES = [
        {"cx":  0.0, "cy":  0.0, "r": 1.2},
        {"cx":  1.8, "cy":  1.0, "r": 0.7},
        {"cx": -1.8, "cy":  1.0, "r": 0.7},
        {"cx":  1.8, "cy": -1.0, "r": 0.7},
        {"cx": -1.8, "cy": -1.0, "r": 0.7},
        {"cx":  3.0, "cy":  0.0, "r": 0.6},
        {"cx": -3.0, "cy":  0.0, "r": 0.6},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 初始化 Gym 环境
        gym.Env.__init__(self)
        
        # 默认配置结构
        default_config = {
            "environment": {
                "laser_range_max": 3.5,
                "laser_beams_num": 24,
                "lin_vel_max": 0.22,
                "ang_vel_max": 1.5,
                "episode_steps_max": 500,
                "step_duration": 0.1,
                "dist_to_goal_threshold": 0.3,
                "dist_to_goal_clip_norm": 5.0,
                "proximity_to_collision_threshold": 0.22,
                "proximity_to_be_safe_min": 0.35,
            },
            "reward": {
                "reward_at_goal": 300.0,
                "penalty_at_collision": -300.0,
                "reward_factor_approaching_goal": 5.0,
                "penalty_elapsing_time": -0.5,
                "penalty_stuck": -2.0,
                "reward_good_orientation": 2.0,
                "penalty_in_safe_proximity": -1.0,
                "penalty_instability": -2.0,
                "penalty_action_smoothness": -0.5, # 新增配置项
            }
        }
        
        # 合并用户配置
        self.config = default_config.copy()
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value

        # --- 提取环境参数 (变量名严格对应 config 键名) ---
        cfg_env = self.config["environment"]
        self.laser_range_max = cfg_env.get("laser_range_max")
        self.laser_beams_num = cfg_env.get("laser_beams_num")
        self.lin_vel_max = cfg_env.get("lin_vel_max")
        self.ang_vel_max = cfg_env.get("ang_vel_max")
        self.episode_steps_max = cfg_env.get("episode_steps_max")
        self.step_duration = cfg_env.get("step_duration")
        self.dist_to_goal_threshold = cfg_env.get("dist_to_goal_threshold")
        self.dist_to_goal_clip_norm = cfg_env.get("dist_to_goal_clip_norm")
        self.proximity_to_collision_threshold = cfg_env.get("proximity_to_collision_threshold")
        self.proximity_to_be_safe_min = cfg_env.get("proximity_to_be_safe_min")
        self.lin_vel_stuck_threshold = cfg_env.get("lin_vel_stuck_threshold")
        # 物理极限 (内部用)
        self.lin_acc_physics_max = cfg_env.get("lin_acc_physics_max")
        self.ang_vel_imu_physics_max = cfg_env.get("ang_vel_imu_physics_max")        
        # 衍生参数
        self.dist_to_goal_gen_min = cfg_env.get("dist_to_goal_gen_min")
        
        
        # --- 提取奖励参数 (变量名严格对应 config 键名) ---
        cfg_rew = self.config["reward"]
        self.reward_at_goal = cfg_rew.get("reward_at_goal")
        self.penalty_at_collision = cfg_rew.get("penalty_at_collision")
        self.reward_factor_approaching_goal = cfg_rew.get("reward_factor_approaching_goal")
        self.penalty_elapsing_time = cfg_rew.get("penalty_elapsing_time")
        self.penalty_stuck = cfg_rew.get("penalty_stuck")
        self.reward_good_orientation = cfg_rew.get("reward_good_orientation")
        self.penalty_in_safe_proximity = cfg_rew.get("penalty_in_safe_proximity")
        self.penalty_instability = cfg_rew.get("penalty_instability")
        self.penalty_action_smoothness = cfg_rew.get("penalty_action_smoothness")

        # 安全区域
        self.safe_zones = self.DEFAULT_SAFE_ZONES

        # --- 初始化 ROS2 Node ---
        if not rclpy.ok():
            rclpy.init()
        
        Node.__init__(self, 'turtlebot3_gym_env')
        
        # --- 数据缓存与锁 (线程安全) ---
        self._lock = threading.Lock()
        self._latest_scan: Optional[LaserScan] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None

        # --- 订阅者 (使用 Best Effort 适配 Gazebo) ---
        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.BEST_EFFORT,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, qos_profile)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self._imu_cb, qos_profile)
        # self.odom_sub = self.create_subscription(
        #     Odometry, '/odometry/filtered', self._odom_cb, qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile)

        # --- 发布者 ---
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # --- ROS2 Spin 线程 ---
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # --- 空间定义 ---
        # obs_size 计算细节: 24(Lidar) + 3(ImuAcc) + 3(ImuGyro) + 2(ImuRP) + 1(GoalAng) + 1(GoalDist) + 2(OdomVel) + 1(LastAct)
        self._obs_size = 37 
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-self.lin_vel_max, -self.ang_vel_max], dtype=np.float32),
            high=np.array([ self.lin_vel_max,  self.ang_vel_max], dtype=np.float32),
            dtype=np.float32,
        )

        # --- 内部状态 ---
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.prev_dist = 0.0 # 上一时刻到目标的距离
        self.episode_step = 0
        # 关键初始化：机器人初始状态为静止
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # ====== 新增：创建独立的服务调用节点（避免死锁） ======
        from rclpy.executors import SingleThreadedExecutor
        self._reset_node = rclpy.create_node('reset_service_node')
        self._reset_client = self._reset_node.create_client(
            SetEntityState, '/set_entity_state')
        
        self._reset_executor = SingleThreadedExecutor()
        self._reset_executor.add_node(self._reset_node)
        self._reset_thread = threading.Thread(
            target=self._reset_executor.spin, daemon=True)
        self._reset_thread.start()
        self.get_logger().info("Reset service thread started.")
        # ====== 新增结束 ======
        
        logger.info("TurtleBot3HouseEnv initialized.")

    # ==================== ROS2 回调与工具 ====================

    def _spin(self):
        rclpy.spin(self)

    def _scan_cb(self, msg: LaserScan):
        with self._lock:
            self._latest_scan = msg

    def _imu_cb(self, msg: Imu):
        with self._lock:
            self._latest_imu = msg

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._latest_odom = msg

    def _get_odom_data(self) -> Optional[Dict]:
        with self._lock:
            if self._latest_odom is None: return None
            p = self._latest_odom.pose.pose.position
            o = self._latest_odom.pose.pose.orientation
            t = self._latest_odom.twist.twist
            
            siny_cosp = 2 * (o.w * o.z + o.x * o.y)
            cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return {
                "x": p.x, "y": p.y, "yaw": yaw,
                "vx": t.linear.x, "vy": t.linear.y, "vz": t.angular.z
            }

    def _get_imu_data(self) -> Optional[Dict]:
        with self._lock:
            if self._latest_imu is None: return None
            a = self._latest_imu.linear_acceleration
            g = self._latest_imu.angular_velocity
            o = self._latest_imu.orientation
            
            sinr_cosp = 2 * (o.w * o.x + o.y * o.z)
            cosr_cosp = 1 - 2 * (o.x * o.x + o.y * o.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (o.w * o.y - o.z * o.x)
            pitch = math.asin(np.clip(sinp, -1.0, 1.0))

            return {
                "acc": [a.x, a.y, a.z],
                "gyro": [g.x, g.y, g.z],
                "rpy": [roll, pitch]
            }

    def _get_laser_data(self) -> np.ndarray:
        with self._lock:
            if self._latest_scan is None:
                return np.ones(self.laser_beams_num, dtype=np.float32)
            
            ranges = np.array(self._latest_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=self.laser_range_max, 
                                   posinf=self.laser_range_max, neginf=0.0)
            ranges = np.clip(ranges, 0, self.laser_range_max)
            
            n = len(ranges)
            step = max(1, n // self.laser_beams_num)
            sampled = np.array([np.min(ranges[i:i+step]) for i in range(0, n, step)])
            
            if len(sampled) > self.laser_beams_num:
                sampled = sampled[:self.laser_beams_num]
            elif len(sampled) < self.laser_beams_num:
                sampled = np.pad(sampled, (0, self.laser_beams_num - len(sampled)), 
                                 'constant', constant_values=self.laser_range_max)
            
            return sampled / self.laser_range_max

    def send_velocity(self, lin_x, ang_z):
        msg = Twist()
        msg.linear.x = float(lin_x)
        msg.angular.z = float(ang_z)
        self.vel_pub.publish(msg)

    # ==================== Gym 接口 ====================

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        gym.Env.reset(self, seed=seed)
        
        # 1. 逻辑层重置
        self.send_velocity(0.0, 0.0)
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.episode_step = 0
        
        # 2. 生成随机起点和目标点
        start = self._sample_safe_position()
        goal = self._sample_safe_position()
        
        attempts = 0
        while self._euclidean(start, goal) < self.dist_to_goal_gen_min:
            goal = self._sample_safe_position()
            attempts += 1
            if attempts >= 20:
                logger.warn("Failed to generate valid goal. Using default (0, 0).")
                goal = (0.0, 0.0)
                break
            
        self.goal_x, self.goal_y = goal

        # 3. 调用 Gazebo 物理重置 (方法 B)
        start_yaw = self.np_random.uniform(0, 2 * math.pi)
        self.reset_gazebo_robot(start[0], start[1], start_yaw)
        
        # ---------------------------------------------------------
        # 优化：轮询等待 Gazebo 状态同步 (替代固定的 sleep(0.2))
        # ---------------------------------------------------------
        # MAX_WAIT_SEC = 0.1      # 最大等待时间 100ms
        # CHECK_FREQ = 50         # 检测频率 50Hz
        # POSITION_TOL = 0.03     # 位置容差 3cm
        MAX_WAIT_SEC = 0.5      # 给足 500ms
        CHECK_FREQ = 50         
        POSITION_TOL = 0.15     # 容差放宽到 15cm，完全够用
        
        start_wait_time = time.time()
        arrived = False
        dist_error = float('inf')  # ← 必须加这行防 None 崩溃
        
        while time.time() - start_wait_time < MAX_WAIT_SEC:
            # 获取当前里程计数据
            odom = self._get_odom_data()
            # if odom:
            if not odom:
                # print(f"[DEBUG] _get_odom_data() returned None, elapsed={time.time()-start_wait_time:.3f}s")
                pass
            else:
                dx = odom['x'] - start[0]
                dy = odom['y'] - start[1]
                dist_error = math.hypot(dx, dy)
                
                # 如果位置误差小于容差，说明 Gazebo 已经更新到位
                if dist_error < POSITION_TOL:
                    arrived = True
                    break
            
            # 短暂休眠，避免 CPU 空转
            time.sleep(1.0 / CHECK_FREQ)
            
            # # 【可选优化】检测是否一出生就撞了
            # # 如果一出生就撞墙，说明这个随机点运气不好（被障碍物挡了），直接重新 Reset
            # if self._check_collision():
            #     # 这里可以递归调用，或者直接返回一个失败的 obs，让 step 去处理
            #     # 最简单的是打印日志，然后让这一局作为“失败的教训”开始
            #     logger.warn("Robot spawned in collision! Starting episode with penalty.")
            #     # 也可以强行 return self.reset() 强制重开，但可能导致死循环，慎用

        if not arrived:
            # 仅在超时时打印警告，避免日志刷屏
            logger.warn(f"Reset sync timeout. Pos error: {dist_error:.3f}m")

        # 极短缓冲：确保物理引擎解算完最后的穿透/稳定
        time.sleep(0.01)

        # 4. 发布 Goal Marker (可视化用)
        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.position.x = self.goal_x
        p.pose.position.y = self.goal_y
        p.pose.position.z = 0.0
        p.pose.orientation.w = 1.0
        self.goal_pub.publish(p)

        # 5. 获取初始观测
        obs = self._get_obs()
        # 再次获取 odom 用于初始化 prev_dist
        odom = self._get_odom_data()
        self.prev_dist = self._goal_dist(odom) if odom else self.dist_to_goal_clip_norm
        
        return obs, {"start": start, "goal": goal}


    # 方法 A: 简单粗暴 - 重置整个世界 (所有物体回到初始位置)
    # 优点：简单
    # 缺点：如果有随机生成的障碍物，它们也会被重置掉，可能失去随机性
    def reset_gazebo_world(self):
        client = self.create_client(Empty, '/reset_world')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        request = Empty.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    # 方法 B: 精确控制 - 仅移动机器人到指定起点 (推荐)
    # 优点：只重置机器人，保留环境中的其他设置 (如随机障碍物)
    # 缺点：代码稍多
    # 推荐的方法 B 实现 (防死锁版)
    def reset_gazebo_robot(self, start_x, start_y, start_yaw=0.0):
        if not self._reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Service /set_entity_state not available! Cannot reset environment.")
        else:
            self.get_logger().info("/set_entity_state service is available.")
        
        # 1. 等待服务可用
        if not self._reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Service /set_entity_state not available! Cannot reset environment.")

        # 2. 构造请求
        request = SetEntityState.Request()
        request.state.name = "waffle"
        request.state.pose.position.x = start_x
        request.state.pose.position.y = start_y
        request.state.pose.position.z = 0.1

        request.state.pose.orientation.x = 0.0
        request.state.pose.orientation.y = 0.0
        request.state.pose.orientation.z = math.sin(start_yaw / 2)
        request.state.pose.orientation.w = math.cos(start_yaw / 2)

        request.state.twist.linear.x = 0.0
        request.state.twist.angular.z = 0.0
        request.state.reference_frame = "world"

        self.get_logger().info(
            f"Reset request: name={request.state.name}, "
            f"pos=({start_x}, {start_y}, 0.1), "
            f"quat=(0.0, 0.0, {math.sin(start_yaw/2)}, {math.cos(start_yaw/2)})"
        )

        # 3. 异步调用（由独立线程的 executor 处理回调）
        future = self._reset_client.call_async(request)

        # 4. 自己实现超时（不使用 future.result(timeout=...)）
        start_time = time.time()
        while not future.done():
            time.sleep(0.01)
            if time.time() - start_time > 5.0:  # 5 秒超时
                raise RuntimeError("Reset robot service call timeout!")

        # 5. 检查底层异常
        if future.exception() is not None:
            raise RuntimeError(
                f"Reset robot service call raised an exception: {future.exception()}"
            )

        # 6. 取结果并判断 success
        result = future.result()
        # 【修改这里】：去掉 status_message
        self.get_logger().info(f"Reset result: success={result.success}")
        
        if not result.success:
            self.get_logger().error("Gazebo failed to reset robot state!")

    # def reset_gazebo_robot(self, start_x, start_y, start_yaw=0.0):
    #     client = self.create_client(SetEntityState, '/set_entity_state')
        
    #     # 1. 等待服务可用 (带超时)
    #     if not client.wait_for_service(timeout_sec=2.0):
    #         raise RuntimeError("Service /set_entity_state not available! Cannot reset environment.")

    #     request = SetEntityState.Request()
    #     request.state.name = "waffle"
    #     request.state.pose.position.x = start_x
    #     request.state.pose.position.y = start_y
    #     request.state.pose.position.z = 0.1
    #     request.state.pose.orientation.x = 0.0  # 补全 x
    #     request.state.pose.orientation.y = 0.0  # 补全 y
    #     request.state.pose.orientation.z = math.sin(start_yaw / 2)
    #     request.state.pose.orientation.w = math.cos(start_yaw / 2)
    #     request.state.twist.linear.x = 0.0
    #     request.state.twist.angular.z = 0.0
    #     request.state.reference_frame = "world"

    #     self.get_logger().info(f"Reset request: name={request.state.name}, "
    #                         f"pos=({start_x}, {start_y}, 0.1), "
    #                         f"quat=({0.0}, {0.0}, {math.sin(start_yaw/2)}, {math.cos(start_yaw/2)})")

    #     # 2. 调用服务并等待结果 (带超时)
    #     future = client.call_async(request)
        
    #     start_time = time.time()
    #     while not future.done():
    #         if time.time() - start_time > 10.0: # 3秒超时
    #             raise RuntimeError("Reset robot service call timeout!")
    #         rclpy.spin_once(self, timeout_sec=0.01) 
            
    #     # 3. 检查返回值
    #     result = future.result()
    #     self.get_logger().info(f"Reset result: success={result.success}, message={result.status_message}")
    #     if not result.success: # 假设服务返回包含 success 字段
    #          self.get_logger().error("Gazebo failed to reset robot state!")



    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 安全性考量：虽然策略网络通常输出有界，但为了防止异常值导致仿真崩溃，做裁剪是必要的
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.send_velocity(action[0], action[1])
        time.sleep(self.step_duration)
        
        obs = self._get_obs()
        odom = self._get_odom_data()
        imu  = self._get_imu_data()
        
        if odom is None:
            return obs, -1.0, False, True, {"error": "odom unavailable"}

        curr_dist = self._goal_dist(odom)
        
        # 终止判定
        goal_reached = curr_dist < self.dist_to_goal_threshold
        collision    = self._check_collision()
        
        # 奖励计算
        reward = self._compute_reward(
            curr_dist=curr_dist, prev_dist=self.prev_dist,
            goal_reached=goal_reached, collision=collision,
            action=action, odom=odom, imu=imu
        )
        
        # 更新状态
        self.prev_dist = curr_dist
        self.last_action = action.copy() # 保存当前动作用于下一步计算平滑性
        self.episode_step += 1
        
        terminated = goal_reached or collision
        truncated  = self.episode_step >= self.episode_steps_max
        
        if terminated or truncated:
            self.send_velocity(0.0, 0.0)
            
        return obs, float(reward), terminated, truncated, {}

    def close(self):
        self.send_velocity(0.0, 0.0)
        if rclpy.ok():
            self.destroy_node()

    # ==================== 内部逻辑 ====================

    def _sample_safe_position(self) -> Tuple[float, float]:
        zone = self.safe_zones[self.np_random.integers(len(self.safe_zones))]
        angle = self.np_random.uniform(0, 2 * math.pi)
        r = self.np_random.uniform(0, zone["r"])
        return (zone["cx"] + r * math.cos(angle), zone["cy"] + r * math.sin(angle))

    @staticmethod
    def _euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _goal_dist(self, odom: Dict) -> float:
        return math.hypot(self.goal_x - odom["x"], self.goal_y - odom["y"])

    def _check_collision(self) -> bool:
        with self._lock:
            if self._latest_scan is None: return False
            ranges = np.array(self._latest_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=self.laser_range_max, posinf=self.laser_range_max)
            return float(np.min(ranges)) < self.proximity_to_collision_threshold

    def _min_laser(self) -> float:
        with self._lock:
            if self._latest_scan is None: return self.laser_range_max
            ranges = np.array(self._latest_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=self.laser_range_max, posinf=self.laser_range_max)
            return float(np.min(ranges))

    def _get_obs(self) -> np.ndarray:
        laser_norm = self._get_laser_data()
        odom = self._get_odom_data()
        imu  = self._get_imu_data()

        imu_acc = np.zeros(3, dtype=np.float32)
        imu_gyro = np.zeros(3, dtype=np.float32)
        imu_rp = np.zeros(2, dtype=np.float32)
        goal_angle, goal_dist = 0.0, 1.0
        odom_vel = np.zeros(2, dtype=np.float32)

        if imu is not None:
            imu_acc = np.clip(np.array(imu['acc']) / self.lin_acc_physics_max, -1.0, 1.0)
            imu_gyro = np.clip(np.array(imu['gyro']) / self.ang_vel_imu_physics_max, -1.0, 1.0)
            # 归一化 Roll/Pitch，约 45度 (pi/4) 映射到 1.0
            imu_rp = np.clip(np.array(imu['rpy'][:2]) / (math.pi/4), -1.0, 1.0)

        if odom is not None:
            dx = self.goal_x - odom['x']
            dy = self.goal_y - odom['y']
            yaw = odom['yaw']
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
            goal_angle = math.atan2(local_y, local_x) / math.pi
            goal_dist = min(math.hypot(dx, dy) / self.dist_to_goal_clip_norm, 1.0)
            odom_vel = np.clip(
                np.array([odom['vx'], odom['vz']]) / np.array([self.lin_vel_max, self.ang_vel_max]), 
                -1.0, 1.0
            )

        last_act_norm = np.clip(self.last_action[0] / self.lin_vel_max, -1.0, 1.0)

        obs = np.concatenate([
            laser_norm, imu_acc, imu_gyro, imu_rp,
            [goal_angle], [goal_dist], odom_vel, [last_act_norm]
        ]).astype(np.float32)
        
        return obs

    def _compute_reward(self, curr_dist, prev_dist, goal_reached, collision, action, odom, imu):
        reward = 0.0
        
        # 1. 终止条件奖励
        if goal_reached: return self.reward_at_goal
        if collision:    return self.penalty_at_collision

        # 2. 距离变化奖励
        dist_delta = prev_dist - curr_dist
        reward += dist_delta * self.reward_factor_approaching_goal

        # 3. 朝向奖励
        yaw = odom['yaw']
        target_angle = math.atan2(self.goal_y - odom['y'], self.goal_x - odom['x'])
        yaw_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
        heading_factor = math.cos(yaw_error)
        
        if abs(odom['vx']) > 0.01:
            reward += heading_factor * self.reward_good_orientation

        # 4. 安全惩罚
        min_dist = self._min_laser()
        if min_dist < self.proximity_to_be_safe_min:
            reward += self.penalty_in_safe_proximity * ((self.proximity_to_be_safe_min - min_dist) / self.proximity_to_be_safe_min)**2

        # 5. 稳定性惩罚 (基于 IMU 的 Roll 和 Pitch)
        if imu is not None:
            roll, pitch = imu['rpy']
            tilt_factor = (roll**2 + pitch**2) / ( (math.pi/6)**2 )
            reward -= abs(self.penalty_instability) * min(tilt_factor, 1.0)

        # 6. 卡住惩罚 (优化版：检测指令与实际速度的偏差)
        # 如果指令速度很大，但实际速度很小，说明发生了打滑或卡死
        vel_error = abs(action[0]) - abs(odom['vx'])
        if vel_error > self.lin_vel_stuck_threshold: # 允许 0.1 的控制误差
             reward += self.penalty_stuck * (vel_error / self.lin_vel_max)

        # 7. 动作平滑性惩罚 (新增)
        # 1. 计算变化量
        diff_lin = action[0] - self.last_action[0]
        diff_ang = action[1] - self.last_action[1]

        # 2. 归一化 (变化量 / 最大速度 = 变化百分比)
        norm_diff_lin = diff_lin / self.lin_vel_max
        norm_diff_ang = diff_ang / self.ang_vel_max

        # 3. 计算加权欧氏距离 (此时两者量纲一致，都是 0~1 的比例)
        smoothness_penalty = np.sqrt(norm_diff_lin**2 + norm_diff_ang**2)

        # 4. 施加惩罚
        reward += self.penalty_action_smoothness * smoothness_penalty
        # 8. 时间惩罚
        reward += self.penalty_elapsing_time
        
        return reward
