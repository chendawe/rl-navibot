
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
import rclpy.parameter
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose

import std_srvs.srv
from gazebo_msgs.srv import SetEntityState
import math

logger = logging.getLogger(__name__)


def fetch_tb3_urdf():
    """在主进程中提前拉取完整的 URDF，子进程直接 pickle 继承，不再查 ROS2"""
    if not rclpy.ok():
        rclpy.init()

    temp_node = rclpy.create_node('urdf_prefetcher')
    urdf_string = None

    # ====== 第一步：找到目标节点名 ======
    target_node = None
    for attempt in range(50):  # 最多等5秒
        nodes = temp_node.get_node_names()
        for n in nodes:
            if 'robot_state_publisher' in n or 'state_publisher' in n:
                target_node = n
                break
        if target_node:
            break
        time.sleep(0.1)

    if not target_node:
        all_nodes = temp_node.get_node_names()
        temp_node.destroy_node()
        raise RuntimeError(
            f"No robot_state_publisher found! Visible nodes: {all_nodes}")

    print(f"[Prefetcher] Found node: {target_node}")

    # ====== 第二步：等 service 可用（这一步才是你之前的盲区） ======
    service_name = f'{target_node}/get_parameters'
    param_client = temp_node.create_client(GetParameters, service_name)

    print(f"[Prefetcher] Waiting for service {service_name} ...")
    if not param_client.wait_for_service(timeout_sec=5.0):
        temp_node.destroy_client(param_client)
        temp_node.destroy_node()
        raise RuntimeError(
            f"Service {service_name} not available after 5s! "
            "DDS service discovery may need more time.")

    print(f"[Prefetcher] Service ready, calling...")

    # ====== 第三步：调用服务拿 URDF ======
    req = GetParameters.Request()
    req.names = ['robot_description']
    future = param_client.call_async(req)

    start = time.time()
    while not future.done() and time.time() - start < 10.0:
        rclpy.spin_once(temp_node, timeout_sec=0.05)  # ← 加这一行来帮忙收快递！

    if future.done() and future.result() is not None:
        values = future.result().values
        if values:
            urdf_string = values[0].string_value

    temp_node.destroy_client(param_client)
    temp_node.destroy_node()

    if urdf_string is None:
        raise RuntimeError("Service call succeeded but robot_description was empty!")

    print(f"[Prefetcher] Successfully got URDF ({len(urdf_string)} chars)")
    return urdf_string


class TurtleBot3NavEnv(gym.Env, Node):
    """
    直接继承 rclpy.Node，去除外部接口依赖。
    所有配置参数均严格对应 config 键名。
    """

    metadata = {"render_modes": []}
    
    SAFE_ZONES = {}
    # dafault
    SAFE_ZONES["default"] = [
        # 角斗场底部
        {"cx": 0, "cy":  0, "r": 1.},
    ]
    
    def __init__(self, robot_urdf: Optional[str] = None, env_config: Optional[Dict[str, Any]] = None):
        gym.Env.__init__(self)
        
        def initialize_locked_ros2_node(self) :
            if not rclpy.ok():
                rclpy.init()
            Node.__init__(self, f'{self.bot["name"]}_gym_env')
            self._lock = threading.Lock()
            
        def start_spin_thread(self) :
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()
        

        self._robot_urdf = robot_urdf

        # ================= 1. 默认配置与合并 (不变) =================
        default_env_config = {
            "bot": { "name": "burger", "laser_range_max": 3.5, "laser_beams_num": 24, "lin_vel_max": 0.22, "ang_vel_max": 1.5,
                        "lin_vel_stuck_threshold": 0.12, "lin_acc_physics_max": 2.0, "ang_vel_imu_physics_max": 3.0,
                        "proximity_to_collision_threshold": 0.11, "proximity_to_be_safe_min": 0.30 },
            "world": { "name": "ttb3_world", "episode_steps_max": 500, "step_duration": 0.1,
                          "dist_to_goal_threshold": 0.35, "dist_to_goal_gen_min": 1.0, "dist_to_goal_clip_norm": 5.0,
                          "safe_zones": {} },
            "reward": { "reward_at_goal": 300.0, "penalty_at_collision": -300.0, "reward_factor_approaching_goal": 5.0,
                       "penalty_elapsing_time": -0.5, "penalty_stuck": -2.0, "reward_good_orientation": 2.0,
                       "penalty_in_safe_proximity": -1.0, "penalty_instability": -2.0, "penalty_action_smoothness": -0.5 }
        }
        if env_config:
            for key, value in env_config.items():
                if isinstance(value, dict) and key in default_env_config:
                    default_env_config[key].update(value)
                else:
                    default_env_config[key] = value

        # ================= 2. 参数极简封装 (无后缀) =================
        
        self.config = {
            "bot"   : default_env_config["bot"],
            "world" : default_env_config["world"],
            "reward": default_env_config["reward"]
        }
        
        self.bot = self.config["bot"]
        self.world = self.config["world"]
        self.rew = self.config["reward"]
        
        # 🔥 Safe Zones 单独拎出来，因为采样时访问频率极高
        self.zones = self.world.pop("safe_zones", {})
        if not self.zones:
            raise ValueError(f"配置中未找到 safe_zones！")

        # ================= 3. 初始化 ROS2 Node =================
        initialize_locked_ros2_node(self)
        self._latest_scan: Optional[LaserScan] = None
        self._latest_imu: Optional[Imu] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_scan_time: float = 0.0
        
        # ================= 4. ROS 接口大一统 =================
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.ros = {
            "subs": {
                "scan": self.create_subscription(LaserScan, '/scan', self._scan_cb, qos),
                "imu":  self.create_subscription(Imu, '/imu', self._imu_cb, qos),
                "odom": self.create_subscription(Odometry, '/odom', self._odom_cb, qos),
            },
            "pubs": {
                "vel":  self.create_publisher(Twist, '/cmd_vel', 10),
                "goal": self.create_publisher(PoseStamped, '/goal_pose', 10),
            },
            "clients": {
                "reset_world": self.create_client(std_srvs.srv.Empty, '/reset_world'),
                "set_entity":  self.create_client(SetEntityState, '/set_entity_state'),
                "delete":      self.create_client(DeleteEntity, '/delete_entity'),
                "spawn":       self.create_client(SpawnEntity, '/spawn_entity'),
            }
        }

        start_spin_thread(self)

        # ================= 5. RL 空间定义 =================
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(38,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-self.bot["lin_vel_max"], -self.bot["ang_vel_max"]], dtype=np.float32),
            high=np.array([ self.bot["lin_vel_max"],  self.bot["ang_vel_max"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.spaces = {"obs": obs_space, "act": act_space}
        
        # ⚠️ Gym 强制要求，必须挂载
        self.observation_space = obs_space
        self.action_space = act_space

        # ================= 6. 运行时状态大一统 (目标也塞进去) =================
        self.state = {
            "goal_x": 0.0,
            "goal_y": 0.0,
            "prev_dist": 0.0,
            "step": 0,
            "last_action": np.array([0.0, 0.0], dtype=np.float32),
        }

        self.print_config()




    def print_config(self):
        print("\n" + "=" * 60)
        print("🤖 TurtleBot3 Navigation Env Configuration")
        print("=" * 60)
        
        print(f"Bot Name   : {self.bot.get('name', 'N/A')}")
        print(f"World Name : {self.world.get('name', 'N/A')}")
        
        print("-" * 60)
        print("📡 Env Bot Config (Physics & Hardware):")
        for k, v in self.bot.items():
            print(f"  - {k:<35} : {v}")
            
        print("-" * 60)
        print("🗺️  Env World Config (Task & Map):")
        for k, v in self.world.items():
            print(f"  - {k:<35} : {v}")
            
        # 因为 safe_zones 被 pop 出来了，所以单独打印 self.zones
        zone_names = list(self.zones.keys())
        total_points = sum(len(pts) for pts in self.zones.values())
        print(f"  - {'safe_zones':<35} : {total_points} points in {len(zone_names)} zones")
        print(f"    {'Zones:':<33} : {zone_names}")
                
        print("-" * 60)
        print("🎁 Reward Config:")
        for k, v in self.rew.items():
            print(f"  - {k:<35} : {v}")
            
        print("-" * 60)
        print("🧠 RL Core Config:")
        obs_shape = self.spaces['obs'].shape
        act_low  = self.spaces['act'].low
        act_high = self.spaces['act'].high
        print(f"  - {'Obs Shape':<35} : {obs_shape}")
        print(f"  - {'Act Space':<35} : [{act_low[0]:.2f}, {act_high[0]:.2f}], [{act_low[1]:.2f}, {act_high[1]:.2f}]")
        
        print("=" * 60 + "\n")


    # ==================== ROS2 回调与工具 ====================

    def _spin(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

    def _scan_cb(self, msg: LaserScan):
        with self._lock:
            # 【核心修复】滤掉打到自身轮子/底盘的噪点
            # 正确做法：遍历原地修改底层数组，而不是生成新 list 覆盖
            for i in range(len(msg.ranges)):
                if msg.ranges[i] < 0.15:
                    msg.ranges[i] = msg.range_max
                    
            self._latest_scan = msg
            self._latest_scan_time = time.time()

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
                "vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z
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
        # 🔥 优化：将字典查询提到循环外，提升执行效率
        beams_num = self.bot["laser_beams_num"]
        max_range = self.bot["laser_range_max"]
        
        with self._lock:
            if self._latest_scan is None:
                return np.ones(beams_num, dtype=np.float32)
            
            ranges = np.array(self._latest_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=max_range, 
                                   posinf=max_range, neginf=0.0)
            ranges = np.clip(ranges, 0, max_range)
            
            n = len(ranges)
            step = max(1, n // beams_num)
            sampled = np.array([np.min(ranges[i:i+step]) for i in range(0, n, step)])
            
            if len(sampled) > beams_num:
                sampled = sampled[:beams_num]
            elif len(sampled) < beams_num:
                sampled = np.pad(sampled, (0, beams_num - len(sampled)), 
                                 'constant', constant_values=max_range)
            
            return sampled / max_range

    def send_velocity(self, lin_x, ang_z):
        msg = Twist()
        msg.linear.x = float(lin_x)
        msg.angular.z = float(ang_z)
        # 🔥 替换：self.vel_pub -> self.ros_pubs["vel"]
        self.ros["pubs"]["vel"].publish(msg)

    def _get_zero_obs(self):
        """
        返回一个全零观测，用于环境异常时的兜底。
        """
        import numpy as np
        # 💡 这里使用 self.observation_space 而不是 self.rl_cfg，因为 Gym 底层强制要求必须存在这个顶级属性
        return np.zeros(self.observation_space.shape, dtype=np.float32)


    # ==================== Gym 接口 ====================
    # ==================== Reset 辅助函数 ====================
    def _wait_for_services(self) -> bool:
        if hasattr(self, '_services_ready') and self._services_ready: return True
        self.get_logger().info("Waiting for Gazebo reset services...")
        try:
            if not self.ros["clients"]["set_entity"].wait_for_service(timeout_sec=5.0):
                raise RuntimeError("SetEntityState service not found!")
            if not self.ros["clients"]["reset_world"].wait_for_service(timeout_sec=5.0):
                raise RuntimeError("ResetWorld service not found!")
            self._services_ready = True
            self.get_logger().info("Gazebo services are ready!")
            return True
        except Exception as e:
            self.get_logger().error(f"Service check failed: {e}")
            return False

    def _clear_sensor_cache(self):
        with self._lock:
            self._latest_scan = None
            self._latest_scan_time = 0.0

    def _parse_reset_options(self, options: Optional[Dict]) -> Dict:
        if not options: return {"start_pos": None, "goal_pos": None, "safe_threshold": None, "skip_spawn_check": False}
        return {
            "start_pos": options.get("start_pos"),
            "goal_pos": options.get("goal_pos"),
            "safe_threshold": options.get("safe_threshold"),
            "skip_spawn_check": options.get("skip_spawn_check", False)
        }

    def _call_gazebo_async(self, req, client_key, timeout=3.0):
        """统一的 Gazebo 异步调用封装，避免满屏幕的 while not future.done"""
        future = self.ros["clients"][client_key].call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < timeout: 
            time.sleep(0.05)

    def _teleport_robot(self, x, y, yaw):
        """纯粹的实体位置强行重置"""
        req = SetEntityState.Request()
        req.state.name = self.bot["name"]
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = "world"
        self._call_gazebo_async(req, "set_entity")

    def _generate_valid_task(self, opts) -> Optional[Dict]:
        """带有重试逻辑的核心任务生成器"""
        custom_start = opts["start_pos"]
        target_goal = opts["goal_pos"]
        use_custom_start = custom_start is not None
        
        for attempt in range(5):
            # 1. 确定候选起点
            if use_custom_start:
                s_x, s_y, s_yaw = custom_start
            else:
                s_x, s_y = self._sample_safe_position()
                s_yaw = self.np_random.uniform(-math.pi, math.pi)
                
            # 2. 确定候选终点
            g_x, g_y = target_goal if target_goal else self._sample_safe_position()
            
            # 3. 距离校验 (仅限全随机时)
            if not use_custom_start and not target_goal:
                if self._euclidean((s_x, s_y), (g_x, g_y)) < self.world["dist_to_goal_gen_min"]:
                    continue
                    
            # 4. 调用仿真器重置并传送
            self._call_gazebo_async(std_srvs.srv.Empty.Request(), "reset_world")
            self._teleport_robot(s_x, s_y, s_yaw)
            time.sleep(0.5) # 等待物理引擎稳定和雷达刷新
            
            # 5. 碰撞与安全性校验
            is_spawn_safe = self._check_spawn_pos(threshold=opts["safe_threshold"]) if not opts["skip_spawn_check"] else True
            is_goal_safe = self._check_goal_pos(g_x, g_y) if not target_goal else True
            
            if is_spawn_safe and is_goal_safe:
                return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}
                
            if not use_custom_start:
                self.get_logger().warn(f"Validation failed, retry {attempt+1}/5...")
                
        return None

    def _fallback_task(self, opts) -> Dict:
        """重试耗尽后的兜底策略"""
        if opts["start_pos"] is not None:
            self.get_logger().error(f"Failed to set custom start pose: {opts['start_pos']}")
            s_x, s_y, s_yaw = opts["start_pos"]
        else:
            self.get_logger().warn("All retries failed! Fallback to (0.5, 0.5).")
            s_x, s_y, s_yaw = 0.5, 0.5, 0.0
            
        self._call_gazebo_async(std_srvs.srv.Empty.Request(), "reset_world")
        self._teleport_robot(s_x, s_y, s_yaw)
        
        # 兜底目标：优先用用户指定的，否则给默认值防止报错
        g_x, g_y = opts["goal_pos"] if opts["goal_pos"] else (0.0, 0.0)
        return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self._robot_urdf is None: raise ValueError("robot_urdf is required")
        
        # 1. 环境就绪检查
        if not self._wait_for_services(): return self._get_zero_obs(), {}
        self._clear_sensor_cache()
        
        # 2. 解析外部传入参数
        opts = self._parse_reset_options(options)
        
        # 3. 核心：生成并校验合法的任务(起点+终点)
        task = self._generate_valid_task(opts)
        if task is None:
            task = self._fallback_task(opts)
            
        # 4. 初始化本回合状态
        self.state["goal_x"], self.state["goal_y"] = task["g_x"], task["g_y"]
        self.state["step"] = 0
        self.state["last_action"] = np.zeros(2, dtype=np.float32)
        
        odom = self._get_odom_data()
        self.state["prev_dist"] = self._goal_dist(odom) if odom else 1.0
        
        return self._get_obs(), {
            "spawn_pos": (task["s_x"], task["s_y"]),
            "goal_pos": (task["g_x"], task["g_y"]),
            "initial_dist": math.hypot(task["s_x"] - task["g_x"], task["s_y"] - task["g_y"])
        }


    # ==================== Step 辅助函数 ====================
    def _get_step_data(self):
        return self._get_obs(), self._get_odom_data(), self._get_imu_data()

    def _check_termination(self, curr_dist: float) -> Tuple[bool, bool]:
        goal_reached = curr_dist < self.world["dist_to_goal_threshold"]
        collision = self._check_collision()
        return goal_reached, collision

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. 执行动作 & 等待物理演化
        self.send_velocity(float(action[0]), float(action[1]))
        time.sleep(self.world["step_duration"])

        # 2. 获取最新数据
        obs, odom, imu = self._get_step_data()
        if odom is None: return obs, -1.0, False, True, {"error": "odom unavailable"}

        # 3. 判定终止条件
        curr_dist = self._goal_dist(odom)
        goal_reached, collision = self._check_termination(curr_dist)

        # 4. 计算奖励
        reward = self._compute_reward(
            curr_dist=curr_dist, prev_dist=self.state["prev_dist"],
            goal_reached=goal_reached, collision=collision,
            action=action, odom=odom, imu=imu
        )

        # 5. 更新内部状态
        self.state["prev_dist"] = curr_dist
        self.state["last_action"] = action.copy()
        self.state["step"] += 1

        terminated = goal_reached or collision
        truncated = self.state["step"] >= self.world["episode_steps_max"]

        if terminated or truncated:
            self.send_velocity(0.0, 0.0)

        return obs, float(reward), terminated, truncated, {
            "collision": collision, "goal_reached": goal_reached,
            "final_dist": curr_dist, "episode_step": self.state["step"]
        }


    def run_episode(self, model, start_pos=None, goal_pos=None, deterministic=True, safe_threshold=None, skip_spawn_check=False, verbose=False):
        options = {}
        if start_pos is not None: options["start_pos"] = start_pos
        if goal_pos is not None: options["goal_pos"] = goal_pos
        if safe_threshold is not None: options["safe_threshold"] = safe_threshold
        if skip_spawn_check: options["skip_spawn_check"] = True
            
        obs, info = self.reset(options=options if options else None)
        
        if verbose:
            print(f"\n[Ep Start] Spawn:{info['spawn_pos']} | Goal:{info['goal_pos']} | Obs Shape:{obs.shape}")

        ep_reward = 0.0
        step = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.step(action)
            ep_reward += reward
            
            if verbose:
                action_np = np.array(action).flatten()
                print(f"  Step {step}: Act=[{action_np[0]:.3f}, {action_np[1]:.3f}], reward={ep_reward}")
                
            step += 1
            done = terminated or truncated
            
        return {
            "reward": ep_reward,
            "success": info.get('goal_reached', False),
            "steps": step,
            "final_dist": info.get('final_dist', -1.0)
        }

    def run_episodes(self, model, num_episodes=10, start_pos_list=None, goal_pos_list=None, deterministic=True, safe_threshold=None, verbose=False):
        results = []
        
        if start_pos_list is not None and goal_pos_list is not None:
            if len(start_pos_list) != len(goal_pos_list):
                raise ValueError("start_pos_list 和 goal_pos_list 长度必须一致！")
            num_episodes = len(start_pos_list)
        elif start_pos_list is not None:
            num_episodes = len(start_pos_list)
        elif goal_pos_list is not None:
            num_episodes = len(goal_pos_list)

        for i in range(num_episodes):
            s_pos = start_pos_list[i] if start_pos_list is not None else None
            g_pos = goal_pos_list[i] if goal_pos_list is not None else None
            
            skip_check = (s_pos is not None)
            
            res = self.run_episode(
                model=model, start_pos=s_pos, goal_pos=g_pos,
                deterministic=deterministic, safe_threshold=safe_threshold,
                skip_spawn_check=skip_check, verbose=verbose
            )
            results.append(res)
            
            # verbose 时让 run_episode 自己打印细节，这里只打印结局
            if verbose:
                status = "✅" if res["success"] else "❌"
                print(f"  -> Result: {status} | Final Reward: {res['reward']:.2f} | Steps: {res['steps']}")
            else:
                status = "✅" if res["success"] else "❌"
                print(f"\r[Episode {i+1}/{num_episodes}] {status} Reward: {res['reward']:.2f} | Steps: {res['steps']}", end="", flush=True)
                
        if not verbose: print()

        total_success = sum(1 for r in results if r["success"])
        return {
            "num_episodes": num_episodes,
            "success_rate": (total_success / num_episodes) * 100 if num_episodes > 0 else 0,
            "mean_reward": np.mean([r["reward"] for r in results]) if results else -1,
            "mean_steps": np.mean([r["steps"] for r in results]) if results else -1,
            "details": results
        }


    def close(self):
        self.send_velocity(0.0, 0.0)
        if rclpy.ok():
            self.destroy_node()


    # ==================== 内部逻辑 ====================

    def _sample_safe_position(self, allowed_zones: list = None) -> Tuple[float, float]:
        """
        从嵌套的安全区字典中随机采样坐标
        Args:
            allowed_zones: 允许的大区名称列表(如 ['zone_arena_bottom'])，None 表示全局随机
        """
        flat_pool = []
        if allowed_zones is None:
            for sub_zones in self.zones.values():
                flat_pool.extend(sub_zones)
        else:
            for zone_name in allowed_zones:
                if zone_name in self.zones:
                    flat_pool.extend(self.zones[zone_name])

        if not flat_pool:
            raise ValueError("No valid safe zones found for sampling!")

        zone = flat_pool[self.np_random.integers(len(flat_pool))]
        angle = self.np_random.uniform(0, 2 * math.pi)
        r = self.np_random.uniform(0, zone["r"])
        pos = (zone["cx"] + r * math.cos(angle), zone["cy"] + r * math.sin(angle))
        return pos

    @staticmethod
    def _euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _goal_dist(self, odom: Dict) -> float:
        return self._euclidean(
            (self.state["goal_x"], self.state["goal_y"]), 
            (odom["x"], odom["y"])
        )
        
    def _check_collision(self) -> bool:
        with self._lock:
            is_none = self._latest_scan is None
            age = time.time() - self._latest_scan_time if not is_none else 1.0
            
        if is_none or age > 0.25:
            return False
            
        laser_norm = self._get_laser_data()
        min_dist = float(np.min(laser_norm)) * self.bot["laser_range_max"]
        
        if min_dist < self.bot["proximity_to_collision_threshold"]:
            self.get_logger().debug(f"Collision! Min dist: {min_dist:.3f}m")
            return True
        return False

    def _check_spawn_pos(self, timeout=3.0, threshold=None):
        if threshold is None:
            threshold = self.bot["proximity_to_be_safe_min"] * 0.9
            
        timeout_deadline = time.time() + timeout
        while time.time() < timeout_deadline:
            laser_norm = self._get_laser_data()
            min_dist = float(np.min(laser_norm)) * self.bot["laser_range_max"]
            if min_dist > threshold:
                return True
            time.sleep(0.1)
            
        self.get_logger().warn(f"Spawn check failed. Min dist: {min_dist:.3f}m < Threshold: {threshold:.3f}m")
        return False

    def _check_goal_pos(self, goal_x: float, goal_y: float) -> bool:
        # 🔥 适配嵌套字典结构：遍历所有大区 -> 遍历大区内的子点
        for sub_zones in self.zones.values():
            for zone in sub_zones:
                dist_to_center = self._euclidean((goal_x, goal_y), (zone["cx"], zone["cy"]))
                if dist_to_center <= zone["r"]:
                    return True
        self.get_logger().warn(f"Goal ({goal_x:.2f}, {goal_y:.2f}) is outside all safe zones!")
        return False

    def _min_laser(self) -> float:
        with self._lock:
            if self._latest_scan is None: 
                return self.bot["laser_range_max"]
            ranges = np.array(self._latest_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=self.bot["laser_range_max"], posinf=self.bot["laser_range_max"])
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
            imu_acc = np.clip(np.array(imu['acc']) / self.bot["lin_acc_physics_max"], -1.0, 1.0)
            imu_gyro = np.clip(np.array(imu['gyro']) / self.bot["ang_vel_imu_physics_max"], -1.0, 1.0)
            imu_rp = np.clip(np.array(imu['rpy'][:2]) / (math.pi/4), -1.0, 1.0)

        if odom is not None:
            dx = self.state["goal_x"] - odom['x']
            dy = self.state["goal_y"] - odom['y']
            yaw = odom['yaw']
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
            goal_angle = math.atan2(local_y, local_x) / math.pi
            goal_dist = min(math.hypot(dx, dy) / self.world["dist_to_goal_clip_norm"], 1.0)
            odom_vel = np.clip(
                np.array([odom['vx'], odom['wz']]) / np.array([self.bot["lin_vel_max"], self.bot["ang_vel_max"]]), 
                -1.0, 1.0
            )

        last_act_norm = np.clip(
            np.array([self.state["last_action"][0], self.state["last_action"][1]]) / 
            np.array([self.bot["lin_vel_max"], self.bot["ang_vel_max"]]), 
            -1.0, 1.0
        )
        obs = np.concatenate([
            laser_norm, imu_acc, imu_gyro, imu_rp,
            [goal_angle], [goal_dist], odom_vel, last_act_norm
        ]).astype(np.float32)
        return obs
    # ==================== Reward 核心调度 ====================
    def _compute_reward(self, curr_dist, prev_dist, goal_reached, collision, action, odom, imu):
        # 1. 终止条件 (一票否决，直接返回)
        if goal_reached: return self.rew["reward_at_goal"]
        if collision:    return self.rew["penalty_at_collision"]

        # 2. 连续项叠加 (严格区分 reward_ 和 penalty_)
        reward  = self._rwd_distance(curr_dist, prev_dist)       # reward_ 前缀
        reward += self._rwd_heading(odom)                        # reward_ 前缀
        reward += self._rwd_safety()                             # penalty_ 前缀
        reward += self._rwd_stability(imu)                       # penalty_ 前缀
        reward += self._rwd_stuck(action, odom)                  # penalty_ 前缀
        reward += self._rwd_smoothness(action)                   # penalty_ 前缀
        reward += self.rew["penalty_elapsing_time"]              # penalty_ 前缀 (常驻)
        
        return reward

    # ==================== Reward 维度拆解 ====================
    
    # --- 🟩 正向奖励区 (基于 reward_ 前缀) ---
    # def _rwd_distance(self, curr_dist, prev_dist) -> float:
    #     dist_delta = prev_dist - curr_dist  # 靠近为正，远离为负
    #     return dist_delta * self.rew["reward_factor_approaching_goal"]
    def _rwd_distance(self, curr_dist, prev_dist) -> float:
        dist_delta = prev_dist - curr_dist  
        # 🔥 修复：忽略极微小的距离变化（比如小于1厘米），判定为雷达噪点
        if abs(dist_delta) < 0.01:
            return 0.0
        return dist_delta * self.rew["reward_factor_approaching_goal"]

    # def _rwd_heading(self, odom) -> float:
    #     yaw = odom['yaw']
    #     target_angle = math.atan2(self.state["goal_y"] - odom['y'], self.state["goal_x"] - odom['x'])
    #     yaw_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
    #     heading_factor = math.cos(yaw_error)  # 朝向好为正，背对为负
        
    #     if abs(odom['vx']) > 0.01:
    #         return heading_factor * self.rew["reward_good_orientation"]
    #     return 0.0
    #     # return heading_factor * self.rew["reward_good_orientation"] * abs(odom['vx'])
    def _rwd_heading(self, odom) -> float:
        yaw = odom['yaw']
        target_angle = math.atan2(self.state["goal_y"] - odom['y'], self.state["goal_x"] - odom['x'])
        yaw_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
        heading_factor = math.cos(yaw_error)  
        
        # 🔥 修复：硬截断。只有真正跑起来（速度大于有效阈值），才给满额朝向分
        if abs(odom['vx']) > 0.05:  
            return heading_factor * self.rew["reward_good_orientation"] # 不打折，直接给满
        return 0.0

    # --- 🟥 负向惩罚区 (基于 penalty_ 前缀，YAML里已带负号) ---
    def _rwd_safety(self) -> float:
        min_dist = self._min_laser()
        safe_min = self.bot["proximity_to_be_safe_min"]
        if min_dist < safe_min:
            # (负数) * (0~1的比例) = 负惩罚
            return self.rew["penalty_in_safe_proximity"] * ((safe_min - min_dist) / safe_min)**2
        return 0.0

    def _rwd_stability(self, imu) -> float:
        if imu is None: return 0.0
        roll, pitch = imu['rpy']
        tilt_factor = (roll**2 + pitch**2) / ( (math.pi/6)**2 )
        # 🔥 修复：去掉了多余的 -abs()，直接 (负数) * (0~1的比例) = 负惩罚
        return self.rew["penalty_instability"] * min(tilt_factor, 1.0)

    def _rwd_stuck(self, action, odom) -> float:
        vel_error = abs(action[0]) - abs(odom['vx'])
        if vel_error > self.bot["lin_vel_stuck_threshold"]: 
             # (负数) * (正数比例) = 负惩罚
             return self.rew["penalty_stuck"] * (vel_error / self.bot["lin_vel_max"])
        return 0.0

    def _rwd_smoothness(self, action) -> float:
        diff_lin = (action[0] - self.state["last_action"][0]) / self.bot["lin_vel_max"]
        diff_ang = (action[1] - self.state["last_action"][1]) / self.bot["ang_vel_max"]
        smoothness_penalty = np.sqrt(diff_lin**2 + diff_ang**2)
        return self.rew["penalty_action_smoothness"] * smoothness_penalty


    # def _compute_reward(self, curr_dist, prev_dist, goal_reached, collision, action, odom, imu):
    #     reward = 0.0
        
    #     # 1. 终止条件奖励
    #     if goal_reached: return self.reward_at_goal
    #     if collision:    return self.penalty_at_collision

    #     # 2. 距离变化奖励
    #     dist_delta = prev_dist - curr_dist
    #     reward += dist_delta * self.reward_factor_approaching_goal

    #     # 3. 朝向奖励
    #     yaw = odom['yaw']
    #     target_angle = math.atan2(self.goal_y - odom['y'], self.goal_x - odom['x'])
    #     yaw_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
    #     heading_factor = math.cos(yaw_error)
        
    #     if abs(odom['vx']) > 0.01:
    #         reward += heading_factor * self.reward_good_orientation

    #     # 4. 安全惩罚
    #     min_dist = self._min_laser()
    #     if min_dist < self.proximity_to_be_safe_min:
    #         reward += self.penalty_in_safe_proximity * ((self.proximity_to_be_safe_min - min_dist) / self.proximity_to_be_safe_min)**2

    #     # 5. 稳定性惩罚 (基于 IMU 的 Roll 和 Pitch)
    #     if imu is not None:
    #         roll, pitch = imu['rpy']
    #         tilt_factor = (roll**2 + pitch**2) / ( (math.pi/6)**2 )
    #         reward -= abs(self.penalty_instability) * min(tilt_factor, 1.0)

    #     # 6. 卡住惩罚 (优化版：检测指令与实际速度的偏差)
    #     # 如果指令速度很大，但实际速度很小，说明发生了打滑或卡死
    #     vel_error = abs(action[0]) - abs(odom['vx'])
    #     if vel_error > self.lin_vel_stuck_threshold: # 允许 0.1 的控制误差
    #          reward += self.penalty_stuck * (vel_error / self.lin_vel_max)
    #         # 这个逻辑的初衷是：我给了很大的油门，但车没动（比如撞墙卡死了或者打滑了），所以要惩罚。

    #         # 需要注意的物理现象：
    #         # TurtleBot3 是有加速度的。如果你上一秒给的速度是 0，这一秒突然给 0.2 m/s 的指令，在 time.sleep(step_duration) 这极短的时间内（比如 0.1秒），底盘的 odom['vx'] 可能只到了 0.05 m/s。
    #         # 这时候 vel_error = 0.2 - 0.05 = 0.15，会误触发卡住惩罚。

    #         # 对策：
    #         # 不需要改代码，但在你后面调参时，self.penalty_stuck 的绝对值一定要设得非常小（比如 -0.1 甚至 -0.05），让它成为一个“长期卡死时的累积惩罚”，而不是“一瞬间没加速就狠罚”。只要它远小于正常靠近目标的奖励，网络就会学到容忍启动时的短暂延迟。


    #     # 7. 动作平滑性惩罚 (新增)
    #     # 1. 计算变化量
    #     diff_lin = action[0] - self.last_action[0]
    #     diff_ang = action[1] - self.last_action[1]

    #     # 2. 归一化 (变化量 / 最大速度 = 变化百分比)
    #     norm_diff_lin = diff_lin / self.lin_vel_max
    #     norm_diff_ang = diff_ang / self.ang_vel_max

    #     # 3. 计算加权欧氏距离 (此时两者量纲一致，都是 0~1 的比例)
    #     smoothness_penalty = np.sqrt(norm_diff_lin**2 + norm_diff_ang**2)

    #     # 4. 施加惩罚
    #     reward += self.penalty_action_smoothness * smoothness_penalty
    #     # 8. 时间惩罚
    #     reward += self.penalty_elapsing_time
        
    #     return reward

class NaviMissionAssigner() :
    pass




# # ==========================================
# # 你的训练脚本主流程
# # ==========================================
# if __name__ == '__main__':
#     # 1. 在创建任何环境之前，先拿到 URDF (只查一次网络)
#     my_urdf = fetch_tb3_urdf()
    
#     # 2. 把 URDF 塞进配置字典里
#     env_config = {
#         "robot_urdf": my_urdf,  # 注入进去
#         # ... 其他你需要的 reward、environment 配置 ...
#     }
    
#     # 3. 无论你是单环境还是多环境，都这么写
#     from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
#     def make_env():
#         def _init():
#             # 此时环境内部不再去查 ROS2，直接用传进来的 my_urdf
#             return TurtleBot3NavEnv(config=env_config) 
#         return _init

#     # 单进程训练
#     # env = DummyVecEnv([make_env()])
    
#     # 多进程训练 (以后直接改这里，绝对不会有端口冲突和查询风暴)
#     # env = SubprocVecEnv([make_env() for _ in range(4)])
    
#     # ... 接下来写 model.learn() ...













# import math
# import time
# import threading
# import logging
# from typing import Optional, Dict, Any, Tuple

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# # ROS2 相关导入
# import rclpy
# import rclpy.parameter
# from rcl_interfaces.srv import GetParameters
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from sensor_msgs.msg import LaserScan, Imu
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist, PoseStamped

# from std_srvs.srv import Empty
# from gazebo_msgs.srv import SetEntityState
# from gazebo_msgs.srv import DeleteEntity, SpawnEntity
# from geometry_msgs.msg import Pose

# import std_srvs.srv
# from gazebo_msgs.srv import SetEntityState
# import math

# logger = logging.getLogger(__name__)


# def fetch_tb3_urdf():
#     """在主进程中提前拉取完整的 URDF，子进程直接 pickle 继承，不再查 ROS2"""
#     if not rclpy.ok():
#         rclpy.init()

#     temp_node = rclpy.create_node('urdf_prefetcher')
#     urdf_string = None

#     # ====== 第一步：找到目标节点名 ======
#     target_node = None
#     for attempt in range(50):  # 最多等5秒
#         nodes = temp_node.get_node_names()
#         for n in nodes:
#             if 'robot_state_publisher' in n or 'state_publisher' in n:
#                 target_node = n
#                 break
#         if target_node:
#             break
#         time.sleep(0.1)

#     if not target_node:
#         all_nodes = temp_node.get_node_names()
#         temp_node.destroy_node()
#         raise RuntimeError(
#             f"No robot_state_publisher found! Visible nodes: {all_nodes}")

#     print(f"[Prefetcher] Found node: {target_node}")

#     # ====== 第二步：等 service 可用（这一步才是你之前的盲区） ======
#     service_name = f'{target_node}/get_parameters'
#     param_client = temp_node.create_client(GetParameters, service_name)

#     print(f"[Prefetcher] Waiting for service {service_name} ...")
#     if not param_client.wait_for_service(timeout_sec=5.0):
#         temp_node.destroy_client(param_client)
#         temp_node.destroy_node()
#         raise RuntimeError(
#             f"Service {service_name} not available after 5s! "
#             "DDS service discovery may need more time.")

#     print(f"[Prefetcher] Service ready, calling...")

#     # ====== 第三步：调用服务拿 URDF ======
#     req = GetParameters.Request()
#     req.names = ['robot_description']
#     future = param_client.call_async(req)

#     start = time.time()
#     while not future.done() and time.time() - start < 10.0:
#         rclpy.spin_once(temp_node, timeout_sec=0.05)  # ← 加这一行来帮忙收快递！

#     if future.done() and future.result() is not None:
#         values = future.result().values
#         if values:
#             urdf_string = values[0].string_value

#     temp_node.destroy_client(param_client)
#     temp_node.destroy_node()

#     if urdf_string is None:
#         raise RuntimeError("Service call succeeded but robot_description was empty!")

#     print(f"[Prefetcher] Successfully got URDF ({len(urdf_string)} chars)")
#     return urdf_string


# class TurtleBot3NavEnv(gym.Env, Node):
#     """
#     直接继承 rclpy.Node，去除外部接口依赖。
#     所有配置参数均严格对应 config 键名。
#     """

#     metadata = {"render_modes": []}
    
#     SAFE_ZONES = {}
#     # dafault
#     SAFE_ZONES["default"] = [
#         # 角斗场底部
#         {"cx": 0, "cy":  0, "r": 1.},
#     ]
    
#     # world
#     SAFE_ZONES["ttb3_world"] = [
#         # 角斗场底部
#         {"cx":  -0.5, "cy":  -1.75, "r": 0.75},
#         {"cx":  0.5, "cy":  -1.75, "r": 0.75}, 
#         # 角斗场正左、左上左下；
#         {"cx":  -2, "cy":  0, "r": 0.6},
#         {"cx":  -1.75, "cy":  0.75, "r": 0.6},
#         {"cx":  -1.75, "cy":  -0.75, "r": 0.6},
#         # 角斗场右上右下
#         {"cx":  1.75, "cy":  0.75, "r": 0.6},
#         {"cx":  1.75, "cy":  -0.75, "r": 0.6},
#         # 角斗场顶部
#         {"cx":  -0.5, "cy":  1.75, "r": 0.75},
#         {"cx":  0.5, "cy":  1.75, "r": 0.75},
#         # 角斗场柱田
#         {"cx":  -0.5, "cy":  -0.5, "r": 0.6},
#         {"cx":  0.5, "cy":  -0.5, "r": 0.6},
#         {"cx":  -0.5, "cy":  0.5, "r": 0.6},
#         {"cx":  0.5, "cy":  0.5, "r": 0.6}, 
#     ]

#     # house
#     SAFE_ZONES["ttb3_house"] = [
#         # 小垃圾房垃圾桶旁边
#         {"cx":  1.2, "cy":  1.5, "r": 0.4},
#         # 单腿茶几房茶几底下
#         {"cx":  6.5, "cy":  -4.2, "r": 0.9},
#         # 书柜红墙房右下角
#         {"cx":  -6, "cy":  2, "r": 0.8},
#         # 书柜木墙房左下角
#         {"cx":  7.5, "cy":  -3, "r": 0.9},
#         # 中央房木桌正中
#         {"cx":  -2.6, "cy":  2.5, "r": 0.6},
#         # 右上房铁桌正中
#         {"cx":  4.9, "cy":  3, "r": 1},
#         # 屋外：正门口+左右下两翼
#         {"cx":  -4, "cy":  -3, "r": 2},
#         {"cx":  2, "cy":  -4, "r": 2},
#         {"cx":  -9, "cy":  -7, "r": 1},
#         {"cx":  6, "cy":  -6, "r": 0.8},
#         # # 屋外：屋后
#         # {"cx":  -4, "cy":  8, "r": 2},
#         # {"cx":  4, "cy":  8, "r": 2},
#         # 屋外：屋右
#         # {"cx":  9, "cy":  8, "r": 1},
#         {"cx":  9, "cy":  -4, "r": 1},
#         # 屋外：屋左
#         # {"cx":  -9, "cy":  8, "r": 1},
#         {"cx":  -9, "cy":  -3, "r": 1},
#     ]

#     def __init__(self, robot_urdf: Optional[str] = None, config: Optional[Dict[str, Any]] = None, bot_name="burger", world="ttb3_world"):
#         # 初始化 Gym 环境
#         gym.Env.__init__(self)
            
#         self.bot_name = bot_name
#         self._robot_urdf = robot_urdf
        
#         # 安全区域
#         try:
#             self.safe_zones = self.SAFE_ZONES[world]
#             print(f"[World] Now bot {self.bot_name} in the world {world}!")
#         except KeyError:
#             valid_worlds = list(self.SAFE_ZONES.keys())
#             raise KeyError(f"无效的 world: {world}，请使用 {valid_worlds}") from None
#         # finally :
#         #     self.safe_zones = self.SAFE_ZONES["default"]

#         # 默认配置结构
#         default_config = {
#             "environment": {
#                 "laser_range_max": 3.5,
#                 "laser_beams_num": 24,
#                 "lin_vel_max": 0.22,
#                 "ang_vel_max": 1.5,
#                 "episode_steps_max": 500,
#                 "step_duration": 0.1,
#                 "dist_to_goal_threshold": 0.35,      # (结合上次讨论改为了0.35)
#                 "dist_to_goal_gen_min": 1.0,         # <--- 补上
#                 "dist_to_goal_clip_norm": 5.0,
#                 "proximity_to_collision_threshold": 0.11, # (结合上次改为了0.11)
#                 "proximity_to_be_safe_min": 0.30,     # (结合上次改为了0.30)
#                 "lin_vel_stuck_threshold": 0.12,      # <--- 补上
#                 "lin_acc_physics_max": 2.0,           # <--- 补上
#                 "ang_vel_imu_physics_max": 3.0,       # <--- 补上
#             },
#             "reward": {
#                 "reward_at_goal": 300.0,
#                 "penalty_at_collision": -300.0,
#                 "reward_factor_approaching_goal": 5.0,
#                 "penalty_elapsing_time": -0.5,
#                 "penalty_stuck": -2.0,
#                 "reward_good_orientation": 2.0,
#                 "penalty_in_safe_proximity": -1.0,
#                 "penalty_instability": -2.0,
#                 "penalty_action_smoothness": -0.5, # 新增配置项
#             }
#         }
        
#         # 合并用户配置
#         self.config = default_config.copy()
#         if config:
#             for key, value in config.items():
#                 if isinstance(value, dict) and key in self.config:
#                     self.config[key].update(value)
#                 else:
#                     self.config[key] = value

#         # --- 提取环境参数 (变量名严格对应 config 键名) ---
#         cfg_env = self.config["environment"]
#         self.laser_range_max = cfg_env.get("laser_range_max")
#         self.laser_beams_num = cfg_env.get("laser_beams_num")
#         self.lin_vel_max = cfg_env.get("lin_vel_max")
#         self.ang_vel_max = cfg_env.get("ang_vel_max")
#         self.episode_steps_max = cfg_env.get("episode_steps_max")
#         self.step_duration = cfg_env.get("step_duration")
#         self.dist_to_goal_threshold = cfg_env.get("dist_to_goal_threshold")
#         self.dist_to_goal_clip_norm = cfg_env.get("dist_to_goal_clip_norm")
#         self.proximity_to_collision_threshold = cfg_env.get("proximity_to_collision_threshold")
#         self.proximity_to_be_safe_min = cfg_env.get("proximity_to_be_safe_min")
#         self.lin_vel_stuck_threshold = cfg_env.get("lin_vel_stuck_threshold")
#         # 物理极限 (内部用)
#         self.lin_acc_physics_max = cfg_env.get("lin_acc_physics_max")
#         self.ang_vel_imu_physics_max = cfg_env.get("ang_vel_imu_physics_max")        
#         # 衍生参数
#         self.dist_to_goal_gen_min = cfg_env.get("dist_to_goal_gen_min")
        
        
#         # --- 提取奖励参数 (变量名严格对应 config 键名) ---
#         cfg_rew = self.config["reward"]
#         self.reward_at_goal = cfg_rew.get("reward_at_goal")
#         self.penalty_at_collision = cfg_rew.get("penalty_at_collision")
#         self.reward_factor_approaching_goal = cfg_rew.get("reward_factor_approaching_goal")
#         self.penalty_elapsing_time = cfg_rew.get("penalty_elapsing_time")
#         self.penalty_stuck = cfg_rew.get("penalty_stuck")
#         self.reward_good_orientation = cfg_rew.get("reward_good_orientation")
#         self.penalty_in_safe_proximity = cfg_rew.get("penalty_in_safe_proximity")
#         self.penalty_instability = cfg_rew.get("penalty_instability")
#         self.penalty_action_smoothness = cfg_rew.get("penalty_action_smoothness")

#         # --- 初始化 ROS2 Node ---
#         if not rclpy.ok():
#             rclpy.init()
        
#         Node.__init__(self, 'turtlebot3_gym_env')
        
#         # --- 数据缓存与锁 (线程安全) ---
#         self._lock = threading.Lock()
#         self._latest_scan: Optional[LaserScan] = None
#         self._latest_imu: Optional[Imu] = None
#         self._latest_odom: Optional[Odometry] = None
#         self._latest_scan_time: float = 0.0

#         # --- 订阅者 (使用 Best Effort 适配 Gazebo) ---
#         qos_profile = QoSProfile(
#             # reliability=ReliabilityPolicy.BEST_EFFORT,
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )

#         self.scan_sub = self.create_subscription(
#             LaserScan, '/scan', self._scan_cb, qos_profile)
#         self.imu_sub = self.create_subscription(
#             Imu, '/imu', self._imu_cb, qos_profile)
#         # self.odom_sub = self.create_subscription(
#         #     Odometry, '/odometry/filtered', self._odom_cb, qos_profile)
#         self.odom_sub = self.create_subscription(
#             Odometry, '/odom', self._odom_cb, qos_profile)

#         # --- 发布者 ---
#         self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

#         # --- ROS2 Spin 线程 ---
#         self._spin_thread = threading.Thread(target=self._spin, daemon=True)
#         self._spin_thread.start()

#         # --- 空间定义 ---
#         # obs_size 计算细节: 24(Lidar) + 3(ImuAcc) + 3(ImuGyro) + 2(ImuRP) + 1(GoalAng) + 1(GoalDist) + 2(OdomVel) + 2(LastAct)
#         self._obs_size = 38
#         self.observation_space = spaces.Box(
#             low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
#         )
#         self.action_space = spaces.Box(
#             low=np.array([-self.lin_vel_max, -self.ang_vel_max], dtype=np.float32),
#             high=np.array([ self.lin_vel_max,  self.ang_vel_max], dtype=np.float32),
#             dtype=np.float32,
#         )

#         # --- 内部状态 ---
#         self.goal_x = 0.0
#         self.goal_y = 0.0
#         self.prev_dist = 0.0 # 上一时刻到目标的距离
#         self.episode_step = 0
#         # 关键初始化：机器人初始状态为静止
#         self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        
#         # # ====== 新增：创建独立的服务调用节点（避免死锁） ======
#         # from rclpy.executors import SingleThreadedExecutor
#         # self._reset_node = rclpy.create_node('reset_service_node')
#         # self._reset_client = self._reset_node.create_client(
#         #     SetEntityState, '/set_entity_state')
        
#         # self._reset_executor = SingleThreadedExecutor()
#         # self._reset_executor.add_node(self._reset_node)
#         # self._reset_thread = threading.Thread(
#         #     target=self._reset_executor.spin, daemon=True)
#         # self._reset_thread.start()
#         # self.get_logger().info("Reset service thread started.")
#         # # ====== 新增结束 ======
        
#         # --- Reset 服务客户端 ---
#         self._reset_world_client = self.create_client(
#             std_srvs.srv.Empty, '/reset_world')
#         self._reset_client = self.create_client(
#             SetEntityState, '/set_entity_state')

#         # --- 删除/重生服务（彻底物理重置） ---
#         self._delete_client = self.create_client(
#             DeleteEntity, '/delete_entity')
#         self._spawn_client = self.create_client(
#             SpawnEntity, '/spawn_entity')
        
        
#         # # --- 读取机器人 URDF（从 ROS2 参数服务器） ---
#         # self._robot_urdf = None
        
#         # nodes = self.get_node_names()
#         # for node_name in nodes:
#         #     if 'robot_state_publisher' in node_name or 'state_publisher' in node_name:
#         #         param_client = None
#         #         try:
#         #             param_client = self.create_client(
#         #                 GetParameters, f'{node_name}/get_parameters')
#         #             if param_client.wait_for_service(timeout_sec=3.0):
#         #                 req = GetParameters.Request()
#         #                 req.names = ['robot_description']
#         #                 future = param_client.call_async(req)
#         #                 start = time.time()
#         #                 while not future.done() and time.time() - start < 5.0:
#         #                     time.sleep(0.05)
#         #                 if future.done() and future.result() is not None:
#         #                     values = future.result().values
#         #                     if values:
#         #                         self._robot_urdf = values[0].string_value
#         #                         self.get_logger().info(
#         #                             f"Got robot_description from {node_name} "
#         #                             f"({len(self._robot_urdf)} chars)")
#         #                         break
#         #         except Exception as e:
#         #             self.get_logger().warn(f"Failed to get URDF from {node_name}: {e}")
#         #         finally:
#         #             if param_client is not None:
#         #                 self.destroy_client(param_client)

#         # if self._robot_urdf is None:
#         #     self.get_logger().error(
#         #         "Could not get robot URDF from parameter server! "
#         #         "Make sure robot_state_publisher is running in Docker.")
#         # # --- 读取机器人 URDF ---

#         self.print_config()
#         logger.info("TurtleBot3HouseEnv initialized.")

#     def print_config(self) :
#         # ==================== 打印实际加载的配置 ====================
#         self.get_logger().info("=" * 60)
#         self.get_logger().info("  TurtleBot3NavEnv 配置加载完毕，实际参数如下：")
#         self.get_logger().info("=" * 60)
#         self.get_logger().info("  [环境参数]")
#         self.get_logger().info(f"    laser_range_max           = {self.laser_range_max}")
#         self.get_logger().info(f"    laser_beams_num           = {self.laser_beams_num}")
#         self.get_logger().info(f"    lin_vel_max               = {self.lin_vel_max}")
#         self.get_logger().info(f"    ang_vel_max               = {self.ang_vel_max}")
#         self.get_logger().info(f"    episode_steps_max         = {self.episode_steps_max}")
#         self.get_logger().info(f"    step_duration             = {self.step_duration}")
#         self.get_logger().info(f"    dist_to_goal_threshold    = {self.dist_to_goal_threshold}")
#         self.get_logger().info(f"    dist_to_goal_gen_min      = {self.dist_to_goal_gen_min}")
#         self.get_logger().info(f"    dist_to_goal_clip_norm    = {self.dist_to_goal_clip_norm}")
#         self.get_logger().info(f"    proximity_to_collision    = {self.proximity_to_collision_threshold}")
#         self.get_logger().info(f"    proximity_to_be_safe_min  = {self.proximity_to_be_safe_min}")
#         self.get_logger().info(f"    lin_vel_stuck_threshold   = {self.lin_vel_stuck_threshold}")
#         self.get_logger().info(f"    lin_acc_physics_max       = {self.lin_acc_physics_max}")
#         self.get_logger().info(f"    ang_vel_imu_physics_max   = {self.ang_vel_imu_physics_max}")
#         self.get_logger().info("  [奖励参数]")
#         self.get_logger().info(f"    reward_at_goal            = {self.reward_at_goal}")
#         self.get_logger().info(f"    penalty_at_collision      = {self.penalty_at_collision}")
#         self.get_logger().info(f"    reward_factor_approaching = {self.reward_factor_approaching_goal}")
#         self.get_logger().info(f"    penalty_elapsing_time     = {self.penalty_elapsing_time}")
#         self.get_logger().info(f"    penalty_stuck             = {self.penalty_stuck}")
#         self.get_logger().info(f"    reward_good_orientation   = {self.reward_good_orientation}")
#         self.get_logger().info(f"    penalty_in_safe_proximity = {self.penalty_in_safe_proximity}")
#         self.get_logger().info(f"    penalty_instability       = {self.penalty_instability}")
#         self.get_logger().info(f"    penalty_action_smoothness = {self.penalty_action_smoothness}")
#         self.get_logger().info("=" * 60)

#     # ==================== ROS2 回调与工具 ====================

#     def _spin(self):
#         while rclpy.ok():
#             rclpy.spin_once(self, timeout_sec=0.01)

#     # def _scan_cb(self, msg: LaserScan):
#     #     with self._lock:
#     #         self._latest_scan = msg
#     def _scan_cb(self, msg: LaserScan):
#         with self._lock:
#             # 【核心修复】滤掉打到自身轮子/底盘的噪点
#             # 正确做法：遍历原地修改底层数组，而不是生成新 list 覆盖
#             for i in range(len(msg.ranges)):
#                 if msg.ranges[i] < 0.15:
#                     msg.ranges[i] = msg.range_max
                    
#             self._latest_scan = msg
#             self._latest_scan_time = time.time()

#     def _imu_cb(self, msg: Imu):
#         with self._lock:
#             self._latest_imu = msg

#     def _odom_cb(self, msg: Odometry):
#         with self._lock:
#             self._latest_odom = msg

#     def _get_odom_data(self) -> Optional[Dict]:
#         with self._lock:
#             if self._latest_odom is None: return None
#             p = self._latest_odom.pose.pose.position
#             o = self._latest_odom.pose.pose.orientation
#             t = self._latest_odom.twist.twist
            
#             siny_cosp = 2 * (o.w * o.z + o.x * o.y)
#             cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
#             yaw = math.atan2(siny_cosp, cosy_cosp)

#             return {
#                 "x": p.x, "y": p.y, "yaw": yaw,
#                 "vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z
#             }

#     def _get_imu_data(self) -> Optional[Dict]:
#         with self._lock:
#             if self._latest_imu is None: return None
#             a = self._latest_imu.linear_acceleration
#             g = self._latest_imu.angular_velocity
#             o = self._latest_imu.orientation
            
#             sinr_cosp = 2 * (o.w * o.x + o.y * o.z)
#             cosr_cosp = 1 - 2 * (o.x * o.x + o.y * o.y)
#             roll = math.atan2(sinr_cosp, cosr_cosp)
            
#             sinp = 2 * (o.w * o.y - o.z * o.x)
#             pitch = math.asin(np.clip(sinp, -1.0, 1.0))

#             return {
#                 "acc": [a.x, a.y, a.z],
#                 "gyro": [g.x, g.y, g.z],
#                 "rpy": [roll, pitch]
#             }

#     def _get_laser_data(self) -> np.ndarray:
#         with self._lock:
#             if self._latest_scan is None:
#                 return np.ones(self.laser_beams_num, dtype=np.float32)
            
#             ranges = np.array(self._latest_scan.ranges)
#             ranges = np.nan_to_num(ranges, nan=self.laser_range_max, 
#                                    posinf=self.laser_range_max, neginf=0.0)
#             ranges = np.clip(ranges, 0, self.laser_range_max)
            
#             n = len(ranges)
#             step = max(1, n // self.laser_beams_num)
#             sampled = np.array([np.min(ranges[i:i+step]) for i in range(0, n, step)])
            
#             if len(sampled) > self.laser_beams_num:
#                 sampled = sampled[:self.laser_beams_num]
#             elif len(sampled) < self.laser_beams_num:
#                 sampled = np.pad(sampled, (0, self.laser_beams_num - len(sampled)), 
#                                  'constant', constant_values=self.laser_range_max)
            
#             return sampled / self.laser_range_max

#     def send_velocity(self, lin_x, ang_z):
#         msg = Twist()
#         msg.linear.x = float(lin_x)
#         msg.angular.z = float(ang_z)
#         self.vel_pub.publish(msg)

#     def _get_zero_obs(self):
#         """
#         返回一个全零观测，用于环境异常时的兜底。
#         """
#         import numpy as np
#         return np.zeros(self.observation_space.shape, dtype=np.float32)

#     # ==================== Gym 接口 ====================
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed, options=options)
        
#         if self._robot_urdf is None:
#             self.get_logger().error("Fatal: robot_urdf string was not provided during Env initialization!")
#             raise ValueError("robot_urdf is required")
 
#         if not hasattr(self, '_services_ready') or not self._services_ready:
#             self.get_logger().info("Waiting for Gazebo reset services...")
#             try:
#                 if not self._reset_client.wait_for_service(timeout_sec=5.0):
#                     raise RuntimeError("SetEntityState service not found!")
#                 if not self._reset_world_client.wait_for_service(timeout_sec=5.0):
#                     raise RuntimeError("ResetWorld service not found!")
#                 self._services_ready = True
#                 self.get_logger().info("Gazebo services are ready!")
#             except Exception as e:
#                 self.get_logger().error(f"Service check failed: {e}")
#                 return self._get_zero_obs(), {}

#         with self._lock:
#             self._latest_scan = None
#             self._latest_scan_time = 0.0

#         # 解析参数
#         custom_start = options.get("start_pos", None) if options else None
#         custom_goal = options.get("goal_pos", None) if options else None
#         safe_threshold = options.get("safe_threshold", None) if options else None
#         skip_spawn_check = options.get("skip_spawn_check", False) if options else False

#         if custom_goal is not None:
#             self.goal_x, self.goal_y = custom_goal[0], custom_goal[1]

#         max_retries = 5
#         success = False
#         use_custom_start = custom_start is not None

#         for attempt in range(max_retries):
#             # 确定起点
#             if use_custom_start:
#                 spawn_x, spawn_y, spawn_yaw = custom_start[0], custom_start[1], custom_start[2]
#             else:
#                 spawn_x, spawn_y = self._sample_safe_position()
#                 spawn_yaw = self.np_random.uniform(-math.pi, math.pi)

#             # 确定终点
#             if custom_goal is None:
#                 self.goal_x, self.goal_y = self._sample_safe_position()

#             # ==========================================
#             # 校验 1: 起点终点距离是否足够远 (仅限随机生成时校验)
#             # ==========================================
#             if not use_custom_start and custom_goal is None:
#                 dist = self._euclidean((spawn_x, spawn_y), (self.goal_x, self.goal_y))
#                 if dist < self.dist_to_goal_gen_min:
#                     self.get_logger().warn(f"Goal too close to spawn ({dist:.2f}m < {self.dist_to_goal_gen_min}m), retrying...")
#                     continue  # 距离太近，直接进入下一次重试，不浪费时间去调Gazebo

#             # --- Gazebo 服务调用 ---
#             future = self._reset_world_client.call_async(std_srvs.srv.Empty.Request())
#             start = time.time()
#             while not future.done() and time.time() - start < 3.0: time.sleep(0.05)

#             req = SetEntityState.Request()
#             req.state.name = self.bot_name
#             req.state.pose.position.x = float(spawn_x)
#             req.state.pose.position.y = float(spawn_y)
#             req.state.pose.position.z = 0.0
#             req.state.pose.orientation.z = math.sin(spawn_yaw / 2.0)
#             req.state.pose.orientation.w = math.cos(spawn_yaw / 2.0)
#             req.state.reference_frame = "world"
#             future = self._reset_client.call_async(req)
#             start = time.time()
#             while not future.done() and time.time() - start < 3.0: time.sleep(0.05)

#             # ==========================================
#             # 校验 2 & 3: 激光防撞检查 & 终点本身安全性
#             # ==========================================
#             time.sleep(0.5)
#             spawn_is_safe = True
#             goal_is_safe = True

#             if not skip_spawn_check:
#                 spawn_is_safe = self._check_spawn_pos(timeout=3.0, threshold=safe_threshold)
            
#             # 只在随机生成终点时，检查终点本身是否合法
#             if custom_goal is None:
#                 goal_is_safe = self._check_goal_pos(self.goal_x, self.goal_y)

#             if spawn_is_safe and goal_is_safe:
#                 success = True
#                 break
#             else:
#                 if not use_custom_start:
#                     self.get_logger().warn(f"Position check failed, retrying... ({attempt+1}/{max_retries})")

#         # 兜底逻辑
#         if not success:
#             if not use_custom_start:
#                 self.get_logger().warn("All retries failed! Fallback to (0.5, 0.5).")
#                 req = SetEntityState.Request()
#                 req.state.name = "burger"
#                 req.state.pose.position.x = 0.5
#                 req.state.pose.position.y = 0.5
#                 req.state.pose.position.z = 0.0
#                 req.state.pose.orientation.w = 1.0
#                 req.state.reference_frame = "world"
#                 future = self._reset_client.call_async(req)
#                 start = time.time()
#                 while not future.done() and time.time() - start < 3.0: time.sleep(0.05)
#             else:
#                 self.get_logger().error(f"Failed to set custom start pose: {custom_start}")

#         obs = self._get_obs()
#         self.prev_dist = self._goal_dist(self._get_odom_data()) if self._get_odom_data() else 1.0
#         self.episode_step = 0
#         self.last_action = np.zeros(2, dtype=np.float32)
        
#         info = {
#             "spawn_pos": (spawn_x, spawn_y),
#             "goal_pos": (self.goal_x, self.goal_y),
#             "initial_dist": math.hypot(spawn_x - self.goal_x, spawn_y - self.goal_y)
#         }
#         return obs, info


#     # def reset(self, seed=None, options=None):
#     #     super().reset(seed=seed, options=options)
        
#     #     if self._robot_urdf is None:
#     #         self.get_logger().error("Fatal: robot_urdf string was not provided during Env initialization!")
#     #         raise ValueError("robot_urdf is required")
 
#     #     # --- 只在第一次 reset 时等待服务 ---
#     #     if not hasattr(self, '_services_ready') or not self._services_ready:
#     #         self.get_logger().info("Waiting for Gazebo reset services...")
#     #         try:
#     #             if not self._reset_client.wait_for_service(timeout_sec=5.0):
#     #                 raise RuntimeError("SetEntityState service not found!")
#     #             if not self._reset_world_client.wait_for_service(timeout_sec=5.0):
#     #                 raise RuntimeError("ResetWorld service not found!")
                
#     #             self._services_ready = True
#     #             self.get_logger().info("Gazebo services are ready!")
#     #         except Exception as e:
#     #             self.get_logger().error(f"Service check failed: {e}")
#     #             return self._get_zero_obs(), {}

#     #     # 清空内部缓存
#     #     with self._lock:
#     #         self._latest_scan = None
#     #         self._latest_scan_time = 0.0

#     #     max_retries = 5
#     #     success = False

#     #     for attempt in range(max_retries):
#     #         spawn_x, spawn_y = self._sample_safe_position()
#     #         spawn_yaw = self.np_random.uniform(-math.pi, math.pi)
#     #         goal_x, goal_y = self._sample_safe_position()

#     #         # 1. 调用 /reset_world (直接用 __init__ 创建好的 client)
#     #         future = self._reset_world_client.call_async(std_srvs.srv.Empty.Request())
#     #         start = time.time()
#     #         while not future.done() and time.time() - start < 3.0:
#     #             time.sleep(0.05)

#     #         # 2. 调用 /set_entity_state 传送机器人
#     #         req = SetEntityState.Request()
#     #         req.state.name = "burger" 
#     #         req.state.pose.position.x = float(spawn_x)
#     #         req.state.pose.position.y = float(spawn_y)
#     #         req.state.pose.position.z = 0.0
#     #         req.state.pose.orientation.z = math.sin(spawn_yaw / 2.0)
#     #         req.state.pose.orientation.w = math.cos(spawn_yaw / 2.0)
#     #         req.state.reference_frame = "world"
            
#     #         future = self._reset_client.call_async(req)
#     #         start = time.time()
#     #         while not future.done() and time.time() - start < 3.0:
#     #             time.sleep(0.05)

#     #         # 3. 等待雷达数据并验证安全距离
#     #         time.sleep(0.5)
#     #         timeout = time.time() + 3.0
#     #         min_dist = 0.0
            
#     #         while time.time() < timeout:
#     #             if self._latest_scan is not None:
#     #                 valid_ranges = [r for r in self._latest_scan.ranges if 0.15 < r < self._latest_scan.range_max]
#     #                 if valid_ranges:
#     #                     min_dist = min(valid_ranges)
#     #                 if min_dist > self.dist_to_goal_gen_min:
#     #                     self.get_logger().info(f"Reset done. Fresh min laser dist: {min_dist:.3f}m (retry {attempt+1})")
#     #                     break
#     #             time.sleep(0.1)

#     #         if min_dist > self.dist_to_goal_gen_min:
#     #             self.goal_x = goal_x
#     #             self.goal_y = goal_y
#     #             success = True
#     #             break
#     #         else:
#     #             self.get_logger().warn(f"Spawn too close ({min_dist:.3f}m), retrying...")

#     #     if not success:
#     #         # 真正的兜底：不仅赋值 goal，还要把机器人传送到安全点！
#     #         self.get_logger().warn("All retries failed! Fallback to (0,0).")
#     #         req = SetEntityState.Request()
#     #         req.state.name = "burger"
#     #         req.state.pose.position.x = 0.5
#     #         req.state.pose.position.y = 0.5
#     #         req.state.pose.position.z = 0.0
#     #         req.state.pose.orientation.w = 1.0
#     #         req.state.reference_frame = "world"
#     #         future = self._reset_client.call_async(req)
#     #         start = time.time()
#     #         while not future.done() and time.time() - start < 3.0:
#     #             time.sleep(0.05)
                
#     #         # 修复变量名：self.goal_x 而不是 self._target_goal_x
#     #         self.goal_x = goal_x
#     #         self.goal_y = goal_y

#     #     obs = self._get_obs()
        
#     #     self.prev_dist = self._goal_dist(self._get_odom_data()) if self._get_odom_data() else 1.0
#     #     self.episode_step = 0
#     #     self.last_action = np.zeros(2, dtype=np.float32) # 初始化上一步动作
        
#     #     info = {
#     #         "spawn_pos": (spawn_x, spawn_y),
#     #         "goal_pos": (self.goal_x, self.goal_y),
#     #         "initial_dist": math.hypot(spawn_x - self.goal_x, spawn_y - self.goal_y)
#     #     }
#     #     return obs, info


#     # # 方法 A: 简单粗暴 - 重置整个世界 (所有物体回到初始位置)
#     # # 优点：简单
#     # # 缺点：如果有随机生成的障碍物，它们也会被重置掉，可能失去随机性
#     # def reset_gazebo_world(self):
#     #     client = self.create_client(Empty, '/reset_world')
#     #     while not client.wait_for_service(timeout_sec=1.0):
#     #         self.get_logger().info('service not available, waiting again...')
#     #     request = Empty.Request()
#     #     future = client.call_async(request)
#     #     # rclpy.spin_until_future_complete(self, future)

#     # # 方法 B: 精确控制 - 仅移动机器人到指定起点 (推荐)
#     # # 优点：只重置机器人，保留环境中的其他设置 (如随机障碍物)
#     # # 缺点：代码稍多
#     # # 推荐的方法 B 实现 (防死锁版)
#     # def reset_gazebo_robot(self, x, y, yaw):
#     #     """彻底物理重置：删除旧机器人 → 重新生成全新机器人"""
        
#     #     # ===== 第1步：删除旧机器人（彻底清零所有物理状态） =====
#     #     if self._delete_client.wait_for_service(timeout_sec=3.0):
#     #         del_req = DeleteEntity.Request()
#     #         del_req.name = "burger"
#     #         future = self._delete_client.call_async(del_req)
#     #         start = time.time()
#     #         while not future.done() and time.time() - start < 5.0:
#     #             time.sleep(0.01)
#     #         if future.done():
#     #             self.get_logger().info("Deleted old robot (all physics state cleared)")
#     #         else:
#     #             self.get_logger().warn("Delete timeout, continuing anyway...")
#     #     else:
#     #         self.get_logger().warn("Delete service not available, falling back to set_entity_state")
#     #         self._reset_gazebo_teleport(x, y, yaw)
#     #         return

#     #     # 等物理引擎处理完删除
#     #     time.sleep(0.2)

#     #     # ===== 第2步：在指定位置生成全新机器人 =====
#     #     if self._robot_urdf is None:
#     #         self.get_logger().error("No URDF loaded! Cannot respawn.")
#     #         return

#     #     if self._spawn_client.wait_for_service(timeout_sec=3.0):
#     #         spawn_req = SpawnEntity.Request()
#     #         spawn_req.name = "burger"
#     #         spawn_req.xml = self._robot_urdf
#     #         spawn_req.initial_pose.position.x = float(x)
#     #         spawn_req.initial_pose.position.y = float(y)
#     #         spawn_req.initial_pose.position.z = 0.0  # 让物理引擎自己放下去
#     #         spawn_req.initial_pose.orientation.w = math.cos(yaw / 2)
#     #         spawn_req.initial_pose.orientation.z = math.sin(yaw / 2)
#     #         spawn_req.reference_frame = "world"

#     #         future = self._spawn_client.call_async(spawn_req)
#     #         start = time.time()
#     #         while not future.done() and time.time() - start < 10.0:
#     #             time.sleep(0.01)

#     #         if future.done():
#     #             self.get_logger().info(
#     #                 f"Spawned new robot at ({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.1f}°")
#     #         else:
#     #             self.get_logger().error("Spawn timeout!")
#     #     else:
#     #         self.get_logger().error("Spawn service not available!")

#     #     # 等新机器人稳定（关节初始化、控制器启动）
#     #     time.sleep(0.5)

#     # def _reset_gazebo_teleport(self, x, y, yaw):
#     #     """旧的传送方法（仅作为 fallback）"""
#     #     req = SetEntityState.Request()
#     #     req.state.name = "burger"
#     #     req.state.pose.position.x = x
#     #     req.state.pose.position.y = y
#     #     req.state.pose.position.z = 0.0
#     #     req.state.pose.orientation.w = math.cos(yaw / 2)
#     #     req.state.pose.orientation.z = math.sin(yaw / 2)
#     #     req.state.twist.linear.x = 0.0
#     #     req.state.twist.linear.y = 0.0
#     #     req.state.twist.linear.z = 0.0
#     #     req.state.twist.angular.x = 0.0
#     #     req.state.twist.angular.y = 0.0
#     #     req.state.twist.angular.z = 0.0
#     #     future = self._reset_client.call_async(req)
#     #     start = time.time()
#     #     while not future.done() and time.time() - start < 5.0:
#     #         time.sleep(0.01)


#     # def reset_gazebo_robot(self, start_x, start_y, start_yaw=0.0):
#     #     client = self.create_client(SetEntityState, '/set_entity_state')
        
#     #     # 1. 等待服务可用 (带超时)
#     #     if not client.wait_for_service(timeout_sec=2.0):
#     #         raise RuntimeError("Service /set_entity_state not available! Cannot reset environment.")

#     #     request = SetEntityState.Request()
#     #     request.state.name = "burger"
#     #     request.state.pose.position.x = start_x
#     #     request.state.pose.position.y = start_y
#     #     request.state.pose.position.z = 0.1
#     #     request.state.pose.orientation.x = 0.0  # 补全 x
#     #     request.state.pose.orientation.y = 0.0  # 补全 y
#     #     request.state.pose.orientation.z = math.sin(start_yaw / 2)
#     #     request.state.pose.orientation.w = math.cos(start_yaw / 2)
#     #     request.state.twist.linear.x = 0.0
#     #     request.state.twist.angular.z = 0.0
#     #     request.state.reference_frame = "world"

#     #     self.get_logger().info(f"Reset request: name={request.state.name}, "
#     #                         f"pos=({start_x}, {start_y}, 0.1), "
#     #                         f"quat=({0.0}, {0.0}, {math.sin(start_yaw/2)}, {math.cos(start_yaw/2)})")

#     #     # 2. 调用服务并等待结果 (带超时)
#     #     future = client.call_async(request)
        
#     #     start_time = time.time()
#     #     while not future.done():
#     #         if time.time() - start_time > 10.0: # 3秒超时
#     #             raise RuntimeError("Reset robot service call timeout!")
#     #         rclpy.spin_once(self, timeout_sec=0.01) 
            
#     #     # 3. 检查返回值
#     #     result = future.result()
#     #     self.get_logger().info(f"Reset result: success={result.success}, message={result.status_message}")
#     #     if not result.success: # 假设服务返回包含 success 字段
#     #          self.get_logger().error("Gazebo failed to reset robot state!")

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         action = np.clip(action, self.action_space.low, self.action_space.high)

#         self.send_velocity(float(action[0]), float(action[1]))
        
#         # 等 step_duration，后台线程持续收数据
#         time.sleep(self.step_duration)

#         obs = self._get_obs()
#         odom = self._get_odom_data()
#         imu  = self._get_imu_data()

#         if odom is None:
#             return obs, -1.0, False, True, {"error": "odom unavailable"}

#         curr_dist = self._goal_dist(odom)
#         goal_reached = curr_dist < self.dist_to_goal_threshold
#         collision    = self._check_collision()

#         reward = self._compute_reward(
#             curr_dist=curr_dist, prev_dist=self.prev_dist,
#             goal_reached=goal_reached, collision=collision,
#             action=action, odom=odom, imu=imu
#         )

#         self.prev_dist = curr_dist
#         self.last_action = action.copy()
#         self.episode_step += 1

#         terminated = goal_reached or collision
#         truncated  = self.episode_step >= self.episode_steps_max

#         if terminated or truncated:
#             self.send_velocity(0.0, 0.0)

#         info = {
#             "collision": collision,
#             "goal_reached": goal_reached,
#             "final_dist": curr_dist,
#             "episode_step": self.episode_step
#         }

#         return obs, float(reward), terminated, truncated, info

#     def run_episode(self, model, start_pos=None, goal_pos=None, deterministic=True, safe_threshold=None, skip_spawn_check=False):
#         options = {}
#         """
#         在当前环境中运行一个完整的 Episode
        
#         参数:
#             model: 加载好的 RL 模型 (如 SAC, PPO)
#             start_pos: 指定起点 [x, y, yaw]，None 则随机安全采样
#             goal_pos: 指定终点 [x, y]，None 则随机安全采样
#             deterministic: 是否使用确定性策略预测动作
        
#         返回:
#             dict: 包含该次 episode 的统计指标
#         """
#         options = {}
#         if start_pos is not None:
#             options["start_pos"] = start_pos
#         if goal_pos is not None:
#             options["goal_pos"] = goal_pos
#         # 补全缺失的 options 传递
#         if safe_threshold is not None:
#             options["safe_threshold"] = safe_threshold
#         if skip_spawn_check:
#             options["skip_spawn_check"] = True
            
#         obs, info = self.reset(options=options if options else None)
        
#         ep_reward = 0.0
#         step = 0
#         done = False
        
#         # 修复参数名
#         while not done:
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, reward, terminated, truncated, info = self.step(action)
#             ep_reward += reward
#             step += 1
#             done = terminated or truncated
            
#         return {
#             "reward": ep_reward,
#             "success": info.get('goal_reached', False),
#             "steps": step,
#             "final_dist": info.get('final_dist', -1.0)
#         }

#     def run_episodes(self, model, num_episodes=10, start_pos_list=None, goal_pos_list=None, deterministic=True, safe_threshold=None):
#         """
#         批量运行 Episode
#         如果传入了 list，则按 list 顺序执行；如果没传，则随机生成。
#         """
#         results = []
        
#         # 参数校验
#         if start_pos_list is not None and goal_pos_list is not None:
#             if len(start_pos_list) != len(goal_pos_list):
#                 raise ValueError("start_pos_list 和 goal_pos_list 长度必须一致！")
#             num_episodes = len(start_pos_list)
#         elif start_pos_list is not None:
#             num_episodes = len(start_pos_list)
#         elif goal_pos_list is not None:
#             num_episodes = len(goal_pos_list)

#         for i in range(num_episodes):
#             s_pos = start_pos_list[i] if start_pos_list is not None else None
#             g_pos = goal_pos_list[i] if goal_pos_list is not None else None
            
#             # 如果指定了起点，默认跳过防撞检查（因为往往是为了测特定难点）
#             skip_check = (s_pos is not None)
            
#             res = self.run_episode(
#                 model=model,
#                 start_pos=s_pos,
#                 goal_pos=g_pos,
#                 deterministic=deterministic,
#                 safe_threshold=safe_threshold,
#                 skip_spawn_check=skip_check
#             )
#             results.append(res)
            
#             # 打印单次进度
#             status = "✅" if res["success"] else "❌"
#             print(f"\r[Episode {i+1}/{num_episodes}] {status} Reward: {res['reward']:.2f} | Steps: {res['steps']}", end="", flush=True)
#         print() # 换行

#         # 汇总统计
#         total_success = sum(1 for r in results if r["success"])
#         return {
#             "num_episodes": num_episodes,
#             "success_rate": (total_success / num_episodes) * 100 if num_episodes > 0 else 0,
#             "mean_reward": np.mean([r["reward"] for r in results]) if results else -1,
#             "mean_steps": np.mean([r["steps"] for r in results]) if results else -1,
#             "details": results  # 保留明细，方便画图
#         }

#     def close(self):
#         self.send_velocity(0.0, 0.0)
#         if rclpy.ok():
#             self.destroy_node()

#     # ==================== 内部逻辑 ====================

#     def _sample_safe_position(self) -> Tuple[float, float]:
#         zone = self.safe_zones[self.np_random.integers(len(self.safe_zones))]
#         angle = self.np_random.uniform(0, 2 * math.pi)
#         r = self.np_random.uniform(0, zone["r"])
#         pos = (zone["cx"] + r * math.cos(angle), zone["cy"] + r * math.sin(angle))
#         self.get_logger().debug(f"Sampled position: {pos} in zone {zone}")
#         return pos

#     @staticmethod
#     def _euclidean(a, b):
#         return math.hypot(a[0] - b[0], a[1] - b[1])

#     def _goal_dist(self, odom: Dict) -> float:
#         return self._euclidean(
#             (self.goal_x, self.goal_y), 
#             (odom["x"], odom["y"])
#         )
        
#     def _check_collision(self) -> bool:
#         # 雷达数据过期检查 (在获取数据前做，如果为空说明没数据)
#         with self._lock:
#             is_none = self._latest_scan is None
#             age = time.time() - self._latest_scan_time if not is_none else 1.0
            
#         if is_none:
#             return False
#         if age > 0.25:
#             return False
            
#         # 直接复用，拿到的是 [0, 1] 归一化后的降采样数据
#         laser_norm = self._get_laser_data()
        
#         # 还原成真实物理距离 (乘以最大范围)
#         min_dist = float(np.min(laser_norm)) * self.laser_range_max
        
#         if min_dist < self.proximity_to_collision_threshold:
#             self.get_logger().debug(f"Collision! Min dist: {min_dist:.3f}m")
#             return True
            
#         return False

#     def _check_spawn_pos(self, timeout=3.0, threshold=None):
#         if threshold is None:
#             threshold = getattr(self, 'proximity_to_be_safe_min', 0.3) * 0.9
            
#         timeout_deadline = time.time() + timeout
        
#         while time.time() < timeout_deadline:
#             # 如果 _latest_scan 为空，_get_laser_data 会返回全 1.0，还原后就是 range_max，肯定 > threshold，不会误判
#             laser_norm = self._get_laser_data()
#             min_dist = float(np.min(laser_norm)) * self.laser_range_max
            
#             if min_dist > threshold:
#                 return True
#             time.sleep(0.1)
            
#         self.get_logger().warn(f"Spawn check failed. Min dist: {min_dist:.3f}m < Threshold: {threshold:.3f}m")
#         return False

#     def _check_goal_pos(self, goal_x: float, goal_y: float) -> bool:
#         for zone in self.safe_zones:
#             dist_to_center = self._euclidean((goal_x, goal_y), (zone["cx"], zone["cy"]))
#             if dist_to_center <= zone["r"]:
#                 return True
                
#         self.get_logger().warn(f"Goal ({goal_x:.2f}, {goal_y:.2f}) is outside all safe zones!")
#         return False


#     # def _check_collision(self) -> bool:
#     #     with self._lock:
#     #         if self._latest_scan is None: 
#     #             self.get_logger().info("No laser scan data available for collision check")
#     #             return False
#     #         # 雷达数据超过 250ms 视为过期，不判碰撞
#     #         age = time.time() - self._latest_scan_time
#     #         if age > 0.25:
#     #             self.get_logger().debug(
#     #                 f"Stale scan ({age:.2f}s old), skip collision check")
#     #             return False
            
#     #         ranges = np.array(self._latest_scan.ranges)
#     #         ranges = np.nan_to_num(ranges, nan=self.laser_range_max, posinf=self.laser_range_max)
            
#     #         min_dist = float(np.min(ranges))
            
#     #         # 添加调试日志
#     #         if min_dist < self.proximity_to_collision_threshold:
#     #             self.get_logger().debug(f"Collision detected! Min laser dist: {min_dist:.3f}m (threshold: {self.proximity_to_collision_threshold:.3f}m)")
#     #         else:
#     #             self.get_logger().debug(f"No collision. Min laser dist: {min_dist:.3f}m (threshold: {self.proximity_to_collision_threshold:.3f}m)")
                
#     #         return min_dist < self.proximity_to_collision_threshold

#     # def _check_spawn_pos(self, timeout=3.0, threshold=None):
#     #     if threshold is None:
#     #         threshold = getattr(self, 'proximity_to_be_safe_min', 0.3) * 0.9
            
#     #     timeout_deadline = time.time() + timeout
#     #     min_dist = 0.0
        
#     #     while time.time() < timeout_deadline:
#     #         # 👇 必须加锁！
#     #         with self._lock:
#     #             scan_data = self._latest_scan.ranges if self._latest_scan is not None else None
                
#     #         if scan_data is not None:
#     #             valid_ranges = [r for r in scan_data if 0.15 < r < self.laser_range_max] # 建议统一用 self.laser_range_max
#     #             if valid_ranges:
#     #                 min_dist = min(valid_ranges)
#     #             if min_dist > threshold:
#     #                 return True
#     #         time.sleep(0.1)
            
#     #     self.get_logger().warn(f"Spawn check failed. Min laser dist: {min_dist:.3f}m < Threshold: {threshold:.3f}m")
#     #     return False

#     # def _check_goal_pos(self, goal_x: float, goal_y: float) -> bool:
#     #     """
#     #     检查目标点是否落在预设的安全区域内
#     #     """
#     #     for zone in self.safe_zones:
#     #         # 计算目标点与安全区域圆心的距离
#     #         dist_to_center = math.hypot(goal_x - zone["cx"], goal_y - zone["cy"])
#     #         # 如果在任意一个安全圆内，则判定为安全
#     #         if dist_to_center <= zone["r"]:
#     #             return True
                
#     #     # 遍历完所有安全区都不在里面，说明可能在障碍物或墙壁上
#     #     self.get_logger().warn(f"Goal ({goal_x:.2f}, {goal_y:.2f}) is outside all safe zones!")
#     #     return False


#     def _min_laser(self) -> float:
#         with self._lock:
#             if self._latest_scan is None: return self.laser_range_max
#             ranges = np.array(self._latest_scan.ranges)
#             ranges = np.nan_to_num(ranges, nan=self.laser_range_max, posinf=self.laser_range_max)
#             return float(np.min(ranges))

#     def _get_obs(self) -> np.ndarray:
#         laser_norm = self._get_laser_data()
#         odom = self._get_odom_data()
#         imu  = self._get_imu_data()

#         imu_acc = np.zeros(3, dtype=np.float32)
#         imu_gyro = np.zeros(3, dtype=np.float32)
#         imu_rp = np.zeros(2, dtype=np.float32)
#         goal_angle, goal_dist = 0.0, 1.0
#         odom_vel = np.zeros(2, dtype=np.float32)

#         if imu is not None:
#             imu_acc = np.clip(np.array(imu['acc']) / self.lin_acc_physics_max, -1.0, 1.0)
#             imu_gyro = np.clip(np.array(imu['gyro']) / self.ang_vel_imu_physics_max, -1.0, 1.0)
#             # 归一化 Roll/Pitch，约 45度 (pi/4) 映射到 1.0
#             imu_rp = np.clip(np.array(imu['rpy'][:2]) / (math.pi/4), -1.0, 1.0)

#         if odom is not None:
#             dx = self.goal_x - odom['x']
#             dy = self.goal_y - odom['y']
#             yaw = odom['yaw']
#             local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
#             local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
#             goal_angle = math.atan2(local_y, local_x) / math.pi
#             goal_dist = min(math.hypot(dx, dy) / self.dist_to_goal_clip_norm, 1.0)
#             odom_vel = np.clip(
#                 np.array([odom['vx'], odom['wz']]) / np.array([self.lin_vel_max, self.ang_vel_max]), 
#                 -1.0, 1.0
#             )

#         last_act_norm = np.clip(
#             np.array([self.last_action[0], self.last_action[1]]) / 
#             np.array([self.lin_vel_max, self.ang_vel_max]), 
#             -1.0, 1.0
#         )
#         obs = np.concatenate([
#             laser_norm, imu_acc, imu_gyro, imu_rp,
#             [goal_angle], [goal_dist], odom_vel, last_act_norm
#         ]).astype(np.float32)
        
#         return obs

#     def _compute_reward(self, curr_dist, prev_dist, goal_reached, collision, action, odom, imu):
#         reward = 0.0
        
#         # 1. 终止条件奖励
#         if goal_reached: return self.reward_at_goal
#         if collision:    return self.penalty_at_collision

#         # 2. 距离变化奖励
#         dist_delta = prev_dist - curr_dist
#         reward += dist_delta * self.reward_factor_approaching_goal

#         # 3. 朝向奖励 【关键修复：防止 0.012m/s 蠕动刷分】
#         yaw = odom['yaw']
#         target_angle = math.atan2(self.goal_y - odom['y'], self.goal_x - odom['x'])
#         yaw_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
#         heading_factor = math.cos(yaw_error)
        
#         # 修复逻辑：去掉 if abs(odom['vx']) > 0.01 的门槛，直接乘以速度！
#         # 这样只有“真正在移动”且“朝向正确”时才给奖励，站着不动朝向再好也是 0 分。
#         reward += heading_factor * self.reward_good_orientation * abs(odom['vx'])

#         # 4. 安全惩罚
#         min_dist = self._min_laser()
#         if min_dist < self.proximity_to_be_safe_min:
#             reward += self.penalty_in_safe_proximity * ((self.proximity_to_be_safe_min - min_dist) / self.proximity_to_be_safe_min)**2

#         # 5. 稳定性惩罚 (基于 IMU 的 Roll 和 Pitch)
#         if imu is not None:
#             roll, pitch = imu['rpy']
#             tilt_factor = (roll**2 + pitch**2) / ( (math.pi/6)**2 )
#             reward -= abs(self.penalty_instability) * min(tilt_factor, 1.0)

#         # 6. 卡住惩罚 
#         vel_error = abs(action[0]) - abs(odom['vx'])
#         if vel_error > self.lin_vel_stuck_threshold: 
#              reward += self.penalty_stuck * (vel_error / self.lin_vel_max)

#         # 7. 动作平滑性惩罚 【关键修复：解除手脚束缚】
#         diff_lin = action[0] - self.last_action[0]
#         diff_ang = action[1] - self.last_action[1]

#         norm_diff_lin = diff_lin / self.lin_vel_max
#         norm_diff_ang = diff_ang / self.ang_vel_max

#         smoothness_penalty = np.sqrt(norm_diff_lin**2 + norm_diff_ang**2)

#         # 修复逻辑：增加一个“容忍阈值”。正常的转弯和加减速不应该被惩罚，
#         # 只有类似抽搐一样的突变（比如一帧内角速度变化超过 30%）才惩罚。
#         SMOOTHNESS_TOLERANCE = 0.3 
#         if smoothness_penalty > SMOOTHNESS_TOLERANCE:
#             reward += self.penalty_action_smoothness * (smoothness_penalty - SMOOTHNESS_TOLERANCE)
            
#         # 8. 时间惩罚
#         reward += self.penalty_elapsing_time
        
#         return reward

#     # def _compute_reward(self, curr_dist, prev_dist, goal_reached, collision, action, odom, imu):
#     #     reward = 0.0
        
#     #     # 1. 终止条件奖励
#     #     if goal_reached: return self.reward_at_goal
#     #     if collision:    return self.penalty_at_collision

#     #     # 2. 距离变化奖励
#     #     dist_delta = prev_dist - curr_dist
#     #     reward += dist_delta * self.reward_factor_approaching_goal

#     #     # 3. 朝向奖励
#     #     yaw = odom['yaw']
#     #     target_angle = math.atan2(self.goal_y - odom['y'], self.goal_x - odom['x'])
#     #     yaw_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
#     #     heading_factor = math.cos(yaw_error)
        
#     #     if abs(odom['vx']) > 0.01:
#     #         reward += heading_factor * self.reward_good_orientation

#     #     # 4. 安全惩罚
#     #     min_dist = self._min_laser()
#     #     if min_dist < self.proximity_to_be_safe_min:
#     #         reward += self.penalty_in_safe_proximity * ((self.proximity_to_be_safe_min - min_dist) / self.proximity_to_be_safe_min)**2

#     #     # 5. 稳定性惩罚 (基于 IMU 的 Roll 和 Pitch)
#     #     if imu is not None:
#     #         roll, pitch = imu['rpy']
#     #         tilt_factor = (roll**2 + pitch**2) / ( (math.pi/6)**2 )
#     #         reward -= abs(self.penalty_instability) * min(tilt_factor, 1.0)

#     #     # 6. 卡住惩罚 (优化版：检测指令与实际速度的偏差)
#     #     # 如果指令速度很大，但实际速度很小，说明发生了打滑或卡死
#     #     vel_error = abs(action[0]) - abs(odom['vx'])
#     #     if vel_error > self.lin_vel_stuck_threshold: # 允许 0.1 的控制误差
#     #          reward += self.penalty_stuck * (vel_error / self.lin_vel_max)
#     #         # 这个逻辑的初衷是：我给了很大的油门，但车没动（比如撞墙卡死了或者打滑了），所以要惩罚。

#     #         # 需要注意的物理现象：
#     #         # TurtleBot3 是有加速度的。如果你上一秒给的速度是 0，这一秒突然给 0.2 m/s 的指令，在 time.sleep(step_duration) 这极短的时间内（比如 0.1秒），底盘的 odom['vx'] 可能只到了 0.05 m/s。
#     #         # 这时候 vel_error = 0.2 - 0.05 = 0.15，会误触发卡住惩罚。

#     #         # 对策：
#     #         # 不需要改代码，但在你后面调参时，self.penalty_stuck 的绝对值一定要设得非常小（比如 -0.1 甚至 -0.05），让它成为一个“长期卡死时的累积惩罚”，而不是“一瞬间没加速就狠罚”。只要它远小于正常靠近目标的奖励，网络就会学到容忍启动时的短暂延迟。


#     #     # 7. 动作平滑性惩罚 (新增)
#     #     # 1. 计算变化量
#     #     diff_lin = action[0] - self.last_action[0]
#     #     diff_ang = action[1] - self.last_action[1]

#     #     # 2. 归一化 (变化量 / 最大速度 = 变化百分比)
#     #     norm_diff_lin = diff_lin / self.lin_vel_max
#     #     norm_diff_ang = diff_ang / self.ang_vel_max

#     #     # 3. 计算加权欧氏距离 (此时两者量纲一致，都是 0~1 的比例)
#     #     smoothness_penalty = np.sqrt(norm_diff_lin**2 + norm_diff_ang**2)

#     #     # 4. 施加惩罚
#     #     reward += self.penalty_action_smoothness * smoothness_penalty
#     #     # 8. 时间惩罚
#     #     reward += self.penalty_elapsing_time
        
#     #     return reward

# class NaviMissionAssigner() :
#     pass




# # # ==========================================
# # # 你的训练脚本主流程
# # # ==========================================
# # if __name__ == '__main__':
# #     # 1. 在创建任何环境之前，先拿到 URDF (只查一次网络)
# #     my_urdf = fetch_tb3_urdf()
    
# #     # 2. 把 URDF 塞进配置字典里
# #     env_config = {
# #         "robot_urdf": my_urdf,  # 注入进去
# #         # ... 其他你需要的 reward、environment 配置 ...
# #     }
    
# #     # 3. 无论你是单环境还是多环境，都这么写
# #     from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
# #     def make_env():
# #         def _init():
# #             # 此时环境内部不再去查 ROS2，直接用传进来的 my_urdf
# #             return TurtleBot3NavEnv(config=env_config) 
# #         return _init

# #     # 单进程训练
# #     # env = DummyVecEnv([make_env()])
    
# #     # 多进程训练 (以后直接改这里，绝对不会有端口冲突和查询风暴)
# #     # env = SubprocVecEnv([make_env() for _ in range(4)])
    
# #     # ... 接下来写 model.learn() ...