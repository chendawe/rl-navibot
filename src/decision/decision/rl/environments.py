# ==========================================
# 1. 标准库与第三方库
# ==========================================
import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# 2. 本地项目模块 (🟢 极致解耦：没有任何 rclpy 的痕迹)
# ==========================================
from core.ros2.master import Ros2Runtime
from core.ros2.channels.bridges.base import BaseBridge
from decision.rl.rewards import RewardContext, get_reward_model

logger = logging.getLogger(__name__)


def fetch_robot_urdf(runtime: Ros2Runtime) -> str:
    """
    在主进程中提前拉取完整的 URDF。
    
    【架构设计说明】
    重构前：依赖具体的 GazeboBridge 单例，隐式触发 rclpy.init()。
    重构后：通过依赖注入传入 Ros2Runtime。我们甚至不需要实例化庞大的 
            RobotBridge 或 GazeboSimulator，而是直接 new 一个最纯粹的 
            BaseBridge 作为“一次性管道”，借用 Runtime 的线程池去 ROS2 
            网络里捞一个字符串。
    
    这就是防腐层的终极威力——让非通信专业的算法代码，看起来像在读纯 Python 脚本。
    """
    # 1. 创建一次性管道：不绑定任何传感器，不绑定任何仿真服务，只为了发一次网络请求
    #    它会自动向 runtime 注册，获得被调度和收发消息的能力
    prefetcher = BaseBridge(node_name='urdf_prefetcher', runtime=runtime)
    
    # 2. 意图表达：我不关心 ROS2 底层的 DDS 怎么发现节点，我只要“按名字找人，拿参数”
    target_node = prefetcher._find_node(
        name_fragment='robot_state_publisher', 
        timeout_sec=5.0
    )
    
    if not target_node:
        raise RuntimeError("未在 ROS2 网络中找到 'robot_state_publisher' 节点！")
        
    req = GetParameters.Request()
    req.names = ['robot_description']
    
    result = prefetcher._call_service(
        service_name=f"{target_node}/get_parameters",
        srv_type=GetParameters,
        request=req,
        timeout_sec=5.0
    )
    
    # 3. 提取结果与兜底防御：防止 Gazebo 模型加载异常导致后续训练全盘崩溃
    urdf_string = None
    if result.values and hasattr(result.values[0], 'string_value'):
        urdf_string = result.values[0].string_value
        
    if not urdf_string:
        raise RuntimeError("成功连接到 ROS2 节点，但获取到的 robot_description 为空！请检查 Gazebo 模型是否正常加载。")
        
    logger.info(f"Successfully fetched URDF ({len(urdf_string)} chars)")
    return urdf_string



import numpy as np

import gymnasium as gym

class TurtleBot3NaviEnv(gym.Env):
    """
    TurtleBot3 导航 RL 环境。
    
    依赖注入：通过 runtime 参数接收 Ros2Runtime 实例，
    内部自行创建 RobotBridge（通信）和 GazeboSimulator（仿真控制）。
    """
    metadata = {"render_modes": []}
    
    SAFE_ZONES = {"default": [{"cx": 0, "cy":  0, "r": 1.}]}

    def __init__(self, runtime, env_config: Optional[Dict[str, Any]] = None, rl_runtime_mode: Optional[str] = "train", robot_urdf: Optional[str] = None):
        gym.Env.__init__(self)
        
        # 支持三种模式: "train"(默认), "eval"(严格测试), "play"(看戏演示)
        self.rl_runtime_mode = rl_runtime_mode
        
        # ================= 1. 默认配置与合并 (完全不变) =================
        default_env_config = {
            "robot": {
                "name": "burger", "laser_range_max": 3.5, "laser_beams_num": 24,
                "laser_noise_threshold": 0.08,
                "lin_vel_max": 0.22, "ang_vel_max": 1.5,
                "lin_vel_stuck_threshold": 0.12, "lin_acc_physics_max": 2.0,
                "ang_vel_imu_physics_max": 3.0, "proximity_to_collision_threshold": 0.11,
                "proximity_to_be_safe_min": 0.30,
            },
            "world": {
                "name": "ttb3_world", "episode_steps_max": 500, "step_duration": 0.1,
                "dist_to_goal_threshold": 0.30,
                "dist_to_goal_gen_min": 1.0, "dist_to_goal_clip_norm": 5.0,
                "safe_zones": {},
            },
            "reward": {
                "name": "ttb3_world",
                "reward_at_goal": 300.0, "penalty_at_collision": -300.0,
                "reward_factor_approaching_goal": 20.0,
                "penalty_elapsing_time": -0.2,
                "penalty_stuck": -0.05,
                "reward_good_orientation": 2.0,
                "penalty_factor_in_safe_proximity": -1.0,
                "penalty_instability": -2.0, "penalty_action_smoothness": -0.5,
            },
        }
        if env_config:
            for key, value in env_config.items():
                if isinstance(value, dict) and key in default_env_config:
                    default_env_config[key].update(value)
                else:
                    default_env_config[key] = value

        self.config = {
            "robot": default_env_config["robot"],
            "world": default_env_config["world"],
            "reward": default_env_config["reward"],
        }
        self.robot = self.config["robot"]
        self.world = self.config["world"]
        self.rew   = self.config["reward"]
        reward_name = self.config["reward"]["name"]
        self.reward_fn = get_reward_model(reward_name, self.config)
        
        self.zones = self.world.pop("safe_zones", {})
        if not self.zones:
            raise ValueError(f"配置中未找到 safe_zones！")

        # ================= 2. 初始化通信桥 (🔥 核心变更区) =================
        #
        # 重构前：self.bridge = GazeboBridge(...) 一个类包揽所有事。
        # 重构后：按职责拆分为两个独立实体，共享同一个 Runtime 线程池。
        #
        #   RobotBridge    → 只管"这台机器人的传感器和执行器"
        #   GazeboSimulator → 只管"Gazebo 这个仿真环境怎么重置、怎么摆放"
        #
        # 换成 Isaac Sim / 真机时，只需替换 Simulator，RobotBridge 零修改。

        # --- 2-. 智能 URDF 调度策略 ---
        urdf_to_inject = robot_urdf  # 默认为 None
        self.current_reset_mode = self.world.get("reset_mode", "teleport")
        
        if urdf_to_inject is None and self.current_reset_mode == "spawn":
            # 🌟 只有在“必须用到 URDF”且“用户没给”时，才触发自动拉取。
            # 这样保证了日常训练 (teleport) 连相关的 import 都不会执行。
            logger.info("[Env] 检测到 spawn 重置模式且未传入 URDF，正在自动拉取...")
            urdf_to_inject = fetch_robot_urdf(runtime)
        
        self._robot_urdf = urdf_to_inject
        
        from core.ros2.channels.bridges.robot import RobotBridge
        from core.ros2.simulators.gazebo import GazeboSimulator

        self.robot_bridge = RobotBridge(runtime, node_name=f'{self.robot["name"]}_bridge')
        self.sim = GazeboSimulator(runtime, node_name=f'{self.robot["name"]}_sim')

        # --- 2a. 仿真服务 → 交给 GazeboSimulator ---
        sim_timeout = self.world.get("sim_service_timeout", 10.0)
        self.sim.wait_for_services(timeout_sec=sim_timeout)
        self.sim.setup(
            reset_world='/reset_world',
            set_entity='/set_entity_state',
            delete_entity='/delete_entity',
            spawn_entity='/spawn_entity',
            entity_name=self.robot["name"],
        )

        # --- 2b. 传感器 + 执行器 → 交给 RobotBridge ---
        self.robot_bridge.setup(
            laser_topic='/scan',
            imu_topic='/imu',
            odom_topic='/odom',
            cmd_vel_topic='/cmd_vel',
            goal_topic='/goal_pose',
            laser_noise_threshold=self.robot["laser_noise_threshold"],
        )

        # --- 2c. 阻塞等待 Gazebo 服务就绪（reset 之前必须确认） ---
        self.sim.wait_for_services()

        # ================= 3. RL 空间定义 (完全不变) =================
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(38,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-self.robot["lin_vel_max"], -self.robot["ang_vel_max"]], dtype=np.float32),
            high=np.array([ self.robot["lin_vel_max"],  self.robot["ang_vel_max"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = obs_space
        self.action_space = act_space

        # ================= 4. 运行时状态大一统 (完全不变) =================
        self.state = {
            "goal_x": 0.0, "goal_y": 0.0, "prev_dist": 0.0,
            "step": 0, "last_action": np.array([0.0, 0.0], dtype=np.float32),
        }
        self.max_step_dist = self.robot["lin_vel_max"] * self.world["step_duration"] * 2.0
        self._last_min_dist = None
        
        self.print_config()

    def print_config(self):
        print("\n" + "=" * 60)
        print("🤖 TurtleBot3 Navigation Env Configuration")
        print("=" * 60)
        
        print(f"robot Name   : {self.robot.get('name', 'N/A')}")
        print(f"World Name : {self.world.get('name', 'N/A')}")
        
        print("-" * 60)
        print("📡 Env robot Config (Physics & Hardware):")
        for k, v in self.robot.items():
            print(f"  - {k:<35} : {v}")
            
        print("-" * 60)
        print("🗺️  Env World Config (Task & Map):")
        for k, v in self.world.items():
            print(f"  - {k:<35} : {v}")
            
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
        obs_shape = self.observation_space.shape
        act_low  = self.action_space.low
        act_high = self.action_space.high
        print(f"  - {'Obs Shape':<35} : {obs_shape}")
        print(f"  - {'Act Space':<35} : [{act_low[0]:.2f}, {act_high[0]:.2f}], [{act_low[1]:.2f}, {act_high[1]:.2f}]")
        
        print("=" * 60 + "\n")

    # ... 在 Env 类的合适位置添加 (建议放在 print_config 之后) ...

    def set_urdf(self, urdf: Optional[str] = None):
        """
        运行时动态更新机器人的 URDF 模型。
        
        设计考量：
        1. 支持外部直接传入新 URDF 字符串。
        2. 支持传入 None 触发自动从 ROS2 网络重新拉取（与 __init__ 行为保持绝对一致）。
        3. 必须将更新同步到底层的 GazeboSimulator，保证数据源单一性。
        
        注意：只有在 GazeboSimulator 配置为 "spawn" 重置模式时，此 URDF 才会在 
              下一次 env.reset() 时真正生效。"teleport" 模式下修改此值无实际物理效果。
        """
        target_urdf = urdf
        
        # 1. 如果没传字符串，尝试自动去底层拉取
        if target_urdf is None:
            from core.ros2.utils.urdf import fetch_robot_urdf
            logger.info("[Env] 收到 set_urdf(None) 指令，正在重新从网络拉取...")
            target_urdf = fetch_robot_urdf(self.runtime) # 假设你把 runtime 存为了 self.runtime
            
        if not target_urdf:
            logger.warning("[Env] 无法获取有效的新 URDF，更新中止。")
            return

        # 2. 更新 Env 自身缓存
        self._robot_urdf = target_urdf
        
        # 3. 🔥 核心：同步推送给底层仿真器，保证状态大一统
        self.sim.update_urdf(target_urdf)
        
        # 4. 友善提示（防止用户在 teleport 模式下改了半天发现没效果去查 bug）
        current_mode = self.world.get("reset_mode", "teleport")
        if current_mode != "spawn":
            logger.warning(
                f"[Env] URDF 已更新，但当前重置模式为 '{current_mode}'。"
                f"若要使新 URDF 生效，请将配置中的 world.reset_mode 设为 'spawn'。"
            )
        else:
            logger.info(f"[Env] URDF 已成功热更新 (长度: {len(target_urdf)} chars)，将在下次 reset 时生效。")


    # ==================== 数据获取与控制 (彻底解耦版) ====================
    # 🧹 清理说明：
    # 1. 删除了 _spin, _scan_cb, _imu_cb, _odom_cb (全在 Bridge 后台线程里了)
    # 2. 删除了所有的 with self._lock: (Bridge 的 get_data 内部已经封装了线程安全)
    # 3. 删除了所有的 ROS2 消息体解包逻辑 (如 msg.pose.pose.position.x)
    # 🏗️ 架构定位：这一层只负责"状态空间适配"，绝不碰"通信协议"。

    def _get_odom_data(self) -> Optional[Dict]:
        """
        获取底盘位姿与速度信息。
        ★ 以前：拿 msg -> 手动拆 p.x, o.w -> 算 atan2 -> 处理异常
        ★ 现在：直接拿算好的纯字典，业务层零感知底层是 Odometry 还是 TF
        """
        return self.robot_bridge.get_odom_data()

    def _get_imu_data(self) -> Optional[Dict]:
        """
        获取 IMU 姿态与角速度信息。
        ★ 以前：拿 msg -> 手动拆 a.x, o.x -> 算 asin 处理奇点
        ★ 现在：直接拿算好的纯列表，Bridge 内部已经用 np.clip 保护了 asin 域
        """
        return self.robot_bridge.get_imu_data()

    def _get_laser_data(self) -> np.ndarray:
        """
        获取降维、归一化后的激光雷达状态。
        （委托给 RobotBridge 的语义化方法）
        """
        return self.robot_bridge.get_laser_normalized(
            beams=self.robot["laser_beams_num"],
            max_range=self.robot["laser_range_max"]
        )

    def send_velocity(self, lin_x: float, ang_z: float):
        """
        下发底盘速度指令。
        ★ 以前：msg = Twist(); msg.linear.x = ... (需要 import geometry_msgs)
        ★ 现在：传两个 float 进去就行，彻底告别 Env 层对 ROS 消息类型的依赖
        """
        self.robot_bridge.send_velocity(lin_x, ang_z)

    def _get_zero_obs(self):
        """
        生成全零观测值的兜底函数。
        🛡️ 为什么需要它？
        在 Gym 的 step/reset 逻辑中，如果遇到异常需要提前终止，
        必须返回一个 shape 和 dtype 与 observation_space 完全一致的数组。
        返回 None 会导致 PPO/SAC 算法在打包 batch 时直接报错崩溃。
        """
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    # ==================== Gym 接口 ====================
    # ==================== Reset 辅助函数 ====================
    def _execute_sim_reset(self, x: float, y: float, yaw: float):
        """
        屏蔽 teleport 与 spawn 细节的底层执行器。
        交给 _generate_valid_task 和 _fallback_task 统一调用。
        """
        mode = self.current_reset_mode
        
        self.sim.reset_world()  # 无论哪种模式，清空物理世界的碰撞痕迹是必须的
        
        if mode == "teleport":
            self.sim.set_robot_pose(x, y, yaw)
        elif mode == "spawn":
            if not self._robot_urdf:
                raise RuntimeError("Spawn 模式缺少 URDF，这不应该发生（已被 reset() 拦截）")
            # ⚠️ 注意：这里的 delete 和 spawn 方法名取决于你 GazeboSimulator 里的具体实现
            self.sim.delete_entity(self.robot["name"])
            time.sleep(0.2)  # spawn 模式下，delete 后需要给 Gazebo 一点点反应时间
            self.sim.spawn_entity(urdf=self._robot_urdf, x=x, y=y, yaw=yaw)
            
        # ⏳ 物理引擎稳定期 (spawn 模式可能需要稍微长一点点，0.5s 是两种模式的折中安全值)
        time.sleep(0.5)
        
    def _wait_for_services(self) -> bool:
        """
        幂等服务检查器。
        🛡️ 防御设计：RL 训练动辄几万步，不可能每次 reset 都去查一次服务。
        采用单次检查加缓存标志 (_services_ready) 的策略，把通信开销降到最低。
        
        📝 注：在新的 __init__ 中已经调用了 self.sim.wait_for_services()，
            所以正常流程下这里永远不会真正再查一次。
            保留它是为了兜底：万一 Gazebo 中途重启，reset 时能优雅感知。
        """
        if hasattr(self, '_services_ready') and self._services_ready:
            return True

        logger.info("Waiting for Gazebo reset services...")
        try:
            self.sim.wait_for_services(timeout_sec=5.0)
            self._services_ready = True
            logger.info("Gazebo services are ready!")
            return True
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Service check failed: {e}")
            return False

    def _clear_sensor_cache(self):
        """
        传感器缓存清理 (空操作)。
        🧠 架构说明：为什么这里是个 pass？
        RobotBridge 的 ATS 会在后台持续高频泵血覆盖 _latest_synced_data。
        如果在 reset 时强行清空缓存，会导致紧随其后的 _get_obs() 触发 fallback，
        返回全零状态而非物理引擎的真实数据。让 ATS 自然刷写才是最安全的。
        """
        pass

    def _parse_reset_options(self, options: Optional[Dict]) -> Dict:
        """
        Gymnasium API 标准适配器。
        作用：将外部可能传入的 None 或残缺字典，安全地展开为带有默认值的完整字典。
        避免在后续的 _generate_valid_task 中出现大量的 if options is not None 判断。
        """
        if not options:
            return {"start_pos": None, "goal_pos": None, "safe_threshold": None, "skip_spawn_check": False}
        return {
            "start_pos": options.get("start_pos"),
            "goal_pos": options.get("goal_pos"),
            "safe_threshold": options.get("safe_threshold"),
            "skip_spawn_check": options.get("skip_spawn_check", False)
        }

    def _generate_valid_task(self, opts) -> Optional[Dict]:
        """
        核心任务生成器：采样 -> 仿真重置 -> 安全校验。
        🎯 核心改变：这里从"ROS 指令发送器"彻底退化成了"纯业务逻辑调度器"。
        """
        custom_start = opts["start_pos"]
        target_goal = opts["goal_pos"]
        use_custom_start = custom_start is not None

        # 🔒 限制最大尝试次数：绝对禁止在 RL 循环里出现无限 while True，否则训练会假死！
        for attempt in range(5):
            # 1. 确定候选起点 (支持随机采样或外部强制指定)
            if use_custom_start:
                s_x, s_y, s_yaw = custom_start
            else:
                s_x, s_y = self._sample_safe_position()
                s_yaw = self.np_random.uniform(-math.pi, math.pi)

            # 2. 确定候选终点
            g_x, g_y = target_goal if target_goal else self._sample_safe_position()

            # 3. 距离校验 (防止生成太简单的任务导致算法退化为原地打转)
            if not use_custom_start and not target_goal:
                if self._euclidean((s_x, s_y), (g_x, g_y)) < self.world["dist_to_goal_gen_min"]:
                    continue

            # 4. 调用仿真器重置并传送
            # ★ 以前：组装 Request -> call_async -> 轮询 Future -> 检查异常 (至少 15 行代码)
            # ★ 现在：两行纯 Python 语义调用。仿真控制全权委托给 GazeboSimulator。
            self._execute_sim_reset(s_x, s_y, s_yaw)

            # ⏳ 物理引擎稳定期：Gazebo 的接触传感器和碰撞体积在瞬移后需要几帧来收敛。
            # 如果没有这个 sleep，_check_spawn_pos 读到的碰撞状态可能是上一帧的脏数据。
            time.sleep(0.5)

            # 5. 碰撞与安全性校验
            is_spawn_safe = self._check_spawn_pos(threshold=opts["safe_threshold"]) if not opts["skip_spawn_check"] else True
            is_goal_safe = self._check_goal_pos(g_x, g_y) if not target_goal else True

            if is_spawn_safe and is_goal_safe:
                return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}

            if not use_custom_start:
                logger.warning(f"Validation failed, retry {attempt+1}/5...")

        # 5次尝试全败，返回 None，交由上层兜底处理
        return None

    def _fallback_task(self, opts) -> Dict:
        """
        终极兜底策略。
        🛡️ 鲁棒性保证：在 RL 分布式训练中，环境崩溃是常态，但"训练进程退出"是绝对不允许的。
        即使 Gazebo 全乱套了，也要强行给算法吐出一个合法的 (obs, info)，哪怕是原地重置。
        """
        if opts["start_pos"] is not None:
            logger.error(f"Failed to set custom start pose: {opts['start_pos']}")
            s_x, s_y, s_yaw = opts["start_pos"]
        else:
            logger.warning("All retries failed! Fallback to (0.5, 0.5).")
            s_x, s_y, s_yaw = 0.5, 0.5, 0.0

        # 即使是兜底，也保持调用接口的一致性
        # ★ 以前：self.sim.reset_world() + self.sim.set_robot_pose(...)
        # ★ 现在：
        self._execute_sim_reset(s_x, s_y, s_yaw)

        g_x, g_y = opts["goal_pos"] if opts["goal_pos"] else (0.0, 0.0)
        return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}

    def reset(self, seed=None, options=None):
        """
        标准 Gymnasium Reset 接口。
        """
        # 1. 调用父类处理 seed，确保 self.np_random 被正确初始化（Gym 强制规范）
        super().reset(seed=seed, options=options)
        if self.current_reset_mode == "spawn":
            if self._robot_urdf is None:
                raise ValueError("在 spawn 重置模式下，必须提供有效的 robot_urdf！")
        
        # 2. 幂等的服务检查
        if not self._wait_for_services():
            # 🔥 极致防御：如果连 Gazebo 都挂了，直接返回全零 obs，保住训练进程的命
            return self._get_zero_obs(), {}

        # 3. 解析外部选项，生成合法任务
        opts = self._parse_reset_options(options)
        task = self._generate_valid_task(opts)
        if task is None:
            task = self._fallback_task(opts)

        # 4. 刷新环境内部状态机 (大一统状态字典)
        self.state["goal_x"], self.state["goal_y"] = task["g_x"], task["g_y"]
        self.state["step"] = 0
        self.state["last_action"] = np.zeros(2, dtype=np.float32)

        # 5. 锚定初始距离 (Reward 计算的基准线)
        odom = self._get_odom_data()
        self.state["prev_dist"] = self._goal_dist(odom) if odom else 1.0

        self._last_min_dist = None  # ★ 新回合重置距离记录

        # 6. 返回标准元组
        return self._get_obs(), {
            "spawn_pos": (task["s_x"], task["s_y"]),
            "goal_pos": (task["g_x"], task["g_y"]),
            "initial_dist": math.hypot(task["s_x"] - task["g_x"], task["s_y"] - task["g_y"])
        }

    # ==================== Step 辅助函数 ====================

    def _get_step_data(self):
        """
        数据聚合器。
        🧹 以前：这里可能是三个带锁的复杂取值操作。
        🚀 现在：一行代码把 RL 需要的所有感官数据打包，为后续的状态拼装和奖励计算供料。
        """
        return self._get_obs(), self._get_odom_data(), self._get_imu_data()

    def _check_termination(self, curr_dist: float) -> Tuple[bool, bool]:
        """
        终止条件裁决机。
        严格遵守 Gymnasium 的语义区分：
        - goal_reached -> terminated (环境内在逻辑结束)
        - collision    -> terminated (进入非法状态)
        """
        goal_reached = curr_dist < self.world["dist_to_goal_threshold"]
        collision = self._check_collision()
        return goal_reached, collision

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        🏆 核心 Step 函数：整个环境的心脏。
        经过重构，这里不再有任何 ROS2 的痕迹，变成了一段纯正的"状态机推演"代码。
        """
        # 1. 动作安全钳位 (防御性编程：防止算法网络输出越界值烧毁底层电机)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 2. 执行动作 & 等待物理演化
        self.send_velocity(float(action[0]), float(action[1]))
        # ⏳ 同步阻塞：Gym 是单线程步进模型，必须等物理引擎跑完这一帧，才能拿下一帧数据
        time.sleep(self.world["step_duration"])

        # 3. 获取最新数据
        obs, odom, imu = self._get_step_data()
        # 🛡️ 极端容错：如果底层通信断开，直接返回 truncated=True 保命
        if odom is None:
            return obs, -1.0, False, True, {"error": "odom unavailable"}

        # 4. 判定终止条件
        curr_dist = self._goal_dist(odom)
        goal_reached, collision = self._check_termination(curr_dist)

        # ==========================================
        # 🔥 组装严格契约
        # ==========================================
        # 这就是重构的巅峰：把所有零散的参数打包成一个纯数据对象，扔给奖励函数。
        ctx = RewardContext(
            goal_reached=goal_reached,
            collision=collision,
            curr_dist=curr_dist,
            prev_dist=self.state["prev_dist"],
            goal_x=self.state["goal_x"],
            goal_y=self.state["goal_y"],
            action=action,
            last_action=self.state["last_action"],
            odom=odom,
            imu=imu,
            min_laser=self._min_laser()
        )

        # ==========================================
        # 🔥 计算奖励 (告别内部臃肿的 if-else)
        # ==========================================
        reward = self.reward_fn.compute(ctx)

        # 5. 推进内部状态机
        self.state["prev_dist"] = curr_dist
        self.state["last_action"] = action.copy()
        self.state["step"] += 1

        # 6. 封装 Gym 标准五元组返回值
        # 🔥 训练 vs 评估 解耦：
        # 训练时：碰墙立即拉闸，节省算力。
        # 评估时：碰墙不拉闸，看策略有没有自我纠错（倒车）能力。
        terminated = goal_reached or collision
        
        # 🎮 Play 模式特权：碰墙不拉闸，看策略的自我纠错能力
        if self.rl_runtime_mode == "play":
            terminated = goal_reached
            
        truncated = self.state["step"] >= self.world["episode_steps_max"]
        
        # 🛑 刹车逻辑：评估模式下碰墙不刹车，把控制权完全交给网络
        should_brake = (self.rl_runtime_mode != "eval" and collision) or terminated or truncated
        if should_brake:
            self.send_velocity(0.0, 0.0)

        return obs, float(reward), terminated, truncated, {
            "collision": collision,
            "goal_reached": goal_reached,
            "final_dist": curr_dist,
            "episode_step": self.state["step"]
        }

    # ==================== 推理 & 批量测试 ====================

    def run_episode(self, model, start_pos=None, goal_pos=None,
                    deterministic=True, safe_threshold=None,
                    skip_spawn_check=False, verbose=False):
        """
        🎮 独立的推理/测试循环。
        为什么要有这个函数？因为在实际调参时，你不想每次都去写 while not done 的样板代码。
        它将 Stable-Baselines3 (或其他库) 的 model 与环境解耦测试，非常适合做快速验证。
        """
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

    def run_episodes(self, model, num_episodes=10,
                     start_pos_list=None, goal_pos_list=None,
                     deterministic=True, safe_threshold=None, verbose=False):
        """
        📊 批量测试统计器。
        支持：动态列表输入、自动对齐长度、静默模式下自带进度条 (\r 刷新)。
        """
        results = []

        # 动态推导实际要跑的 episode 数量
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

            # 如果是自定义起点，通常意味着你想精准测试某个位置，此时跳过安全校验能大幅提速
            skip_check = (s_pos is not None)

            res = self.run_episode(
                model=model, start_pos=s_pos, goal_pos=g_pos,
                deterministic=deterministic, safe_threshold=safe_threshold,
                skip_spawn_check=skip_check, verbose=verbose
            )
            results.append(res)

            # 双模态日志输出：verbose 时打印细节，否则用 \r 覆盖当前行做进度条
            if verbose:
                status = "✅" if res["success"] else "❌"
                print(f"  -> Result: {status} | Final Reward: {res['reward']:.2f} | Steps: {res['steps']}")
            else:
                status = "✅" if res["success"] else "❌"
                print(f"\r[Episode {i+1}/{num_episodes}] {status} Reward: {res['reward']:.2f} | Steps: {res['steps']}", end="", flush=True)

        if not verbose:
            print()  # 进度条结束后换行

        # 汇总统计指标
        total_success = sum(1 for r in results if r["success"])
        return {
            "num_episodes": num_episodes,
            "success_rate": (total_success / num_episodes) * 100 if num_episodes > 0 else 0,
            "mean_reward": np.mean([r["reward"] for r in results]) if results else -1,
            "mean_steps": np.mean([r["steps"] for r in results]) if results else -1,
            "details": results
        }

    def close(self):
        """
        🛑 环境生命周期终结点。

        ⚠️ 为什么这里只做了停车，没有调用 rclpy.shutdown() 或 destroy_node()？
        
        新架构下的生命周期归属：
            Runtime   → 掌管所有 Node 的生杀大权（init / shutdown / executor）
            Bridge    → Runtime 的租户，自己不管自己的生死
        
        如果 Env 在 close() 里顺手把 RobotBridge / GazeboSimulator 的底层 Node 杀掉，
        Runtime 的线程池里就会出现悬空引用，导致其他正在运行的租户（如 Streamer）
        瞬间段错误崩溃。
        
        正确的关闭顺序由上层业务代码统一编排：
            runtime.shutdown()  → 一键销毁所有 Node + 终止 executor + rclpy.shutdown()
        
        这里只做最纯粹的逻辑收尾：停车。
        """
        self.send_velocity(0.0, 0.0)

    # ==================== 内部逻辑 ====================

    def _sample_safe_position(self, allowed_zones: list = None) -> Tuple[float, float]:
        """
        基于极坐标的安全区域随机采样。
        算法说明：在指定圆心和半径的安全区内，通过随机角度和随机半径生成坐标。
        注意：均匀分布在面积上时，半径需要取 sqrt(random)，这里直接取 uniform 
             会导致点集中在圆心附近，但在 RL 任务中这种轻微偏差通常可接受。
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
        """计算二维欧几里得距离。"""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _goal_dist(self, odom: Dict) -> float:
        """计算机器人当前位姿到目标点的直线距离。"""
        return self._euclidean(
            (self.state["goal_x"], self.state["goal_y"]), 
            (odom["x"], odom["y"])
        )

    def _check_collision(self) -> bool:
        """
        基于激光雷达的碰撞检测。
        说明：直接拿 360 维原始数据算最小距离，和奖励函数的 _min_laser 保持同源。
        """
        # 1. 获取当前最小距离（委托给 Bridge）
        min_dist = self.robot_bridge.get_laser_min_dist(
            max_range=self.robot["laser_range_max"]
        )

        # 2. 物理连续性校验（防自身死角瞬移）
        if self._last_min_dist is not None:
            dist_diff = self._last_min_dist - min_dist
            if dist_diff > self.max_step_dist:
                logger.debug(f"🚫 物理不可能跳变: {self._last_min_dist:.3f} -> {min_dist:.3f}, 忽略!")
                self._last_min_dist = min_dist
                return False

        # 3. 更新历史记录
        self._last_min_dist = min_dist

        # 4. 碰撞判定（核心逻辑下沉至 Bridge）
        if self.robot_bridge.is_collision(
            threshold=self.robot["proximity_to_collision_threshold"],
            max_range=self.robot["laser_range_max"]
        ):
            logger.debug(f"Collision! Min dist: {min_dist:.3f}m")
            return True
        return False

    def _check_spawn_pos(self, timeout=3.0, threshold=None):
        """
        重置后的出生点安全性校验。
        设计考量：Gazebo 在执行 set_entity_state 瞬移后，物理引擎的碰撞状态需要几帧时间更新。
                 这里采用轮询机制，等待机器人周围的激光读数稳定且大于安全阈值。
        """
        if threshold is None:
            threshold = self.robot["proximity_to_be_safe_min"] * 0.9

        timeout_deadline = time.time() + timeout
        while time.time() < timeout_deadline:
            laser_norm = self._get_laser_data()
            min_dist = float(np.min(laser_norm)) * self.robot["laser_range_max"]
            if min_dist > threshold:
                return True
            time.sleep(0.1)

        logger.warning(f"Spawn check timeout. Min dist: {min_dist:.3f}m < Threshold: {threshold:.3f}m")
        return False

    def _check_goal_pos(self, goal_x: float, goal_y: float) -> bool:
        """
        目标点有效性校验。
        说明：确保生成的目标点落在配置文件定义的安全圆内，防止目标点刷在墙壁或障碍物上。
        """
        for sub_zones in self.zones.values():
            for zone in sub_zones:
                dist_to_center = self._euclidean((goal_x, goal_y), (zone["cx"], zone["cy"]))
                if dist_to_center <= zone["r"]:
                    return True
        logger.warning(f"Goal ({goal_x:.2f}, {goal_y:.2f}) is outside all safe zones!")
        return False

    def _min_laser(self) -> float:
        """
        获取原始激光点中的最小距离。
        （委托给 RobotBridge 的语义化方法）
        """
        return self.robot_bridge.get_laser_min_dist(self.robot["laser_range_max"])

    def _get_obs(self) -> np.ndarray:
        """
        状态空间拼接函数 (维度: 38)
        布局说明：
        [0:24]   -> 激光雷达 (24维, 归一化)
        [24:27]  -> IMU 加速度 (3维, 除以物理最大值截断)
        [27:30]  -> IMU 角速度 (3维, 除以物理最大值截断)
        [30:32]  -> IMU 姿态角 Roll/Pitch (2维, 除以45度截断)
        [32]     -> 目标相对偏航角 (1维, 除以PI归一化)
        [33]     -> 目标相对距离 (1维, 除以最大裁剪距离归一化)
        [34:36]  -> 底盘当前速度 (2维, 除以最大速度归一化)
        [36:38]  -> 上一步动作 (2维, 除以最大速度归一化)
        """
        laser_norm = self._get_laser_data()
        odom = self._get_odom_data()
        # imu  = self._get_imu_data()

        # 初始化零值占位符，防止传感器首帧未就绪时数组拼接报错
        imu_acc = np.zeros(3, dtype=np.float32)
        imu_gyro = np.zeros(3, dtype=np.float32)
        imu_rp = np.zeros(2, dtype=np.float32)
        goal_angle, goal_dist = 0.0, 1.0
        odom_vel = np.zeros(2, dtype=np.float32)

        # 委托给 RobotBridge 的语义化方法
        imu_norm = self.robot_bridge.get_imu_normalized(
            acc_max=self.robot["lin_acc_physics_max"],
            gyro_max=self.robot["ang_vel_imu_physics_max"]
        )
        imu_acc = imu_norm["acc"]
        imu_gyro = imu_norm["gyro"]
        imu_rp = imu_norm["rpy"]

        goal_rel = self.robot_bridge.get_goal_relative(self.state["goal_x"], self.state["goal_y"])
        goal_angle = goal_rel["angle"] / math.pi
        goal_dist = min(goal_rel["dist"] / self.world["dist_to_goal_clip_norm"], 1.0)

        if odom is not None:
            # 当前线速度和角速度归一化
            odom_vel = np.clip(
                np.array([odom['vx'], odom['wz']]) / np.array([self.robot["lin_vel_max"], self.robot["ang_vel_max"]]),
                -1.0, 1.0
            )
        else:
            odom_vel = np.zeros(2, dtype=np.float32)

        # 历史动作归一化
        last_act_norm = np.clip(
            np.array([self.state["last_action"][0], self.state["last_action"][1]]) /
            np.array([self.robot["lin_vel_max"], self.robot["ang_vel_max"]]),
            -1.0, 1.0
        )

        # 按 38 维顺序严格拼接
        obs = np.concatenate([
            laser_norm, imu_acc, imu_gyro, imu_rp,
            [goal_angle], [goal_dist], odom_vel, last_act_norm
        ]).astype(np.float32)
        return obs
