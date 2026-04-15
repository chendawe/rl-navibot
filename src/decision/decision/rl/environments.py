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
from core.bridges.gazebo_ros2_bridge import GazeboRos2Bridge
from decision.rl.rewards import RewardContext, get_reward_model

logger = logging.getLogger(__name__)


def fetch_tb3_urdf() -> str:
    """
    在主进程中提前拉取完整的 URDF。
    
    【架构设计说明】
    重构前：这里需要实例化 Node、拼装 GetParameters.Request、处理异步 Future。
    重构后：完全剥离 ROS2 依赖，仅通过 Bridge 暴露的“业务意图 API”获取。
    这就是防腐层的威力——让非通信专业的算法代码，看起来像在读纯 Python 脚本。
    """
    # 1. 触发单例：因为是第一次在主进程调用，这里会真正执行 rclpy.init() 并开启后台 spin 线程
    bridge = GazeboRos2Bridge(node_name='urdf_prefetcher')
    
    # 2. 意图表达：我不关心 ROS2 怎么找节点、怎么建 Client，我只要“远程节点的某个参数”
    urdf_string = bridge.get_remote_parameter(
        node_fragment='robot_state_publisher', 
        param_name='robot_description',
        timeout_sec=10.0
    )
    
    # 3. 兜底防御：防止 Gazebo 模型加载异常导致后续训练全盘崩溃
    if not urdf_string:
        raise RuntimeError("成功连接到 ROS2 节点，但获取到的 robot_description 为空！请检查 Gazebo 模型是否正常加载。")
        
    logger.info(f"Successfully fetched URDF ({len(urdf_string)} chars)")
    return urdf_string



class TurtleBot3NavEnv(gym.Env):
    """
    """
    metadata = {"render_modes": []}
    
    SAFE_ZONES = {"default": [{"cx": 0, "cy":  0, "r": 1.}]}
    
    def __init__(self, robot_urdf: Optional[str] = None, env_config: Optional[Dict[str, Any]] = None):
        gym.Env.__init__(self) # ★ 只继承 Gym 了！
        
        self._robot_urdf = robot_urdf

        # ================= 1. 默认配置与合并 (完全不变) =================
        default_env_config = {
            "robot": {
                "name": "burger", "laser_range_max": 3.5, "laser_beams_num": 24,
                "laser_noise_threshold": 0.08,                              # ★ 新增
                "lin_vel_max": 0.22, "ang_vel_max": 1.5,
                "lin_vel_stuck_threshold": 0.12, "lin_acc_physics_max": 2.0,
                "ang_vel_imu_physics_max": 3.0, "proximity_to_collision_threshold": 0.11,
                "proximity_to_be_safe_min": 0.30,
            },
            "world": {
                "name": "ttb3_world", "episode_steps_max": 500, "step_duration": 0.1,
                "dist_to_goal_threshold": 0.30,                            # ★ 0.35 → 0.30 同步 YAML
                "dist_to_goal_gen_min": 1.0, "dist_to_goal_clip_norm": 5.0,
                "safe_zones": {},
            },
            "reward": {
                "name": "ttb3_world",
                "reward_at_goal": 300.0, "penalty_at_collision": -300.0,
                "reward_factor_approaching_goal": 20.0,                    # ★ 5.0 → 20.0 同步 YAML
                "penalty_elapsing_time": -0.2,                             # ★ -0.5 → -0.2 同步 YAML
                "penalty_stuck": -0.05,                                    # ★ -2.0 → -0.05 同步 YAML
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

        self.config = {"robot": default_env_config["robot"], "world": default_env_config["world"], "reward": default_env_config["reward"]}
        self.robot = self.config["robot"]
        self.world = self.config["world"]
        self.rew = self.config["reward"]
        reward_name = self.config["reward"]["name"]
        self.reward_fn = get_reward_model(reward_name, self.config)
        
        self.zones = self.world.pop("safe_zones", {})
        if not self.zones: raise ValueError(f"配置中未找到 safe_zones！")

        # ================= 2. 初始化通信桥 (替代原 Node + Spin) =================
        self.bridge = GazeboRos2Bridge(node_name=f'{self.robot["name"]}_gym_env')

        # ================= 3. 纯净接口声明 (完全剥离 ROS 细节) =================
        # 🔥 核心改变：环境层只管“要什么”，不管“怎么连”。
        # QoS策略、消息类型转换、甚至激光去噪逻辑，全部推给 Bridge！
        
        self.bridge.setup_sensors(
            laser_topic='/scan',
            imu_topic='/imu',
            odom_topic='/odom',
            laser_noise_threshold=self.robot["laser_noise_threshold"]  # 把你原来写死的 0.15 作为配置传进去
        )
        
        self.bridge.setup_actuators(
            cmd_vel_topic='/cmd_vel',
            goal_topic='/goal_pose'
        )
        
        self.bridge.setup_sim_services(
            reset_world='/reset_world',
            set_entity='/set_entity_state',
            delete_entity='/delete_entity',
            spawn_entity='/spawn_entity',
            entity_name=self.robot["name"]  # 告诉 Bridge 我们要控制谁（比如 'burger'）
        )
        
        # ================= 4. RL 空间定义 (完全不变) =================
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(38,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-self.robot["lin_vel_max"], -self.robot["ang_vel_max"]], dtype=np.float32),
            high=np.array([ self.robot["lin_vel_max"],  self.robot["ang_vel_max"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = obs_space
        self.action_space = act_space

        # ================= 5. 运行时状态大一统 (完全不变) =================
        self.state = {
            "goal_x": 0.0, "goal_y": 0.0, "prev_dist": 0.0,
            "step": 0, "last_action": np.array([0.0, 0.0], dtype=np.float32),
        }
        self.max_step_dist = self.robot["lin_vel_max"] * self.world["step_duration"] * 2.0
        self._last_min_dist = None
        
        self.print_config()

    def print_config(self):
        print("\n" + "=" * 60)
        print("🤖 Turtlerobot3 Navigation Env Configuration")
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
        # 🔥 修改点：适配重构后的标准 Gym 属性
        obs_shape = self.observation_space.shape
        act_low  = self.action_space.low
        act_high = self.action_space.high
        print(f"  - {'Obs Shape':<35} : {obs_shape}")
        print(f"  - {'Act Space':<35} : [{act_low[0]:.2f}, {act_high[0]:.2f}], [{act_low[1]:.2f}, {act_high[1]:.2f}]")
        
        print("=" * 60 + "\n")



    # ==================== 数据获取与控制 (彻底解耦版) ====================
    # 🧹 清理说明：
    # 1. 删除了 _spin, _scan_cb, _imu_cb, _odom_cb (全在 Bridge 后台线程里了)
    # 2. 删除了所有的 with self._lock: (Bridge 的 get_data 内部已经封装了线程安全)
    # 3. 删除了所有的 ROS2 消息体解包逻辑 (如 msg.pose.pose.position.x)
    # 🏗️ 架构定位：这一层只负责“状态空间适配”，绝不碰“通信协议”。

    def _get_odom_data(self) -> Optional[Dict]:
        """
        获取底盘位姿与速度信息。
        ★ 以前：拿 msg -> 手动拆 p.x, o.w -> 算 atan2 -> 处理异常
        ★ 现在：直接拿算好的纯字典，业务层零感知底层是 Odometry 还是 TF
        """
        return self.bridge.get_odom_data()

    def _get_imu_data(self) -> Optional[Dict]:
        """
        获取 IMU 姿态与角速度信息。
        ★ 以前：拿 msg -> 手动拆 a.x, o.x -> 算 asin 处理奇点
        ★ 现在：直接拿算好的纯列表，Bridge 内部已经用 np.clip 保护了 asin 域
        """
        return self.bridge.get_imu_data()

    def _get_laser_data(self) -> np.ndarray:
        """
        获取降维、归一化后的激光雷达状态。
        这是 RL 环境中最容易炸裂的地方（NaN 传播、维度不匹配），这里的防御逻辑是保命符。
        """
        beams_num = self.robot["laser_beams_num"]
        max_range = self.robot["laser_range_max"]
        
        # ★ 以前：msg = self.bridge.get_data('/scan') 然后 msg.ranges
        # ★ 现在：Bridge 直接丢一个纯净的 np.ndarray 过来
        ranges = self.bridge.get_laser_ranges()
        
        # 1. 防御性兜底：如果仿真器第一帧还没来得及发数据，用满量程填充（代表周围绝对安全）
        if ranges is None:
            return np.ones(beams_num, dtype=np.float32)
            
        # ======= 2. 数据清洗 (必做！Gazebo 的 Ray 传感器偶尔会抽风) =======
        # 🛡️ 作用：把 NaN (非数字)、Inf (无穷大) 替换为合法值，防止它们进入神经网络导致梯度爆炸
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        # 🛡️ 作用：硬性截断，防止物理引擎穿透时出现负数距离
        ranges = np.clip(ranges, 0, max_range)
        
        # ======= 3. RL 状态空间适配 (核心逻辑) =======
        # 💡 为什么下采样留在 Env？因为这是 RL 状态空间的定义（38维），
        # 换成 Isaac Sim 可能底层传来的就是 24 维，不需要下采样。
        # 把下采样放在这里，换仿真器时只需改这里，不用去动底层通信桥。
        
        n = len(ranges)
        step = max(1, n // beams_num)
        truncated_len = (n // step) * step
        
        # 🎯 算法解释：为什么用 .min(axis=1) 而不是 .mean()？
        # 因为对于避障任务，漏掉一个近处障碍物是致命的。按区域取最小值，能最大程度保留危险特征。
        sampled = ranges[:truncated_len].reshape(-1, step).min(axis=1)
        
        # 🛡️ 维度强一致性校验：无论底层雷达发了多少束光，出去的必须是 38 维
        if len(sampled) > beams_num:
            sampled = sampled[:beams_num]
        elif len(sampled) < beams_num:
            sampled = np.pad(sampled, (0, beams_num - len(sampled)), 'constant', constant_values=max_range)
            
        # 归一化到 [0, 1]，神经网络最喜欢的输入区间
        return sampled / max_range

    def send_velocity(self, lin_x: float, ang_z: float):
        """
        下发底盘速度指令。
        ★ 以前：msg = Twist(); msg.linear.x = ... (需要 import geometry_msgs)
        ★ 现在：传两个 float 进去就行，彻底告别 Env 层对 ROS 消息类型的依赖
        """
        self.bridge.send_velocity(lin_x, ang_z)

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
    
    def _wait_for_services(self) -> bool:
        """
        幂等服务检查器。
        🛡️ 防御设计：RL 训练动辄几万步，不可能每次 reset 都去查一次服务。
        采用单次检查加缓存标志 (_services_ready) 的策略，把通信开销降到最低。
        """
        if hasattr(self, '_services_ready') and self._services_ready:
            return True
            
        logger.info("Waiting for Gazebo reset services...")
        try:
            # ★ 以前：自己写 while 循环查 client.service_is_ready()
            # ★ 现在：超时逻辑、异常抛出全在 Bridge 内部消化，Env 只管拿 True/False
            self.bridge.wait_for_sim_services(timeout_sec=5.0)
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
        因为 Bridge 是全局单例，且后台线程在持续高频泵血覆盖 _data_store。
        如果在 reset 时强行清空缓存，会导致紧随其后的 _get_obs() 读到 None，
        从而触发不必要的 NaN 保护逻辑。让物理引擎自然刷写数据才是最安全的。
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

    # 🔥 _call_gazebo_async 和 _teleport_robot → 整个删除！
    # 它们的职责已经被 bridge.reset_world() 和 bridge.set_robot_pose() 完全吸收。
    # Env 层不再持有任何 ROS2 Service 的 Request 类型依赖。

    def _generate_valid_task(self, opts) -> Optional[Dict]:
        """
        核心任务生成器：采样 -> 仿真重置 -> 安全校验。
        🎯 核心改变：这里从“ROS 指令发送器”彻底退化成了“纯业务逻辑调度器”。
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
            # ★ 现在：两行纯 Python 语义调用。这就是防腐层带来的震撼！
            self.bridge.reset_world()
            self.bridge.set_robot_pose(s_x, s_y, s_yaw)
            
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
        🛡️ 鲁棒性保证：在 RL 分布式训练中，环境崩溃是常态，但“训练进程退出”是绝对不允许的。
        即使 Gazebo 全乱套了，也要强行给算法吐出一个合法的 (obs, info)，哪怕是原地重置。
        """
        if opts["start_pos"] is not None:
            logger.error(f"Failed to set custom start pose: {opts['start_pos']}")
            s_x, s_y, s_yaw = opts["start_pos"]
        else:
            logger.warning("All retries failed! Fallback to (0.5, 0.5).")
            s_x, s_y, s_yaw = 0.5, 0.5, 0.0

        # 即使是兜底，也要保持调用接口的一致性
        self.bridge.reset_world()
        self.bridge.set_robot_pose(s_x, s_y, s_yaw)

        g_x, g_y = opts["goal_pos"] if opts["goal_pos"] else (0.0, 0.0)
        return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}

    def reset(self, seed=None, options=None):
        """
        标准 Gymnasium Reset 接口。
        """
        # 1. 调用父类处理 seed，确保 self.np_random 被正确初始化（Gym 强制规范）
        super().reset(seed=seed, options=options)
        
        if self._robot_urdf is None:
            raise ValueError("robot_urdf is required")

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
        经过重构，这里不再有任何 ROS2 的痕迹，变成了一段纯正的“状态机推演”代码。
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
        if odom is None: return obs, -1.0, False, True, {"error": "odom unavailable"}

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
            last_action=self.state["last_action"], # 取上一步的动作用于计算平滑度惩罚
            odom=odom,
            imu=imu,
            min_laser=self._min_laser() # 提前算好最小距离传进去
        )
        
        # ==========================================
        # 🔥 计算奖励 (告别内部臃肿的 if-else)
        # ==========================================
        # 现在的 step 函数根本不知道奖励是怎么算的，它只负责当个传话筒。
        reward = self.reward_fn.compute(ctx)

        # 5. 推进内部状态机
        self.state["prev_dist"] = curr_dist
        self.state["last_action"] = action.copy()
        self.state["step"] += 1

        # 6. 封装 Gym 标准五元组返回值
        terminated = goal_reached or collision
        truncated = self.state["step"] >= self.world["episode_steps_max"]

        # 🛑 刹车逻辑：一旦 episode 结束，必须清零速度，防止 Gazebo 里机器人在下次 reset 前乱撞
        if terminated or truncated:
            self.send_velocity(0.0, 0.0)

        return obs, float(reward), terminated, truncated, {
            "collision": collision, "goal_reached": goal_reached,
            "final_dist": curr_dist, "episode_step": self.state["step"]
        }

    def run_episode(self, model, start_pos=None, goal_pos=None, deterministic=True, safe_threshold=None, skip_spawn_check=False, verbose=False):
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

    def run_episodes(self, model, num_episodes=10, start_pos_list=None, goal_pos_list=None, deterministic=True, safe_threshold=None, verbose=False):
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
                
        if not verbose: print() # 进度条结束后换行

        # 汇总统计指标
        total_success = sum(1 for r in results if r["success"])
        return {
            "num_episodes": num_episodes,
            "success_rate": (total_success / num_episodes) * 100 if num_episodes > 0 else 0,
            "mean_reward": np.mean([r["reward"] for r in results]) if results else -1,
            "mean_steps": np.mean([r["steps"] for r in results]) if results else -1,
            "details": results
        }

    # def close(self):
    #     self.send_velocity(0.0, 0.0)
    #     if rclpy.ok():
    #         self.destroy_node()

    def close(self):
        """
        🛑 环境生命周期终结点。
        
        ⚠️ 为什么删掉了 rclpy.shutdown() 和 destroy_node()？
        因为 `GazeboRos2Bridge` 是全局单例！
        在 RL 分布式训练（如 Ray）或多线程 Web 请求中，如果 Env 被 GC 回收时顺手把底层 Node 杀了，
        会导致其他正在使用该 Bridge 的进程瞬间底层崩溃（段错误 Segfault）。
        
        真正的 ROS 资源释放，应该交给操作系统的进程退出机制去兜底。
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
        说明：_get_laser_data() 返回的是归一化到 [0, 1] 的数据，
             因此需要乘以 laser_range_max 还原为真实物理距离，再与碰撞阈值比较。
        """
        # 1. 获取当前最小距离
        min_dist = self._min_laser()
        
        # 2. 物理连续性校验（防自身死角瞬移）
        if self._last_min_dist is not None:
            dist_diff = self._last_min_dist - min_dist
            if dist_diff > self.max_step_dist:
                logger.debug(f"🚫 物理不可能跳变: {self._last_min_dist:.3f} -> {min_dist:.3f}, 忽略!")
                self._last_min_dist = min_dist
                return False
        
        # 3. 更新历史记录
        self._last_min_dist = min_dist
        
        # laser_norm = self._get_laser_data()
        # min_dist = float(np.min(laser_norm)) * self.robot["laser_range_max"]
        
        # if min_dist < self.robot["proximity_to_collision_threshold"]:
        #     logger.debug(f"Collision detected. Min dist: {min_dist:.3f}m")
        #     return True
        # return False
        
        # ★ 修改前：laser_norm = self._get_laser_data() (24维下采样数据)
        # ★ 修改后：直接拿 360 维原始数据算，和奖励函数的 _min_laser 保持同源
        min_dist = self._min_laser() 
        
        if min_dist < self.robot["proximity_to_collision_threshold"]:
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
        用途：主要供给 Reward 函数计算接近障碍物的惩罚项。
        """
        ranges = self.bridge.get_laser_ranges()
        if ranges is None: 
            return self.robot["laser_range_max"]
            
        # 必须过滤 NaN 和 Inf，否则 np.min 会抛出异常或返回异常值导致奖励崩溃
        ranges = np.nan_to_num(ranges, nan=self.robot["laser_range_max"], posinf=self.robot["laser_range_max"])
        return float(np.min(ranges))

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
        imu  = self._get_imu_data()

        # 初始化零值占位符，防止传感器首帧未就绪时数组拼接报错
        imu_acc = np.zeros(3, dtype=np.float32)
        imu_gyro = np.zeros(3, dtype=np.float32)
        imu_rp = np.zeros(2, dtype=np.float32)
        goal_angle, goal_dist = 0.0, 1.0
        odom_vel = np.zeros(2, dtype=np.float32)

        if imu is not None:
            imu_acc = np.clip(np.array(imu['acc']) / self.robot["lin_acc_physics_max"], -1.0, 1.0)
            imu_gyro = np.clip(np.array(imu['gyro']) / self.robot["ang_vel_imu_physics_max"], -1.0, 1.0)
            imu_rp = np.clip(np.array(imu['rpy'][:2]) / (math.pi/4), -1.0, 1.0)

        if odom is not None:
            # 坐标系转换：将世界坐标系下的目标偏移量，转换为机器人本体坐标系下的偏移量
            dx = self.state["goal_x"] - odom['x']
            dy = self.state["goal_y"] - odom['y']
            yaw = odom['yaw']
            # 旋转矩阵二维展开：[cos(yaw), sin(yaw); -sin(yaw), cos(yaw)] * [dx, dy]^T
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
            
            # 本体坐标系下的角度和距离
            goal_angle = math.atan2(local_y, local_x) / math.pi
            goal_dist = min(math.hypot(dx, dy) / self.world["dist_to_goal_clip_norm"], 1.0)
            
            # 当前线速度和角速度归一化
            odom_vel = np.clip(
                np.array([odom['vx'], odom['wz']]) / np.array([self.robot["lin_vel_max"], self.robot["ang_vel_max"]]), 
                -1.0, 1.0
            )

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

# def fetch_tb3_urdf() -> str:
#     """
#     在主进程中提前拉取完整的 URDF。
#     重构说明：直接复用 GazeboRos2Bridge 的单例节点和后台 Spin 线程，
#     彻底抛弃临时建节点和手动 spin 的脏活累活。
#     """
#     # 1. 拿到全局单例（如果没 init，它会帮你处理）
#     bridge = GazeboRos2Bridge(node_name='urdf_prefetcher')
#     logger = bridge.node.get_logger()
#     node = bridge.node  # 拿到底层真实节点
    
#     urdf_string = None

#     # ====== 第一步：找到目标节点名 ======
#     target_node = None
#     for attempt in range(50):  # 最多等5秒
#         nodes = node.get_node_names()
#         for n in nodes:
#             if 'robot_state_publisher' in n or 'state_publisher' in n:
#                 target_node = n
#                 break
#         if target_node:
#             break
#         time.sleep(0.1)

#     if not target_node:
#         all_nodes = node.get_node_names()
#         raise RuntimeError(f"No robot_state_publisher found! Visible nodes: {all_nodes}")

#     logger.info(f"Found target node: {target_node}")

#     # ====== 第二步：等 service 可用 ======
#     service_name = f'{target_node}/get_parameters'
#     param_client = node.create_client(GetParameters, service_name)

#     logger.info(f"Waiting for service {service_name} ...")
#     if not param_client.wait_for_service(timeout_sec=5.0):
#         node.destroy_client(param_client)
#         raise RuntimeError(f"Service {service_name} not available after 5s!")

#     logger.info("Service ready, requesting URDF...")

#     # ====== 第三步：调用服务拿 URDF ======
#     req = GetParameters.Request()
#     req.names = ['robot_description']
#     future = param_client.call_async(req)

#     # 🔥 核心重构点：不用再写 rclpy.spin_once 了！
#     # 因为 Bridge 的后台守护线程已经在 0.01 秒一次地 spin，会自动帮我们把 future 的结果填上。
#     # 我们只需要单纯地等它干完活就行。
#     start = time.time()
#     while not future.done() and time.time() - start < 5.0:
#         time.sleep(0.05)

#     if future.done() and future.result() is not None:
#         values = future.result().values
#         if values:
#             urdf_string = values[0].string_value

#     # 保持干净，用完销毁客户端（节点不能销毁，留给后续用）
#     node.destroy_client(param_client)

#     if urdf_string is None:
#         raise RuntimeError("Service call succeeded but robot_description was empty!")

#     logger.info(f"Successfully fetched URDF ({len(urdf_string)} chars)")
#     return urdf_string


# class Turtlerobot3NavEnv(gym.Env):
#     """
#     """
#     metadata = {"render_modes": []}
    
#     SAFE_ZONES = {"default": [{"cx": 0, "cy":  0, "r": 1.}]}
    
#     def __init__(self, robot_urdf: Optional[str] = None, env_config: Optional[Dict[str, Any]] = None):
#         gym.Env.__init__(self) # ★ 只继承 Gym 了！
        
#         self._robot_urdf = robot_urdf

#         # ================= 1. 默认配置与合并 (完全不变) =================
#         default_env_config = {
#             "robot": { "name": "burger", "laser_range_max": 3.5, "laser_beams_num": 24, "lin_vel_max": 0.22, "ang_vel_max": 1.5,
#                         "lin_vel_stuck_threshold": 0.12, "lin_acc_physics_max": 2.0, "ang_vel_imu_physics_max": 3.0,
#                         "proximity_to_collision_threshold": 0.11, "proximity_to_be_safe_min": 0.30 },
#             "world": { "name": "ttb3_world", "episode_steps_max": 500, "step_duration": 0.1,
#                           "dist_to_goal_threshold": 0.35, "dist_to_goal_gen_min": 1.0, "dist_to_goal_clip_norm": 5.0,
#                           "safe_zones": {} },
#             "reward": { "name": "ttb3_world", "reward_at_goal": 300.0, "penalty_at_collision": -300.0, "reward_factor_approaching_goal": 5.0,
#                        "penalty_elapsing_time": -0.5, "penalty_stuck": -2.0, "reward_good_orientation": 2.0,
#                        "penalty_factor_in_safe_proximity": -1.0, "penalty_instability": -2.0, "penalty_action_smoothness": -0.5 }
#         }
#         if env_config:
#             for key, value in env_config.items():
#                 if isinstance(value, dict) and key in default_env_config:
#                     default_env_config[key].update(value)
#                 else:
#                     default_env_config[key] = value

#         self.config = {"robot": default_env_config["robot"], "world": default_env_config["world"], "reward": default_env_config["reward"]}
#         self.robot = self.config["robot"]
#         self.world = self.config["world"]
#         self.rew = self.config["reward"]
#         reward_name = self.config["reward"]["name"]
#         self.reward_fn = get_reward_model(reward_name, self.config)
        
#         self.zones = self.world.pop("safe_zones", {})
#         if not self.zones: raise ValueError(f"配置中未找到 safe_zones！")

#         # ================= 2. 初始化通信桥 (替代原 Node + Spin) =================
#         self.bridge = GazeboRos2Bridge(node_name=f'{self.robot["name"]}_gym_env')

#         # ================= 3. ROS 接口大一统 (使用 Bridge 的 API) =================
#         qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        
#         # 定义 LaserScan 的预处理函数（过滤噪点）
#         def filter_scan_noise(msg):
#             for i in range(len(msg.ranges)):
#                 if msg.ranges[i] < 0.15:
#                     msg.ranges[i] = msg.range_max

#         # 订阅
#         self.bridge.create_subscriber('/scan', LaserScan, preprocess_cb=filter_scan_noise, qos=qos)
#         self.bridge.create_subscriber('/imu', Imu, qos=qos)
#         self.bridge.create_subscriber('/odom', Odometry, qos=qos)
        
#         # 发布
#         self.vel_pub = self.bridge.create_publisher('/cmd_vel', Twist, 10)
#         self.goal_pub = self.bridge.create_publisher('/goal_pose', PoseStamped, 10)
        
#         # 服务客户端 (Bridge 暂时不封装复杂的 Client 调用，直接用底层 node 创建即可)
#         self.ros_clients = {
#             "reset_world": self.bridge.node.create_client(Empty, '/reset_world'),
#             "set_entity":  self.bridge.node.create_client(SetEntityState, '/set_entity_state'),
#             "delete":      self.bridge.node.create_client(DeleteEntity, '/delete_entity'),
#             "spawn":       self.bridge.node.create_client(SpawnEntity, '/spawn_entity'),
#         }

#         # ================= 4. RL 空间定义 (完全不变) =================
#         obs_space = spaces.Box(low=-1.0, high=1.0, shape=(38,), dtype=np.float32)
#         act_space = spaces.Box(
#             low=np.array([-self.robot["lin_vel_max"], -self.robot["ang_vel_max"]], dtype=np.float32),
#             high=np.array([ self.robot["lin_vel_max"],  self.robot["ang_vel_max"]], dtype=np.float32),
#             dtype=np.float32,
#         )
#         self.observation_space = obs_space
#         self.action_space = act_space

#         # ================= 5. 运行时状态大一统 (完全不变) =================
#         self.state = {
#             "goal_x": 0.0, "goal_y": 0.0, "prev_dist": 0.0,
#             "step": 0, "last_action": np.array([0.0, 0.0], dtype=np.float32),
#         }
#         self.print_config()

#     def print_config(self):
#         print("\n" + "=" * 60)
#         print("🤖 Turtlerobot3 Navigation Env Configuration")
#         print("=" * 60)
        
#         print(f"robot Name   : {self.robot.get('name', 'N/A')}")
#         print(f"World Name : {self.world.get('name', 'N/A')}")
        
#         print("-" * 60)
#         print("📡 Env robot Config (Physics & Hardware):")
#         for k, v in self.robot.items():
#             print(f"  - {k:<35} : {v}")
            
#         print("-" * 60)
#         print("🗺️  Env World Config (Task & Map):")
#         for k, v in self.world.items():
#             print(f"  - {k:<35} : {v}")
            
#         zone_names = list(self.zones.keys())
#         total_points = sum(len(pts) for pts in self.zones.values())
#         print(f"  - {'safe_zones':<35} : {total_points} points in {len(zone_names)} zones")
#         print(f"    {'Zones:':<33} : {zone_names}")
                
#         print("-" * 60)
#         print("🎁 Reward Config:")
#         for k, v in self.rew.items():
#             print(f"  - {k:<35} : {v}")
            
#         print("-" * 60)
#         print("🧠 RL Core Config:")
#         # 🔥 修改点：适配重构后的标准 Gym 属性
#         obs_shape = self.observation_space.shape
#         act_low  = self.action_space.low
#         act_high = self.action_space.high
#         print(f"  - {'Obs Shape':<35} : {obs_shape}")
#         print(f"  - {'Act Space':<35} : [{act_low[0]:.2f}, {act_high[0]:.2f}], [{act_low[1]:.2f}, {act_high[1]:.2f}]")
        
#         print("=" * 60 + "\n")



#     # ==================== 数据获取与控制 (彻底解耦版) ====================
#     # 🧹 清理说明：
#     # 1. 删除了 _spin, _scan_cb, _imu_cb, _odom_cb (全在 Bridge 里了)
#     # 2. 删除了所有的 with self._lock: (Bridge 的 get_data 内部自带线程安全)

#     def _get_odom_data(self) -> Optional[Dict]:
#         msg = self.bridge.get_data('/odom') # ★ 一行拿数据，不用管锁
#         if msg is None: return None
#         p = msg.pose.pose.position
#         o = msg.pose.pose.orientation
#         t = msg.twist.twist
        
#         siny_cosp = 2 * (o.w * o.z + o.x * o.y)
#         cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
#         yaw = math.atan2(siny_cosp, cosy_cosp)

#         return {
#             "x": p.x, "y": p.y, "yaw": yaw,
#             "vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z
#         }

#     def _get_imu_data(self) -> Optional[Dict]:
#         msg = self.bridge.get_data('/imu') # ★ 一行拿数据
#         if msg is None: return None
#         a = msg.linear_acceleration
#         g = msg.angular_velocity
#         o = msg.orientation
        
#         sinr_cosp = 2 * (o.w * o.x + o.y * o.z)
#         cosr_cosp = 1 - 2 * (o.x * o.x + o.y * o.y)
#         roll = math.atan2(sinr_cosp, cosr_cosp)
        
#         sinp = 2 * (o.w * o.y - o.z * o.x)
#         pitch = math.asin(np.clip(sinp, -1.0, 1.0))

#         return {
#             "acc": [a.x, a.y, a.z],
#             "gyro": [g.x, g.y, g.z],
#             "rpy": [roll, pitch]
#         }

#     def _get_laser_data(self) -> np.ndarray:
#         beams_num = self.robot["laser_beams_num"]
#         max_range = self.robot["laser_range_max"]
        
#         msg = self.bridge.get_data('/scan') # ★ 一行拿数据，而且噪点已经在 Bridge 注册时被过滤了
#         if msg is None:
#             return np.ones(beams_num, dtype=np.float32)
            
#         # 下面的业务逻辑完全保持你原本的高效写法
#         ranges = np.array(msg.ranges)
#         ranges = np.nan_to_num(ranges, nan=max_range, 
#                                posinf=max_range, neginf=0.0)
#         ranges = np.clip(ranges, 0, max_range)
        
#         # n = len(ranges)
#         # step = max(1, n // beams_num)
#         # sampled = np.array([np.min(ranges[i:i+step]) for i in range(0, n, step)])
#         # 极客写法（可选，仅为了跑得更快，可读性稍差）：
#         n = len(ranges)
#         step = max(1, n // beams_num)
#         # 截断到能被 step 整除的长度
#         truncated_len = (n // step) * step
#         sampled = ranges[:truncated_len].reshape(-1, step).min(axis=1)
        
#         if len(sampled) > beams_num:
#             sampled = sampled[:beams_num]
#         elif len(sampled) < beams_num:
#             sampled = np.pad(sampled, (0, beams_num - len(sampled)), 
#                              'constant', constant_values=max_range)
            
#         return sampled / max_range

#     def send_velocity(self, lin_x, ang_z):
#         msg = Twist()
#         msg.linear.x = float(lin_x)
#         msg.angular.z = float(ang_z)
#         # 🔥 替换：原来是自己维护的 pub 字典，现在统一走 Bridge
#         self.bridge.publish('/cmd_vel', msg)

#     def _get_zero_obs(self):
#         """
#         返回一个全零观测，用于环境异常时的兜底。
#         """
#         return np.zeros(self.observation_space.shape, dtype=np.float32)

#     # ==================== Gym 接口 ====================
#     # ==================== Reset 辅助函数 ====================
#     def _wait_for_services(self) -> bool:
#         if hasattr(self, '_services_ready') and self._services_ready: return True
#         # 🔥 修改：使用 Bridge 底层 Node 的标准 ROS2 Logger
#         logger = self.bridge.node.get_logger()
#         logger.info("Waiting for Gazebo reset services...")
#         try:
#             # 🔥 修改：self.ros["clients"] -> self.ros_clients
#             if not self.ros_clients["set_entity"].wait_for_service(timeout_sec=5.0):
#                 raise RuntimeError("SetEntityState service not found!")
#             if not self.ros_clients["reset_world"].wait_for_service(timeout_sec=5.0):
#                 raise RuntimeError("ResetWorld service not found!")
#             self._services_ready = True
#             logger.info("Gazebo services are ready!")
#             return True
#         except Exception as e:
#             logger.error(f"Service check failed: {e}")
#             return False

#     def _clear_sensor_cache(self):
#         # ⚠️ 重构重点：不再手动清空 self._latest_scan = None
#         # 原因：Bridge 是单例，和 Web 前端共享数据。
#         # 如果 Gym 在 Reset 时清空了雷达数据，前端的雷达可视化会瞬间闪烁消失。
#         # 由于 _get_laser_data() 已经对 None 做了全 1 兜底，这里直接 Pass 是最安全的做法。
#         pass

#     def _parse_reset_options(self, options: Optional[Dict]) -> Dict:
#         if not options: return {"start_pos": None, "goal_pos": None, "safe_threshold": None, "skip_spawn_check": False}
#         return {
#             "start_pos": options.get("start_pos"),
#             "goal_pos": options.get("goal_pos"),
#             "safe_threshold": options.get("safe_threshold"),
#             "skip_spawn_check": options.get("skip_spawn_check", False)
#         }

#     def _call_gazebo_async(self, req, client_key, timeout=3.0):
#         """统一的 Gazebo 异步调用封装，避免满屏幕的 while not future.done"""
#         # 🔥 修改：self.ros["clients"] -> self.ros_clients
#         future = self.ros_clients[client_key].call_async(req)
#         start = time.time()
#         while not future.done() and time.time() - start < timeout: 
#             time.sleep(0.05)
#         if not future.done():
#             raise RuntimeError(f"Gazebo service '{client_key}' timed out after {timeout}s!")
        
#     def _teleport_robot(self, x, y, yaw):
#         """纯粹的实体位置强行重置"""
#         req = SetEntityState.Request()
#         req.state.name = self.robot["name"]
#         req.state.pose.position.x = float(x)
#         req.state.pose.position.y = float(y)
#         req.state.pose.position.z = 0.0
#         req.state.pose.orientation.z = math.sin(yaw / 2.0)
#         req.state.pose.orientation.w = math.cos(yaw / 2.0)
#         req.state.reference_frame = "world"
#         self._call_gazebo_async(req, "set_entity")

#     def _generate_valid_task(self, opts) -> Optional[Dict]:
#         """带有重试逻辑的核心任务生成器"""
#         # 🔥 修改：局部获取 logger 提升循环内性能
#         logger = self.bridge.node.get_logger()
#         custom_start = opts["start_pos"]
#         target_goal = opts["goal_pos"]
#         use_custom_start = custom_start is not None
        
#         for attempt in range(5):
#             # 1. 确定候选起点
#             if use_custom_start:
#                 s_x, s_y, s_yaw = custom_start
#             else:
#                 s_x, s_y = self._sample_safe_position()
#                 s_yaw = self.np_random.uniform(-math.pi, math.pi)
                
#             # 2. 确定候选终点
#             g_x, g_y = target_goal if target_goal else self._sample_safe_position()
            
#             # 3. 距离校验 (仅限全随机时)
#             if not use_custom_start and not target_goal:
#                 if self._euclidean((s_x, s_y), (g_x, g_y)) < self.world["dist_to_goal_gen_min"]:
#                     continue
                    
#             # 4. 调用仿真器重置并传送
#             self._call_gazebo_async(std_srvs.srv.Empty.Request(), "reset_world")
#             self._teleport_robot(s_x, s_y, s_yaw)
#             time.sleep(0.5) # 等待物理引擎稳定和雷达刷新
            
#             # 5. 碰撞与安全性校验
#             is_spawn_safe = self._check_spawn_pos(threshold=opts["safe_threshold"]) if not opts["skip_spawn_check"] else True
#             is_goal_safe = self._check_goal_pos(g_x, g_y) if not target_goal else True
            
#             if is_spawn_safe and is_goal_safe:
#                 return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}
                
#             if not use_custom_start:
#                 # 🔥 修改：self.get_logger() -> logger
#                 logger.warn(f"Validation failed, retry {attempt+1}/5...")
                
#         return None

#     def _fallback_task(self, opts) -> Dict:
#         """重试耗尽后的兜底策略"""
#         # 🔥 修改：self.get_logger() -> logger
#         logger = self.bridge.node.get_logger()
#         if opts["start_pos"] is not None:
#             logger.error(f"Failed to set custom start pose: {opts['start_pos']}")
#             s_x, s_y, s_yaw = opts["start_pos"]
#         else:
#             logger.warn("All retries failed! Fallback to (0.5, 0.5).")
#             s_x, s_y, s_yaw = 0.5, 0.5, 0.0
            
#         self._call_gazebo_async(std_srvs.srv.Empty.Request(), "reset_world")
#         self._teleport_robot(s_x, s_y, s_yaw)
        
#         # 兜底目标：优先用用户指定的，否则给默认值防止报错
#         g_x, g_y = opts["goal_pos"] if opts["goal_pos"] else (0.0, 0.0)
#         return {"s_x": s_x, "s_y": s_y, "s_yaw": s_yaw, "g_x": g_x, "g_y": g_y}

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed, options=options)
#         if self._robot_urdf is None: raise ValueError("robot_urdf is required")
        
#         # 1. 环境就绪检查
#         if not self._wait_for_services(): return self._get_zero_obs(), {}
#         self._clear_sensor_cache()
        
#         # 2. 解析外部传入参数
#         opts = self._parse_reset_options(options)
        
#         # 3. 核心：生成并校验合法的任务(起点+终点)
#         task = self._generate_valid_task(opts)
#         if task is None:
#             task = self._fallback_task(opts)
            
#         # 4. 初始化本回合状态
#         self.state["goal_x"], self.state["goal_y"] = task["g_x"], task["g_y"]
#         self.state["step"] = 0
#         self.state["last_action"] = np.zeros(2, dtype=np.float32)
        
#         odom = self._get_odom_data()
#         self.state["prev_dist"] = self._goal_dist(odom) if odom else 1.0
        
#         return self._get_obs(), {
#             "spawn_pos": (task["s_x"], task["s_y"]),
#             "goal_pos": (task["g_x"], task["g_y"]),
#             "initial_dist": math.hypot(task["s_x"] - task["g_x"], task["s_y"] - task["g_y"])
#         }


#     # ==================== Step 辅助函数 ====================
#     def _get_step_data(self):
#         return self._get_obs(), self._get_odom_data(), self._get_imu_data()

#     def _check_termination(self, curr_dist: float) -> Tuple[bool, bool]:
#         goal_reached = curr_dist < self.world["dist_to_goal_threshold"]
#         collision = self._check_collision()
#         return goal_reached, collision

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         action = np.clip(action, self.action_space.low, self.action_space.high)

#         # 1. 执行动作 & 等待物理演化
#         self.send_velocity(float(action[0]), float(action[1]))
#         time.sleep(self.world["step_duration"])

#         # 2. 获取最新数据
#         obs, odom, imu = self._get_step_data()
#         if odom is None: return obs, -1.0, False, True, {"error": "odom unavailable"}

#         # 3. 判定终止条件
#         curr_dist = self._goal_dist(odom)
#         goal_reached, collision = self._check_termination(curr_dist)

#         # 4. 计算奖励
#         # ==========================================
#         # 🔥 组装严格契约 (Context)
#         # ==========================================
#         ctx = RewardContext(
#             goal_reached=goal_reached,
#             collision=collision,
#             curr_dist=curr_dist,
#             prev_dist=self.state["prev_dist"],
#             goal_x=self.state["goal_x"],
#             goal_y=self.state["goal_y"],
#             action=action,
#             last_action=self.state["last_action"], # 取上一步的动作
#             odom=odom,
#             imu=imu,
#             min_laser=self._min_laser() # 提前算好最小距离
#         )
#         # ==========================================
#         # 🔥 计算奖励 (告别内部臃肿的 if-else)
#         # ==========================================
#         reward = self.reward_fn.compute(ctx)

#         # 5. 更新内部状态
#         self.state["prev_dist"] = curr_dist
#         self.state["last_action"] = action.copy()
#         self.state["step"] += 1

#         terminated = goal_reached or collision
#         truncated = self.state["step"] >= self.world["episode_steps_max"]

#         if terminated or truncated:
#             self.send_velocity(0.0, 0.0)

#         return obs, float(reward), terminated, truncated, {
#             "collision": collision, "goal_reached": goal_reached,
#             "final_dist": curr_dist, "episode_step": self.state["step"]
#         }


#     def run_episode(self, model, start_pos=None, goal_pos=None, deterministic=True, safe_threshold=None, skip_spawn_check=False, verbose=False):
#         options = {}
#         if start_pos is not None: options["start_pos"] = start_pos
#         if goal_pos is not None: options["goal_pos"] = goal_pos
#         if safe_threshold is not None: options["safe_threshold"] = safe_threshold
#         if skip_spawn_check: options["skip_spawn_check"] = True
            
#         obs, info = self.reset(options=options if options else None)
        
#         if verbose:
#             print(f"\n[Ep Start] Spawn:{info['spawn_pos']} | Goal:{info['goal_pos']} | Obs Shape:{obs.shape}")

#         ep_reward = 0.0
#         step = 0
#         done = False
        
#         while not done:
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, reward, terminated, truncated, info = self.step(action)
#             ep_reward += reward
            
#             if verbose:
#                 action_np = np.array(action).flatten()
#                 print(f"  Step {step}: Act=[{action_np[0]:.3f}, {action_np[1]:.3f}], reward={ep_reward}")
                
#             step += 1
#             done = terminated or truncated
            
#         return {
#             "reward": ep_reward,
#             "success": info.get('goal_reached', False),
#             "steps": step,
#             "final_dist": info.get('final_dist', -1.0)
#         }

#     def run_episodes(self, model, num_episodes=10, start_pos_list=None, goal_pos_list=None, deterministic=True, safe_threshold=None, verbose=False):
#         results = []
        
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
            
#             skip_check = (s_pos is not None)
            
#             res = self.run_episode(
#                 model=model, start_pos=s_pos, goal_pos=g_pos,
#                 deterministic=deterministic, safe_threshold=safe_threshold,
#                 skip_spawn_check=skip_check, verbose=verbose
#             )
#             results.append(res)
            
#             # verbose 时让 run_episode 自己打印细节，这里只打印结局
#             if verbose:
#                 status = "✅" if res["success"] else "❌"
#                 print(f"  -> Result: {status} | Final Reward: {res['reward']:.2f} | Steps: {res['steps']}")
#             else:
#                 status = "✅" if res["success"] else "❌"
#                 print(f"\r[Episode {i+1}/{num_episodes}] {status} Reward: {res['reward']:.2f} | Steps: {res['steps']}", end="", flush=True)
                
#         if not verbose: print()

#         total_success = sum(1 for r in results if r["success"])
#         return {
#             "num_episodes": num_episodes,
#             "success_rate": (total_success / num_episodes) * 100 if num_episodes > 0 else 0,
#             "mean_reward": np.mean([r["reward"] for r in results]) if results else -1,
#             "mean_steps": np.mean([r["steps"] for r in results]) if results else -1,
#             "details": results
#         }


#     # def close(self):
#     #     self.send_velocity(0.0, 0.0)
#     #     if rclpy.ok():
#     #         self.destroy_node()

#     def close(self):
#         self.send_velocity(0.0, 0.0)


#     # ==================== 内部逻辑 ====================
#     def _sample_safe_position(self, allowed_zones: list = None) -> Tuple[float, float]:
#         """
#         从嵌套的安全区字典中随机采样坐标
#         Args:
#             allowed_zones: 允许的大区名称列表(如 ['zone_arena_robottom'])，None 表示全局随机
#         """
#         flat_pool = []
#         if allowed_zones is None:
#             for sub_zones in self.zones.values():
#                 flat_pool.extend(sub_zones)
#         else:
#             for zone_name in allowed_zones:
#                 if zone_name in self.zones:
#                     flat_pool.extend(self.zones[zone_name])

#         if not flat_pool:
#             raise ValueError("No valid safe zones found for sampling!")

#         zone = flat_pool[self.np_random.integers(len(flat_pool))]
#         angle = self.np_random.uniform(0, 2 * math.pi)
#         r = self.np_random.uniform(0, zone["r"])
#         pos = (zone["cx"] + r * math.cos(angle), zone["cy"] + r * math.sin(angle))
#         return pos

#     @staticmethod
#     def _euclidean(a, b):
#         return math.hypot(a[0] - b[0], a[1] - b[1])

#     def _goal_dist(self, odom: Dict) -> float:
#         return self._euclidean(
#             (self.state["goal_x"], self.state["goal_y"]), 
#             (odom["x"], odom["y"])
#         )
        
#     def _check_collision(self) -> bool:
#         # 🔥 重构简化：去掉了 self._lock 和 self._latest_scan_time 的判断
#         # 原理：如果雷达断连，_get_laser_data() 会返回全 1.0 数组，
#         # min_dist 会等于 laser_range_max，必然大于碰撞阈值，天然防误判。
#         laser_norm = self._get_laser_data()
#         min_dist = float(np.min(laser_norm)) * self.robot["laser_range_max"]
        
#         if min_dist < self.robot["proximity_to_collision_threshold"]:
#             self.bridge.node.get_logger().debug(f"Collision! Min dist: {min_dist:.3f}m")
#             return True
#         return False

#     def _check_spawn_pos(self, timeout=3.0, threshold=None):
#         if threshold is None:
#             threshold = self.robot["proximity_to_be_safe_min"] * 0.9
            
#         timeout_deadline = time.time() + timeout
#         while time.time() < timeout_deadline:
#             laser_norm = self._get_laser_data()
#             min_dist = float(np.min(laser_norm)) * self.robot["laser_range_max"]
#             if min_dist > threshold:
#                 return True
#             time.sleep(0.1)
            
#         self.bridge.node.get_logger().warn(f"Spawn check failed. Min dist: {min_dist:.3f}m < Threshold: {threshold:.3f}m")
#         return False

#     def _check_goal_pos(self, goal_x: float, goal_y: float) -> bool:
#         # 🔥 适配嵌套字典结构：遍历所有大区 -> 遍历大区内的子点
#         for sub_zones in self.zones.values():
#             for zone in sub_zones:
#                 dist_to_center = self._euclidean((goal_x, goal_y), (zone["cx"], zone["cy"]))
#                 if dist_to_center <= zone["r"]:
#                     return True
#         self.bridge.node.get_logger().warn(f"Goal ({goal_x:.2f}, {goal_y:.2f}) is outside all safe zones!")
#         return False

#     def _min_laser(self) -> float:
#         # 🔥 重构：直接通过 Bridge 获取原始 msg，内部自带线程安全
#         msg = self.bridge.get_data('/scan')
#         if msg is None: 
#             return self.robot["laser_range_max"]
#         ranges = np.array(msg.ranges)
#         ranges = np.nan_to_num(ranges, nan=self.robot["laser_range_max"], posinf=self.robot["laser_range_max"])
#         return float(np.min(ranges))

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
#             imu_acc = np.clip(np.array(imu['acc']) / self.robot["lin_acc_physics_max"], -1.0, 1.0)
#             imu_gyro = np.clip(np.array(imu['gyro']) / self.robot["ang_vel_imu_physics_max"], -1.0, 1.0)
#             imu_rp = np.clip(np.array(imu['rpy'][:2]) / (math.pi/4), -1.0, 1.0)

#         if odom is not None:
#             dx = self.state["goal_x"] - odom['x']
#             dy = self.state["goal_y"] - odom['y']
#             yaw = odom['yaw']
#             local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
#             local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
#             goal_angle = math.atan2(local_y, local_x) / math.pi
#             goal_dist = min(math.hypot(dx, dy) / self.world["dist_to_goal_clip_norm"], 1.0)
#             odom_vel = np.clip(
#                 np.array([odom['vx'], odom['wz']]) / np.array([self.robot["lin_vel_max"], self.robot["ang_vel_max"]]), 
#                 -1.0, 1.0
#             )

#         last_act_norm = np.clip(
#             np.array([self.state["last_action"][0], self.state["last_action"][1]]) / 
#             np.array([self.robot["lin_vel_max"], self.robot["ang_vel_max"]]), 
#             -1.0, 1.0
#         )
#         obs = np.concatenate([
#             laser_norm, imu_acc, imu_gyro, imu_rp,
#             [goal_angle], [goal_dist], odom_vel, last_act_norm
#         ]).astype(np.float32)
#         return obs
