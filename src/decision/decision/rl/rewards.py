from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypedDict


# ==========================================
# 1. 严格的数据契约
# ==========================================
class RewardContext(TypedDict):
    """
    Env 在调用 compute 时，必须严格按这个结构塞数据。
    使用 TypedDict 的好处：如果在 Env 里拼写错了键名（比如把 goal_x 写成 goal_xx），
    IDE 会直接报错，而不是像普通 dict 一样在运行时默默返回 None 导致隐式 Bug。
    """
    # 终局标志
    goal_reached: bool
    collision: bool
    
    # 距离与目标 (将 Env 的内部状态剥离出来传入，保持 Reward 类的无状态性)
    curr_dist: float
    prev_dist: float
    goal_x: float
    goal_y: float
    
    # 动作 (将历史动作传入，避免 Reward 类去反向读取 Env.state，解耦彻底)
    action: np.ndarray      # 当前 step 的动作 [lin, ang]
    last_action: np.ndarray # 上一步的动作 [lin, ang]
    
    # 传感器字典 (None 表示数据缺失/断连，Reward 内部需做防御性判断)
    odom: Optional[Dict[str, Any]]
    imu: Optional[Dict[str, Any]]
    min_laser: float  


# ==========================================
# 2. 顶层的极简基类
# ==========================================
class BaseReward(ABC):
    """所有地图奖励的抽象基类，保证对外接口统一"""
    @abstractmethod
    def compute(self, ctx: RewardContext) -> float:
        pass


# ==========================================
# 3. TB3 公用惩罚池
# ==========================================
class Ttb3CommonPenaltiesMixin:
    """
    纯工具类 Mixin，只封装所有地图通用的【惩罚】逻辑。
    为什么用 Mixin 而不是基类？
    因为 House 和 World 的正向奖励逻辑不同，但负向惩罚是一样的。
    Mixin 允许它们像拼乐高一样，共享这组惩罚代码，同时继承不同的 BaseReward。
    """
    def __init__(self, config: Dict):
        """统一接收完整 config，内部自行拆包，避免子类重复写赋值逻辑"""
        self.config = config
        self.robot = config["robot"]
        self.world = config["world"]
        self.rew = config["reward"]
    
    # 老burger的sac按这个算的
    # def _pen_safety(self, ctx: RewardContext) -> float:
    #     """
    #     靠近障碍物惩罚 (二次方惩罚)
    #     物理直觉：越近越疼。使用 ((safe - curr) / safe)^2 实现非线性的陡增惩罚曲线。
    #     只有当距离小于安全线时才触发，避免全局打压探索欲。
    #     """
    #     min_dist = ctx["min_laser"]
    #     safe_min = self.robot["proximity_to_be_safe_min"]
    #     if min_dist < safe_min:
    #         return self.rew["penalty_factor_in_safe_proximity"] * ((safe_min - min_dist) / safe_min)**2
    #     return 0.0
    
    def _pen_safety(self, ctx: RewardContext) -> float:
        """
        自适应 ABS 指数惩罚 (自动计算 k，保证刹车点在安全区正中间)
        """
        min_dist = ctx["min_laser"]
        safe_min = self.robot["proximity_to_be_safe_min"]
        collision_min = self.robot["proximity_to_collision_threshold"]
        
        # 读取 YAML 里的刹车系数
        penalty_factor = abs(self.rew["penalty_factor_in_safe_proximity"]) 
        
        if min_dist < safe_min:
            depth = (safe_min - min_dist) / (safe_min - collision_min)
            depth = max(0.0, min(1.0, depth))
            
            # 1. 真正动态计算单步最大贪欲收益
            # max_step_dist = 最大速度(m/s) * 控制步长
            # 假设你在 robot 字典里有 max_linear_speed，没有的话换成你实际的配置键名
            max_speed = self.robot.get("max_linear_speed", 0.22) 
            control_step = self.robot.get("control_step", 0.1)  # 10Hz 就是 0.1s
            max_step_dist = max_speed * control_step
            
            max_greed_reward = self.rew["reward_factor_approaching_goal"] * max_step_dist
            
            # 2. 根据目标刹车点，反推指数 k (修正了之前的数学近似)
            if penalty_factor > max_greed_reward:
                ratio = max_greed_reward / penalty_factor
                target_depth = 0.5  # 在正中间刹车
                
                # 数学解析解：k = ln(ratio) / (target_depth - 1)
                # 当 ratio 很小时，这个公式极其精准
                k = math.log(ratio) / (target_depth - 1.0)
                k = max(0.5, min(k, 20.0)) # 防止除零或奇点
            else:
                k = 50.0 # 刹车片太薄，直接变成阶梯函数
                
            # 3. 执行指数惩罚
            exp_curve = (math.exp(k * depth) - 1.0) / (math.exp(k) - 1.0)
            return -penalty_factor * exp_curve
            
        return 0.0


    def _pen_stability(self, ctx: RewardContext) -> float:
        """
        IMU 姿态不稳定惩罚
        物理直觉：差速驱动的 TB3 在急转弯时容易侧翻或打滑。
        通过 Roll 和 Pitch 的平方和惩罚倾斜程度，超过 30度(pi/6)封顶。
        """
        imu = ctx["imu"]
        if imu is None: return 0.0
        roll, pitch = imu['rpy']
        tilt_factor = (roll**2 + pitch**2) / ((math.pi/6)**2)
        return self.rew["penalty_instability"] * min(tilt_factor, 1.0)

    def _pen_stuck(self, ctx: RewardContext) -> float:
        """
        指令与实际速度不符惩罚（打滑/卡住惩罚）
        物理直觉：我让你往前走(action[0])，你却没走(odom['vx'])，说明你被卡住了。
        这比单纯判断 "速度为0" 更聪明，因为它能抓到"轮子打滑空转"的死局。
        """
        action = ctx["action"]
        odom = ctx["odom"]
        if odom is None: return 0.0
        vel_error = abs(action[0]) - abs(odom['vx'])
        if vel_error > self.robot["lin_vel_stuck_threshold"]: 
             return self.rew["penalty_stuck"] * (vel_error / self.robot["lin_vel_max"])
        return 0.0

    def _pen_smoothness(self, ctx: RewardContext) -> float:
        """
        动作突变惩罚（Jerk 加加速度惩罚，防抽搐）
        物理直觉：真实的电机有惯性，0.22突然变成-0.22会产生极大的物理冲击。
        计算动作空间中的欧氏距离，让网络学会"温柔地转弯"。
        """
        action = ctx["action"]
        last_action = ctx["last_action"]
        diff_lin = (action[0] - last_action[0]) / self.robot["lin_vel_max"]
        diff_ang = (action[1] - last_action[1]) / self.robot["ang_vel_max"]
        smoothness_penalty = np.sqrt(diff_lin**2 + diff_ang**2)
        return self.rew["penalty_action_smoothness"] * smoothness_penalty


# ==========================================
# 4. Turtlerobot3 House 环境组装车间
# ==========================================
class Ttb3HouseReward(BaseReward, Ttb3CommonPenaltiesMixin):
    """
    House（房屋）特化奖励。
    特点：室内环境狭窄，激光噪点对距离奖励干扰大，要求动作精准。
    """
    def __init__(self, config: Dict):
        super().__init__(config)

    def compute(self, ctx: RewardContext) -> float:
        if ctx["collision"]: return self.rew["penalty_at_collision"]
        if ctx["goal_reached"]: return self.rew["reward_at_goal"]

        reward = 0.0
        # --- 🟩 正向奖励 ---
        reward += self._rwd_distance(ctx)            
        reward += self._rwd_heading(ctx)             
        
        # --- 🟥 负向惩罚 (继承自 Mixin) ---
        reward += self._pen_safety(ctx)              
        reward += self._pen_stability(ctx)           
        reward += self._pen_stuck(ctx)               
        reward += self._pen_smoothness(ctx)          
        
        # --- ⚪ 固定消耗 (逼迫网络走最短路径) ---
        reward += self.rew["penalty_elapsing_time"]  
        return float(reward)

    def _rwd_distance(self, ctx: RewardContext) -> float:
        """
        House 特有：死区过滤奖励。
        为什么？室内走廊里激光会因为墙面反光出现 ±0.005m 的跳动。
        如果不设死区，网络即使站着不动，也能通过这些噪点"骗"到微小的正向奖励。
        """
        dist_delta = ctx["prev_dist"] - ctx["curr_dist"]  
        if abs(dist_delta) < 0.01:  # 死区：小于 1cm 的移动视为噪点，不给分
            return 0.0
        return dist_delta * self.rew["reward_factor_approaching_goal"]
    
    def _rwd_heading(self, ctx: RewardContext) -> float:
        """
        House 特有：硬截断朝向奖励，防原地骗分。
        室内空间小，网络喜欢原地转圈对准目标来白嫖朝向分。
        要求：不仅朝向要对（cos夹角），而且必须有实质的前进速度（vx > 0.05）才给分。
        """
        odom = ctx["odom"]
        if odom is None: return 0.0
        
        target_angle = math.atan2(ctx["goal_y"] - odom['y'], ctx["goal_x"] - odom['x'])
        yaw_error = math.atan2(math.sin(target_angle - odom['yaw']), math.cos(target_angle - odom['yaw']))
        heading_factor = math.cos(yaw_error)  # 1.0 表示完全正对，0.0 表示垂直，-1.0 表示背对
        
        if abs(odom['vx']) > 0.05:  # 硬门槛：速度必须大于 5cm/s
            return heading_factor * self.rew["reward_good_orientation"]
        return 0.0


# ==========================================
# 5. Turtlerobot3 World 环境组装车间
# ==========================================
class Ttb3WorldReward(BaseReward, Ttb3CommonPenaltiesMixin):
    """
    World（空旷世界）特化奖励。
    特点：空间大，不需要死区防噪点；鼓励大角度灵活转向。
    """
    def __init__(self, config: Dict):
        super().__init__(config)

    def compute(self, ctx: RewardContext) -> float:
        if ctx["collision"]: return self.rew["penalty_at_collision"]
        if ctx["goal_reached"]: return self.rew["reward_at_goal"]

        reward = 0.0
        reward += self._rwd_distance(ctx)            
        reward += self._rwd_heading(ctx)             
        
        reward += self._pen_safety(ctx)              
        reward += self._pen_stability(ctx)           
        reward += self._pen_stuck(ctx)               
        reward += self._pen_smoothness(ctx)          
        
        reward += self.rew["penalty_elapsing_time"]  
        return float(reward)

    def _rwd_distance(self, ctx: RewardContext) -> float:
        """
        World 特有：无死区奖励。
        空旷环境下哪怕移动 1mm 也是真实的探索，不用过滤，直接按比例给分。
        """
        dist_delta = ctx["prev_dist"] - ctx["curr_dist"]  
        return dist_delta * self.rew["reward_factor_approaching_goal"]

    def _rwd_heading(self, ctx: RewardContext) -> float:
        """
        World 特有：宽松朝向奖励。
        空旷环境下，微小的调整也应受到鼓励，因此速度门槛放低到 1cm/s。
        """
        odom = ctx["odom"]
        if odom is None: return 0.0
        target_angle = math.atan2(ctx["goal_y"] - odom['y'], ctx["goal_x"] - odom['x'])
        yaw_error = math.atan2(math.sin(target_angle - odom['yaw']), math.cos(target_angle - odom['yaw']))
        heading_factor = math.cos(yaw_error)  
        
        if abs(odom['vx']) > 0.01:  # 宽松门槛：1cm/s 即可
            return heading_factor * self.rew["reward_good_orientation"]
        return 0.0


# ==========================================
# 6. 奖励工厂 (必须放在文件最底部！)
# ==========================================
_REWARD_REGISTRY = {
    "ttb3_house": Ttb3HouseReward,
    "ttb3_world": Ttb3WorldReward,
}

def get_reward_model(name: str, config: dict) -> BaseReward:
    """
    根据 YAML 里的 name，实例化对应的 Reward 类。
    符合开闭原则：以后加新地图，只需写新类并注册到这里，不用改 Env 代码。
    """
    if name not in _REWARD_REGISTRY:
        raise ValueError(f"未知的奖励模型 '{name}'，当前可选: {list(_REWARD_REGISTRY.keys())}")
    
    reward_class = _REWARD_REGISTRY[name]
    return reward_class(config)
