from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypedDict


# ==========================================
# 1. 严格的数据契约 (防止传参手滑)
# ==========================================
class RewardContext(TypedDict):
    """Env 在调用 compute 时，必须严格按这个结构塞数据"""
    # 终局标志
    goal_reached: bool
    collision: bool
    
    # 距离与目标 (将 Env 的内部状态剥离出来传入)
    curr_dist: float
    prev_dist: float
    goal_x: float
    goal_y: float
    
    # 动作 (将历史动作传入，避免 Reward 类去读 Env.state)
    action: np.ndarray
    last_action: np.ndarray
    
    # 传感器字典 (None 表示数据缺失/断连)
    odom: Optional[Dict[str, Any]]
    imu: Optional[Dict[str, Any]]
    min_laser: float  


# ==========================================
# 2. 顶层的极简基类 (跨模态通用)
# ==========================================
class BaseReward(ABC):
    @abstractmethod
    def compute(self, ctx: RewardContext) -> float:
        pass


# ==========================================
# 3. TB3 公用惩罚池
# ==========================================
class Ttb3CommonPenaltiesMixin:
    """
    纯工具类，只封装所有地图通用的【惩罚】逻辑。
    """
    # 为了让 IDE 不报错，做个类型提示声明
    def __init__(self, config: Dict):
        """🔥 统一接收完整 config，内部自行拆包"""
        self.config = config
        self.bot = config["bot"]
        self.world = config["world"]
        self.rew = config["reward"]
        
    def _pen_safety(self, ctx: RewardContext) -> float:
        """靠近障碍物惩罚"""
        min_dist = ctx["min_laser"]
        safe_min = self.bot["proximity_to_be_safe_min"]
        if min_dist < safe_min:
            return self.rew["penalty_in_safe_proximity"] * ((safe_min - min_dist) / safe_min)**2
        return 0.0

    def _pen_stability(self, ctx: RewardContext) -> float:
        """IMU 姿态不稳定惩罚"""
        imu = ctx["imu"]
        if imu is None: return 0.0
        roll, pitch = imu['rpy']
        tilt_factor = (roll**2 + pitch**2) / ((math.pi/6)**2)
        return self.rew["penalty_instability"] * min(tilt_factor, 1.0)

    def _pen_stuck(self, ctx: RewardContext) -> float:
        """指令与实际速度不符（打滑/卡住）惩罚"""
        action = ctx["action"]
        odom = ctx["odom"]
        if odom is None: return 0.0
        vel_error = abs(action[0]) - abs(odom['vx'])
        if vel_error > self.bot["lin_vel_stuck_threshold"]: 
             return self.rew["penalty_stuck"] * (vel_error / self.bot["lin_vel_max"])
        return 0.0

    def _pen_smoothness(self, ctx: RewardContext) -> float:
        """动作突变惩罚（防抽搐）"""
        action = ctx["action"]
        last_action = ctx["last_action"]
        diff_lin = (action[0] - last_action[0]) / self.bot["lin_vel_max"]
        diff_ang = (action[1] - last_action[1]) / self.bot["ang_vel_max"]
        smoothness_penalty = np.sqrt(diff_lin**2 + diff_ang**2)
        return self.rew["penalty_action_smoothness"] * smoothness_penalty


# ==========================================
# 4. TurtleBot3 House 环境组装车间
# ==========================================
class Ttb3HouseReward(BaseReward, Ttb3CommonPenaltiesMixin):
    def __init__(self, config: Dict):
        super().__init__(config)

    def compute(self, ctx: RewardContext) -> float:
        if ctx["goal_reached"]: return self.rew["reward_at_goal"]
        if ctx["collision"]: return self.rew["penalty_at_collision"]

        reward = 0.0
        # --- 🟩 正向奖励 ---
        reward += self._rwd_distance(ctx)            
        reward += self._rwd_heading(ctx)             
        
        # --- 🟥 负向惩罚 (继承自 Mixin) ---
        reward += self._pen_safety(ctx)              
        reward += self._pen_stability(ctx)           
        reward += self._pen_stuck(ctx)               
        reward += self._pen_smoothness(ctx)          
        
        # --- ⚪ 固定消耗 ---
        reward += self.rew["penalty_elapsing_time"]  
        return float(reward)

    def _rwd_distance(self, ctx: RewardContext) -> float:
        """House 特有：过滤极微小噪点"""
        dist_delta = ctx["prev_dist"] - ctx["curr_dist"]  
        if abs(dist_delta) < 0.01: 
            return 0.0
        return dist_delta * self.rew["reward_factor_approaching_goal"]
    
    def _rwd_heading(self, ctx: RewardContext) -> float:
        """House 特有：硬截断严格，防原地骗分"""
        odom = ctx["odom"]
        if odom is None: return 0.0
        target_angle = math.atan2(ctx["goal_y"] - odom['y'], ctx["goal_x"] - odom['x'])
        yaw_error = math.atan2(math.sin(target_angle - odom['yaw']), math.cos(target_angle - odom['yaw']))
        heading_factor = math.cos(yaw_error)  
        
        if abs(odom['vx']) > 0.05:  
            return heading_factor * self.rew["reward_good_orientation"]
        return 0.0


# ==========================================
# 5. TurtleBot3 World 环境组装车间
# ==========================================
class Ttb3WorldReward(BaseReward, Ttb3CommonPenaltiesMixin):
    def __init__(self, config: Dict):
        super().__init__(config)

    def compute(self, ctx: RewardContext) -> float:
        if ctx["goal_reached"]: return self.rew["reward_at_goal"]
        if ctx["collision"]: return self.rew["penalty_at_collision"]

        reward = 0.0
        # --- 🟩 正向奖励 ---
        reward += self._rwd_distance(ctx)            
        reward += self._rwd_heading(ctx)             
        
        # --- 🟥 负向惩罚 (继承自 Mixin) ---
        reward += self._pen_safety(ctx)              
        reward += self._pen_stability(ctx)           
        reward += self._pen_stuck(ctx)               
        reward += self._pen_smoothness(ctx)          
        
        # --- ⚪ 固定消耗 ---
        reward += self.rew["penalty_elapsing_time"]  
        return float(reward)

    def _rwd_distance(self, ctx: RewardContext) -> float:
        """World 特有：不过滤微小移动，鼓励空旷环境下的持续探索"""
        dist_delta = ctx["prev_dist"] - ctx["curr_dist"]  
        return dist_delta * self.rew["reward_factor_approaching_goal"]

    def _rwd_heading(self, ctx: RewardContext) -> float:
        """World 特有：相对宽松，微动也给分"""
        odom = ctx["odom"]
        if odom is None: return 0.0
        target_angle = math.atan2(ctx["goal_y"] - odom['y'], ctx["goal_x"] - odom['x'])
        yaw_error = math.atan2(math.sin(target_angle - odom['yaw']), math.cos(target_angle - odom['yaw']))
        heading_factor = math.cos(yaw_error)  
        
        if abs(odom['vx']) > 0.01:
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
    根据 YAML 里的 name，实例化对应的 Reward 类
    """
    if name not in _REWARD_REGISTRY:
        raise ValueError(f"未知的奖励模型 '{name}'，当前可选: {list(_REWARD_REGISTRY.keys())}")
    
    reward_class = _REWARD_REGISTRY[name]
    # 🔥 完美闭环：工厂将完整的 config 扔给具体类
    return reward_class(config)
