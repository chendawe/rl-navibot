"""
Gazebo 仿真环境控制器。

职责：
- 世界重置
- 机器人物理状态设置（位置、朝向）
- 模型实体管理（生成、删除）

设计原则：
- 只依赖 Gazebo 特有的 ROS2 Service，与具体机器人型号无关。
- 换成 Isaac Sim / PyBullet 时，只需写一个对应的 Simulator 类即可。
"""

import math
import time
import logging
from typing import Optional

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity

from core.ros2.channels.bridges.base import BaseBridge

logger = logging.getLogger(__name__)


class GazeboSimulator(BaseBridge):
    """
    Gazebo 仿真环境控制器。

    持有 Gazebo 服务的长期客户端，避免高频 reset 时反复创建/销毁。
    """

    def __init__(self, runtime, node_name: str = "gazebo_simulator"):
        super().__init__(node_name=node_name, runtime=runtime)
        self._sim_clients = {}
        self._entity_name = ""
        # 🔥 新增：双模态配置
        self._reset_mode = "teleport"  # 默认高性能瞬移模式
        self._robot_urdf = None        # 缓存 URDF 字符串


    # ================================================================
    #  一、模板方法实现
    # ================================================================

    def setup(self, *,
              reset_world: str, set_entity: str,
              delete_entity: str, spawn_entity: str,
              entity_name: str,
              reset_mode: str = "teleport",
              robot_urdf: str = None,
              **kwargs):
        """
            注册所有 Gazebo 仿真服务客户端
            新增 reset_mode 参数：
            - "teleport": 极速瞬移 (默认，适合大规模 RL 训练)
            - "spawn": 删除旧模型并重新生成 URDF (慢，但绝对干净，适合 Debug)
        """
        self._reset_mode = reset_mode
        self._robot_urdf = robot_urdf
        
        if self._reset_mode == "spawn" and not self._robot_urdf:
            raise ValueError("使用 spawn 重置模式时，必须提供 robot_urdf!")
        
        self._entity_name = entity_name
        self._sim_clients = {
            "reset_world": self.create_client(Empty, reset_world),
            "set_entity":  self.create_client(SetEntityState, set_entity),
            "delete":      self.create_client(DeleteEntity, delete_entity),
            "spawn":       self.create_client(SpawnEntity, spawn_entity),
        }
        logger.info(f"[GazeboSimulator] 服务客户端已注册，实体名: {entity_name}")

    # ... 在 GazeboSimulator 类中添加 ...

    def update_urdf(self, urdf: str):
        """
        接收来自上层的 URDF 更新指令。
        为什么不直接让 Env 去 sim._robot_urdf = xxx？
        因为面向对象设计的封装性：Simulator 的内部缓存应该由自己的方法来修改，
        未来如果这里需要加版本号校验、XML 格式预检，都在这里加，Env 层无需改动。
        """
        if not isinstance(urdf, str):
            raise TypeError("更新 URDF 必须传入合法的字符串!")
        self._robot_urdf = urdf

    # ================================================================
    #  二、服务可用性检测
    # ================================================================

    def wait_for_services(self, timeout_sec: float = 10.0, retries: int = 3):
        """
        阻塞等待所有 Gazebo 仿真服务上线。
        支持 retries：DDS 发现偶尔第一轮会丢包，重试几次比暴力加超时更靠谱。
        """
        import time
        
        for attempt in range(1, retries + 1):
            deadline = time.time() + timeout_sec
            pending = set(self._sim_clients.keys())
            
            while pending and time.time() < deadline:
                still_missing = set()
                for name, client in self._sim_clients.items():
                    if name in pending and not client.service_is_ready():
                        still_missing.add(name)
                pending = still_missing
                if pending:
                    time.sleep(0.5)
            
            if not pending:
                logger.info(f"[Sim] 所有仿真服务已就绪 (尝试 {attempt}/{retries})")
                return
            
            logger.warning(f"[Sim] 第 {attempt} 轮仍有服务未就绪: {pending}，重试中...")
            time.sleep(1.0)
        
        raise TimeoutError(
            f"仿真服务 {pending} 在 {timeout_sec}s × {retries} 次内未上线！"
            f"请确认 Gazebo 已启动且 gazebo_ros 包已正确加载。"
        )
    
    # ================================================================
    #  三、世界级控制
    # ================================================================

    def reset_world(self, timeout_sec: float = 5.0):
        """清空整个物理世界（物体归位、速度归零）"""
        future = self._sim_clients["reset_world"].call_async(Empty.Request())
        start = time.time()
        while not future.done() and time.time() - start < timeout_sec:
            time.sleep(0.05)
        if not future.done():
            raise TimeoutError(f"reset_world 超时 ({timeout_sec}s)!")
        if future.exception():
            raise RuntimeError(f"reset_world 失败: {future.exception()}")

    # ================================================================
    #  四、实体级控制
    # ================================================================

    def set_robot_pose(self, x: float, y: float, yaw: float = 0.0,
                       frame: str = "world"):
        """将机器人瞬移到指定位姿（不经过物理引擎，直接设状态）"""
        req = SetEntityState.Request()
        req.state.name = self._entity_name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = frame

        future = self._sim_clients["set_entity"].call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < 3.0:
            time.sleep(0.05)
        if not future.done():
            raise TimeoutError("set_robot_pose 超时 (3s)!")
        if future.exception():
            raise RuntimeError(f"set_robot_pose 失败: {future.exception()}")

    def delete_robot(self, timeout_sec: float = 3.0):
        """从物理世界中删除机器人实体"""
        req = DeleteEntity.Request()
        req.name = self._entity_name
        future = self._sim_clients["delete"].call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < timeout_sec:
            time.sleep(0.05)
        if not future.done():
            raise TimeoutError("delete_robot 超时!")
        if future.exception():
            raise RuntimeError(f"delete_robot 失败: {future.exception()}")

    def spawn_robot(self, xml: str, timeout_sec: float = 5.0):
        """向物理世界中生成机器人实体"""
        req = SpawnEntity.Request()
        req.name = self._entity_name
        req.xml = xml
        future = self._sim_clients["spawn"].call_async(req)
        start = time.time()
        while not future.done() and time.time() - start < timeout_sec:
            time.sleep(0.05)
        if not future.done():
            raise TimeoutError("spawn_robot 超时!")
        if future.exception():
            raise RuntimeError(f"spawn_robot 失败: {future.exception()}")
