# core/ros2/channels/injectors/map_dynamic_grid_wall.py

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import OccupancyGrid
import logging

logger = logging.getLogger(__name__)


class MapDynamicGridWallInjector(Node):
    """
    通过伪造 OccupancyGrid 实现动态虚拟墙。
    适用场景：不依赖 Nav2 特定插件，或者需要“跟随机器人移动的限定圈”。
    """
    def __init__(self, runtime):
        super().__init__('dynamic_grid_wall_injector')
        runtime.register_node(self)

        # 声明参数
        self.declare_parameter('radius', 5.0)         # 限制半径(米)
        self.declare_parameter('thickness', 1.2)      # 墙厚(米)
        self.declare_parameter('resolution', 0.05)    # 地图分辨率(米/像素)
        self.declare_parameter('update_freq', 2.0)    # 更新频率
        
        self.radius = self.get_parameter('radius').value
        self.thickness = self.get_parameter('thickness').value
        self.resolution = self.get_parameter('resolution').value
        update_freq = self.get_parameter('update_freq').value

        # 计算需要的正方形地图边长(像素)
        self.map_size_px = int((self.radius * 2 + self.thickness * 2) / self.resolution)
        self.thickness_px = int(self.thickness / self.resolution)
        self.radius_px = int(self.radius / self.resolution)
        self.center_px = self.map_size_px // 2

        # ⚠️ 致命坑：必须使用 TRANSIENT_LOCAL，否则 Nav2 的 StaticLayer 会直接无视这条消息
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        self.pub = self.create_publisher(OccupancyGrid, '/dynamic_wall_map', qos)

        # 定时器：不断根据机器人当前位置重新画圈
        self._timer_cg = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(1.0 / update_freq, self._update_wall, callback_group=self._timer_cg)
        
        logger.info(f"🧱 [DynamicGridWall] 启动，半径:{self.radius}m, 厚度:{self.thickness}m")

    def _update_wall(self):
        """每次触发时，重新生成一堵圆形的墙"""
        # 实际应用中，这里应该监听 TF 获取机器人真实坐标
        # 为了演示，我们假设机器人始终在 (0,0)
        robot_x, robot_y = 0.0, 0.0 

        # 计算地图原点（左下角坐标）
        origin_x = robot_x - (self.map_size_px * self.resolution) / 2
        origin_y = robot_y - (self.map_size_px * self.resolution) / 2

        # 用 numpy 快速生成二维网格坐标
        y_indices, x_indices = np.ogrid[:self.map_size_px, :self.map_size_px]
        
        # 计算每个像素到中心的距离
        dist = np.sqrt((x_indices - self.center_px)**2 + (y_indices - self.center_px)**2)

        # 初始化全部为未知 (-1)
        grid_data = np.full((self.map_size_px, self.map_size_px), -1, dtype=np.int8)

        # 1. 内部圆填 0 (空白)
        inner_mask = dist <= self.radius_px
        grid_data[inner_mask] = 0

        # 2. 边界环填 100 (障碍)
        outer_radius_px = self.radius_px + self.thickness_px
        wall_mask = (dist > self.radius_px) & (dist <= outer_radius_px)
        grid_data[wall_mask] = 100

        # 组装 ROS 消息
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = self.map_size_px
        msg.info.height = self.map_size_px
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.orientation.w = 1.0
        
        # 将二维数组拉平为一维列表存入 data
        msg.data = grid_data.flatten().tolist()

        self.pub.publish(msg)


# global_costmap:
#   global_costmap:
#     ros__parameters:
#       # 插件列表：原有的 static_layer (真地图) + 新增的 dynamic_wall_layer (假地图)
#       plugins: ["static_layer", "dynamic_wall_layer", "obstacle_layer", "inflation_layer"]

#       # 原来的真实地图
#       static_layer:
#         plugin: "nav2_costmap_2d::StaticLayer"
#         map_topic: "/map"
#         map_subscribe_transient_local: True

#       # 新增的动态虚拟墙层
#       dynamic_wall_layer:
#         plugin: "nav2_costmap_2d::StaticLayer"
#         map_topic: "/dynamic_wall_map"  # 订阅伪造的地图
#         map_subscribe_transient_local: True
#         enabled: True
#         # ⚠️ 注意这个参数！如果设为 true，假地图里为 -1 的地方，会覆盖真实地图！
#         # 设为 false，意味着：假地图里是 100 的地方变成障碍，是 -1 的地方“不作数”（保留真地图原样）
#         track_unknown_space: false 
