# core/ros2/channels/actors/map_boundary_guard.py

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import CostmapFilterInfo
from enum import Enum

import logging
logger = logging.getLogger(__name__)


class ExpandMode(Enum):
    OUTWARD = "outward"
    INWARD = "inward"
    CENTER = "center"


class MapBoundaryGuardInjector(Node):
    """
    环境影响器：直接生成 OccupancyGrid 注入 Nav2 KeepoutFilter。
    这是最贴合 Nav2 底层源码的直连方式，无需额外中间件。
    """
    def __init__(self, runtime=None, 
                 min_x: float = -10.0, max_x: float = 10.0,
                 min_y: float = -10.0, max_y: float = 10.0,
                 thickness: float = 1.2,
                 mode: str = "outward",
                 resolution: float = 0.05): # 假地图的分辨率，通常和真地图保持一致或稍低
        super().__init__('boundary_guard')
        if runtime:
            runtime.register_node(self)
            logger.info(f"🛡️ [{self.get_name()}] 已入驻，准备构建虚拟边界...")
        else:
            # 如果没有 runtime，自己创建一个 executor 转
            self._standalone_executor = rclpy.executors.MultiThreadedExecutor()
            self._standalone_executor.add_node(self)
    
        try:
            self.expand_mode = ExpandMode(mode.lower())
        except ValueError:
            logger.error(f"不支持的扩展模式: {mode}，回退到 outward")
            self.expand_mode = ExpandMode.OUTWARD
            
        self.thickness = thickness
        self.resolution = resolution

        # 1. 计算出这面墙占据的真实物理范围和像素范围
        # 返回: (grid_origin_x, grid_origin_y, grid_width_px, grid_height_px)
        self.origin_x, self.origin_y, self.width_px, self.height_px = self._calc_grid_bounds(min_x, max_x, min_y, max_y, thickness)
        
        # 2. 用 numpy 生成“只有墙”的二维数组
        grid_array = self._generate_wall_grid(min_x, max_x, min_y, max_y, thickness)

        # 3. ⚠️ 关键 QoS：必须是 TRANSIENT_LOCAL，否则 Nav2 绝对收不到！
        qos_transient = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # 4. 创建发布器
        # 注意：Info 里的 base 字段，必须和 Mask 的话题名一模一样！
        self.mask_topic_name = "/keepout_filter_mask"
        self.info_pub = self.create_publisher(CostmapFilterInfo, '/keepout_filter_info', qos_transient)
        self.mask_pub = self.create_publisher(OccupancyGrid, self.mask_topic_name, qos_transient)

        # 5. 将生成的数组封装成 ROS 消息，缓存起来
        self._mask_msg = self._build_grid_msg(grid_array)

        # 6. 智能等待发射
        self._timer_cg = MutuallyExclusiveCallbackGroup()
        self._timer = self.create_timer(1.0, self._check_and_publish, callback_group=self._timer_cg)

    def _calc_grid_bounds(self, bx_min, bx_max, by_min, by_max, t):
        """计算假地图的原点和尺寸"""
        if self.expand_mode == ExpandMode.OUTWARD:
            ox, oy = bx_min - t, by_min - t
            w = (bx_max - bx_min) + 2 * t
            h = (by_max - by_min) + 2 * t
        elif self.expand_mode == ExpandMode.INWARD:
            ox, oy = bx_min, by_min
            w, h = (bx_max - bx_min), (by_max - by_min)
        else: # CENTER
            half_t = t / 2.0
            ox, oy = bx_min - half_t, by_min - half_t
            w = (bx_max - bx_min) + t
            h = (by_max - by_min) + t
            
        w_px = int(np.ceil(w / self.resolution))
        h_px = int(np.ceil(h / self.resolution))
        return ox, oy, w_px, h_px

    def _generate_wall_grid(self, bx_min, bx_max, by_min, by_max, t):
        """核心逻辑：用 numpy 切片画出 4 面墙（杜绝角落漏风）"""
        # 初始化全 0 (在 KeepoutFilter 里，0 代表非禁区，100 代表禁区)
        grid = np.zeros((self.height_px, self.width_px), dtype=np.int8)

        # 将物理坐标转换为像素索引
        def to_px(phys_x, phys_y):
            px = int(round((phys_x - self.origin_x) / self.resolution))
            py = int(round((phys_y - self.origin_y) / self.resolution))
            return px, py

        t_px = int(round(t / self.resolution))

        # 根据模式计算四面墙的像素坐标范围 (左闭右开)
        if self.expand_mode == ExpandMode.OUTWARD:
            # 注意 numpy 数组是 [行(对应Y), 列(对应X)]
            # 下边墙: Y从 0 到 t_px
            _, y_end = to_px(0, by_min + t_px)
            grid[0:y_end, :] = 100
            
            # 上边墙: Y从 (h - t_px) 到 h
            y_start, _ = to_px(0, by_max)
            grid[y_start:self.height_px, :] = 100
            
            # 左边墙: X从 0 到 t_px (Y拉通防漏风)
            _, x_end = to_px(bx_min + t_px, 0)
            grid[:, 0:x_end] = 100
            
            # 右边墙: X从 (w - t_px) 到 w (Y拉通防漏风)
            x_start, _ = to_px(bx_max, 0)
            grid[:, x_start:self.width_px] = 100

        elif self.expand_mode == ExpandMode.INWARD:
            _, y_end = to_px(0, by_min + t_px)
            grid[0:y_end, :] = 100

            y_start, _ = to_px(0, by_max - t_px)
            grid[y_start:self.height_px, :] = 100

            _, x_end = to_px(bx_min + t_px, 0)
            grid[:, 0:x_end] = 100

            x_start, _ = to_px(bx_max - t_px, 0)
            grid[:, x_start:self.width_px] = 100

        else: # CENTER
            half_t_px = t_px // 2
            grid[0:half_t_px, :] = 100
            grid[self.height_px - half_t_px:self.height_px, :] = 100
            grid[:, 0:half_t_px] = 100
            grid[:, self.width_px - half_t_px:self.width_px] = 100

        return grid

    def _build_grid_msg(self, grid_array) -> OccupancyGrid:
        """把 numpy 数组包装成 ROS 消息"""
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = self.width_px
        msg.info.height = self.height_px
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid_array.flatten().tolist()
        return msg

    def _check_and_publish(self):
        if self.info_pub.get_subscription_count() > 0:
            # 发说明书
            info_msg = CostmapFilterInfo()
            info_msg.type = 1  # 1 = KEEPOUT
            info_msg.base = self.mask_topic_name  # 指向 Mask 话题
            self.info_pub.publish(info_msg)

            # 发假地图
            # 注意：每次发前更新一下时间戳，防止某些版本 Nav2 的缓存问题
            self._mask_msg.header.stamp = self.get_clock().now().to_msg()
            self.mask_pub.publish(self._mask_msg)

            logger.info(f"🧱 [{self.get_name()}] 检测到 Nav2 就绪，OccupancyGrid 围墙已发射！")
            self._timer.cancel()

# ... 前面的类代码保持不变 ...

import sys

def main(args=None):
    # 1. 常规 ROS2 节点初始化
    rclpy.init(args=args)
    
    # 2. 实例化节点（不传 runtime，所以我们在 __init__ 里要处理一下）
    # 稍微修改一下 __init__，让 runtime 变成可选参数：
    # def __init__(self, runtime=None, ...):
    #     super().__init__('boundary_guard')
    #     if runtime:
    #         runtime.register_node(self)
    #     else:
    #         # 如果没有 runtime，自己创建一个 executor 转
    #         self._standalone_executor = rclpy.executors.MultiThreadedExecutor()
    #         self._standalone_executor.add_node(self)
    
    injector = MapBoundaryGuardInjector(
        min_x=-10.0, max_x=10.0, 
        min_y=-10.0, max_y=10.0, 
        thickness=1.2, 
        mode="outward"
    )
    
    # 3. 阻塞 spin（如果没有你的 Ros2Runtime，它就自己转）
    if hasattr(injector, '_standalone_executor'):
        injector._standalone_executor.spin()
    
    # 4. 退出清理
    injector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# 通过vector object实现，要给nav2用的话需要server、中转转发

# # core/ros2/channels/actors/map_boundary_guard.py

# import rclpy
# from rclpy.node import Node
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# from nav2_msgs.msg import CostmapFilterInfo
# from geometry_msgs.msg import PolygonStamped, Polygon, Point32
# from enum import Enum

# import logging
# logger = logging.getLogger(__name__)


# class ExpandMode(Enum):
#     OUTWARD = "outward"   # 给定线往外扩
#     INWARD = "inward"     # 给定线往内扩
#     CENTER = "center"     # 给定线作为中心线，两边各扩一半


# class MapBoundaryGuardInjector(Node):
#     """
#     环境影响器：向 Nav2 的 global_costmap 注入虚拟围墙。
#     支持基于给定基准线的向外、向内、居中扩展。
#     """

#     def __init__(self, runtime, 
#                  min_x: float = -10.0, max_x: float = 10.0,
#                  min_y: float = -10.0, max_y: float = 10.0,
#                  thickness: float = 1.2,
#                  mode: str = "outward"):  # 新增 mode 参数
        
#         # 1. 初始化原始 Node
#         super().__init__('boundary_guard')
        
#         # 2. 自动入驻 Ros2Runtime 的线程池
#         runtime.register_node(self)
#         logger.info(f"🛡️ [{self.get_name()}] 已入驻，准备构建虚拟边界...")

#         # 3. 解析扩展模式
#         try:
#             self.expand_mode = ExpandMode(mode.lower())
#         except ValueError:
#             logger.error(f"不支持的扩展模式: {mode}，请使用 outward/inward/center")
#             self.expand_mode = ExpandMode.OUTWARD
            
#         self.thickness = thickness
        
#         # 4. 根据模式生成严丝合缝的 4 面墙 (核心逻辑)
#         self.walls_polygon = self._calculate_walls(min_x, max_x, min_y, max_y, thickness)

#         # 5. 创建发布器
#         self.info_pub = self.create_publisher(CostmapFilterInfo, '/keepout_filter_info', 10)
#         self.poly_pub = self.create_publisher(PolygonStamped, '/keepout_polygons_topic', 10)

#         # 6. 延迟发送
#         self._timer_cg = MutuallyExclusiveCallbackGroup()
#         self._timer = self.create_timer(
#             3.0, self._publish_boundaries, callback_group=self._timer_cg
#         )

#     def _calculate_walls(self, bx_min, bx_max, by_min, by_max, t):
#         """
#         计算四面墙的坐标。
#         核心原则：无论哪种模式，四个角落必须强制重叠，杜绝光栅化对角漏风。
#         """
#         walls = []
        
#         if self.expand_mode == ExpandMode.OUTWARD:
#             # 往外扩：墙在给定边界的外侧
#             # 四面墙全部拉通到外边界，角落必然产生 t*t 大小的实心重叠块，绝对封死
#             walls.append(self._create_wall_rect(bx_min - t, by_min - t, bx_max + t, by_min)) # 下
#             walls.append(self._create_wall_rect(bx_min - t, by_max, bx_max + t, by_max + t)) # 上
#             walls.append(self._create_wall_rect(bx_min - t, by_min - t, bx_min, by_max + t)) # 左 (Y拉通)
#             walls.append(self._create_wall_rect(bx_max, by_min - t, bx_max + t, by_max + t)) # 右 (Y拉通)
            
#         elif self.expand_mode == ExpandMode.INWARD:
#             # 往内扩：墙在给定边界的内侧
#             # 为了防止内部四个角漏风，左右墙的 Y 坐标同样拉通到底（会产生微小的内部实心角，但安全第一）
#             walls.append(self._create_wall_rect(bx_min, by_min, bx_max, by_min + t))         # 下
#             walls.append(self._create_wall_rect(bx_min, by_max - t, bx_max, by_max))         # 上
#             walls.append(self._create_wall_rect(bx_min, by_min, bx_min + t, by_max))         # 左 (Y拉通)
#             walls.append(self._create_wall_rect(bx_max - t, by_min, bx_max, by_max))         # 右 (Y拉通)
            
#         elif self.expand_mode == ExpandMode.CENTER:
#             # 居中扩：给定边界作为墙的中心线
#             half_t = t / 2.0
#             # 同样采用角落全拉通重叠策略
#             walls.append(self._create_wall_rect(bx_min - half_t, by_min - half_t, bx_max + half_t, by_min + half_t)) # 下
#             walls.append(self._create_wall_rect(bx_min - half_t, by_max - half_t, bx_max + half_t, by_max + half_t)) # 上
#             walls.append(self._create_wall_rect(bx_min - half_t, by_min - half_t, bx_min + half_t, by_max + half_t)) # 左
#             walls.append(self._create_wall_rect(bx_max - half_t, by_min - half_t, bx_max + half_t, by_max + half_t)) # 右

#         return walls

#     def _create_wall_rect(self, x1, y1, x2, y2) -> Polygon:
#         """辅助函数：生成矩形多边形"""
#         poly = Polygon()
#         poly.points = [
#             Point32(x=x1, y=y1, z=0.0),
#             Point32(x=x2, y=y1, z=0.0),
#             Point32(x=x2, y=y2, z=0.0),
#             Point32(x=x1, y=y2, z=0.0)
#         ]
#         return poly

#     def _publish_boundaries(self):
#         """一次性发送说明书和形状"""
#         info_msg = CostmapFilterInfo()
#         info_msg.type = 1  # 1 = KEEPOUT
#         info_msg.base = "/keepout_polygons_topic"
#         self.info_pub.publish(info_msg)

#         for wall in self.walls_polygon:
#             poly_stamped_msg = PolygonStamped()
#             poly_stamped_msg.header.frame_id = 'map'
#             poly_stamped_msg.header.stamp = self.get_clock().now().to_msg()
#             poly_stamped_msg.polygon = wall
#             self.poly_pub.publish(poly_stamped_msg)

#         logger.info(f"🧱 [{self.get_name()}] 虚拟围墙已发射！模式: {self.expand_mode.value}, 厚度: {self.thickness}m")
#         self._timer.cancel()





# # 假设这是你的业务入口
# runtime = Ros2Runtime()

# # 1. 启动状态观察者
# map_provider = MapProvider(runtime, topic='/map')

# # 2. 启动环境影响器 (一行代码注入环境限制)
# # 参数可以根据你的实际房子大小动态传入
# boundary_guard = BoundaryGuard(
#     runtime, 
#     min_x=-8.0, max_x=8.0, 
#     min_y=-8.0, max_y=8.0, 
#     thickness=1.2  # 记得大于 2 倍 inflation_radius
# )

# # ... 后面启动 explore_lite 等其他逻辑 ...




# # 假设你的房子真实外墙正好是 -8 到 8

# # 1. 往外扩：墙贴着房子外墙外侧。适合“探索到墙外一点点就停下”
# guard_out = MapBoundaryGuardInjector(
#     runtime, 
#     min_x=-8.0, max_x=8.0, min_y=-8.0, max_y=8.0, 
#     thickness=1.5, 
#     mode="outward"
# )

# # 2. 往内扩：墙在房子外墙内侧。适合“不让机器人靠近真实外墙（比如怕撞坏墙纸）”
# guard_in = MapBoundaryGuardInjector(
#     runtime, 
#     min_x=-8.0, max_x=8.0, min_y=-8.0, max_y=8.0, 
#     thickness=0.5, 
#     mode="inward"
# )

# # 3. 居中扩展：给定的坐标正好是墙的正中心。适合“我量出来这面墙就在 X=8.0 这个位置”
# guard_center = MapBoundaryGuardInjector(
#     runtime, 
#     min_x=-8.0, max_x=8.0, min_y=-8.0, max_y=8.0, 
#     thickness=1.2, 
#     mode="center"
# )
