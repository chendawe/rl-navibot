from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction

import os

def generate_launch_description():

    # !!! 注意 !!! 这里默认用的是 turtlebot3_house
    # 如果你是自己的机器人（比如 rl-navibot），把下面两行改成你自己的包名和 launch 文件名
    gazebo_pkg = 'turtlebot3_gazebo'
    gazebo_launch = 'turtlebot3_house.launch.py'


# 找到根本原因了！这是一个 ROS 2 新手极其容易踩的巨坑：坐标系前导斜杠（/）问题。

# 你看你刚才贴的 /tf_static 输出：


#     frame_id: /base_footprint
#   child_frame_id: /base_link
# 还有：


#     frame_id: /base_link
#   child_frame_id: /base_scan
# 在 ROS 2 中，base_link 和 /base_link 被认为是两个完全不同的坐标系！

# 而你的 Gazebo 底盘控制插件（diff drive）默认发布的是没有斜杠的 odom -> base_footprint，你的 SLAM 发布的也是没有斜杠的 map -> odom。

# 这就导致你的 TF 树硬生生断裂成了两棵树：

# 树 1：map -> odom -> base_footprint (没斜杠)
# 树 2：/base_footprint -> /base_link -> /base_scan (有斜杠)
# Nav2 想找没有斜杠的 base_link，结果发现它在另一棵树上（或者根本不存在），于是直接报错罢工。


    return LaunchDescription([
        # # 0. 修复 URDF 斜杠导致的 TF 断裂问题
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='fix_base_footprint_slash',
        #     arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', '/base_footprint']
        # ),

        # 1. 启动 Gazebo 仿真环境
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    FindPackageShare(gazebo_pkg).find(gazebo_pkg),
                    'launch', gazebo_launch
                ),
            ),                          # ← PythonLaunchDescriptionSource 的右括号在这里
            launch_arguments={
                'use_sim_time': 'True',
                'gui': 'false',
            }.items(),                 # ← launch_arguments 结束
        ),                              # ← IncludeLaunchDescription 的右括号在这里


        # 2. 启动 SLAM（建图）
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    FindPackageShare('nav2_bringup').find('nav2_bringup'),
                    'launch', 'slam_launch.py'
                )
            ),
            launch_arguments={
                'use_sim_time': 'True',
                # 同样显式传入 params_file
                'params_file': os.path.join(
                    FindPackageShare('nav2_bringup').find('nav2_bringup'),
                    'params', 'nav2_params.yaml'
                ),
            }.items()
        ),


        # 3. 启动 Nav2 导航栈（显式传入 params_file 避免 Humble 空串 bug）
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(
        #             FindPackageShare('nav2_bringup').find('nav2_bringup'),
        #             'launch', 'navigation_launch.py'
        #         )
        #     ),
        #     launch_arguments={
        #         'use_sim_time': 'True',
        #         'map_subscribe_transient_local': 'True',
        #         # 显式传入参数文件路径（用字符串即可）
        #         # 'params_file': os.path.join(
        #         #     FindPackageShare('nav2_bringup').find('nav2_bringup'),
        #         #     'params', 'nav2_params.yaml'
        #         # ),
        #         'params_file': os.path.join(
        #             '/root/workspace/rl-navibot/src/core/core/launch',  # 获取当前 launch 文件所在的目录
        #             'config', 'nav2_params.yaml'           # 往上一级，进 config 文件夹
        #         ),
        #         # --- 下面这四行是解决卡顿加的 ---
        #         'controller_frequency': '10.0',      # 局部路径规划频率 (默认20，改小减轻CPU)
        #         'bt_loop_duration': '50',             # 行为树执行间隔毫秒 (对应降到20Hz，默认是10ms/100Hz)
        #         'local_costmap_update_frequency': '2.0',  # 局部代价地图更新频率 (默认5.0)
        #         'local_costmap_publish_frequency': '1.0',  # 局部代价地图发布频率 (默认2.0)
        #         # ------------------------------
        #     }.items()
        # ),
        # 启动 Nav2 导航 (注意：这里一定要加降频参数！！！)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                    FindPackageShare('nav2_bringup').find('nav2_bringup'),
                    'launch', 'navigation_launch.py'
                )),
            launch_arguments={
                'use_sim_time': 'True',
                'params_file': os.path.join(
                    '/root/workspace/rl-navibot/src/core/core/launch',  # 获取当前 launch 文件所在的目录
                    'config', 'nav2_params.yaml'           # 往上一级，进 config 文件夹
                ),
                # --- 下面这四行是解决卡顿的核心 ---
                'controller_frequency': '10.0',           
                'bt_loop_duration': '50',                
                'local_costmap_update_frequency': '2.0', 
                'local_costmap_publish_frequency': '1.0',
                # -----------------------------------
            }.items(),
        ),


        # 4. 启动 Explore Lite (延迟5秒启动，等待代价地图初始化)
        # TimerAction(
        #     period=30.0,  # 延迟 5 秒
        #     actions=[
        #         Node(
        #             package='explore_lite',
        #             executable='explore',
        #             name='explore_client',
        #             output='screen',
        #             parameters=[
        #                 {'use_sim_time': True},
        #                 {'robot_base_frame': 'base_link'}  # 顺便把这个写死，防止找成 /base_link
        #             ],
        #             remappings=[('costmap', 'global_costmap/costmap')]
        #         )
        #     ]
        # ),
        
        TimerAction(
            period=15.0,  # 延迟加到 15 秒，确保代价地图充分展开
            actions=[
                Node(
                    package='explore_lite',
                    executable='explore',
                    name='explore_client',
                    output='screen',
                    parameters=[
                        {'use_sim_time': True},
                        {'robot_base_frame': 'base_link'},
                        {'progress_timeout': 30.0},       # 30秒内没进展才放弃，而不是几秒就放弃
                        {'min_frontier_size': 0.3},        # 过滤掉太小的无效边界，只追大边界
                    ],
                    remappings=[('costmap', 'global_costmap/costmap')]
                )
            ]
        ),

    ])
