# start_explore_with_walls.launch.py

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # 1. 找到 Turtlebot3 仿真 launch 文件的绝对路径
    gazebo_launch_dir = FindPackageShare('turtlebot3_gazebo').find('turtlebot3_gazebo')
    gazebo_launch_file = os.path.join(gazebo_launch_dir, 'launch', 'turtlebot3_house.launch.py')
    
    # 2. 找到 Nav2 SLAM launch 文件的绝对路径 (请根据你实际用的包名调整，比如 nav2_bringup 或 slam_toolbox)
    # 这里假设你用的是 nav2_bringup 的 slam_launch.py
    nav2_launch_dir = FindPackageShare('nav2_bringup').find('nav2_bringup')
    slam_launch_file = os.path.join(nav2_launch_dir, 'launch', 'slam_launch.py')

    # 3. 你的 Python 注入脚本的绝对路径
    # ⚠️ 注意：这里必须填你电脑上 map_boundary_guard.py 的绝对路径！
    injector_script_path = '~/workspace/rl-navibot/build/core/core/ros2/channels/actors/map_boundary_guard.py'

    return LaunchDescription([
        # 动作 A：启动 Gazebo 仿真环境
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_file),
        ),

        # 动作 B：启动 SLAM 建图
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch_file),
            launch_arguments={'use_sim_time': 'True'}.items()
        ),

        # 动作 C：启动你的虚拟墙注入器
        # 使用 ExecuteProcess 就相当于在终端里敲了一行 python3 xxx.py
        # 不用加延迟！因为你代码里写了等订阅者，它自己会卡住等 Nav2 准备好
        ExecuteProcess(
            cmd=['python3', injector_script_path],
            output='screen', # 把 print/log 输出到当前终端方便你调试
        ),
        
        # 动作 D：(可选) 启动 Explore Lite 探索节点
        # 假设你的 explore 节点也是用类似方式启动的
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'explore_lite', 'explore'],
        #     output='screen',
        # ),
    ])
