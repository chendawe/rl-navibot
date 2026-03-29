from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 启动机器人驱动节点
        Node(
            package='robot_world',
            executable='robot_driver',
            name='robot_driver',
            output='screen',
            # 这里暂时不需要参数，因为你的 C++ 代码里目前是写死的配置
        ),

        # 2. 启动世界管理节点
        Node(
            package='robot_world',
            executable='world_manager',
            name='world_manager',
            output='screen',
            parameters=[{
                'robot_name': 'robot',  # 这里假设你的 Gazebo 里的模型名字叫 'robot'
                'reset_x': 0.0,
                'reset_y': 0.0
            }]
        )
    ])
