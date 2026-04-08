from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # robot_count = 2

    return LaunchDescription([
        Node(
            package='robot_world',
            executable='robot_driver',
            name='bot_driver',
            namespace='robot_name',
            output='screen'
        ),
        Node(
            package='robot_world',
            executable='world_manager',
            name='world_manager',
            output='screen'
        ),
    ])
