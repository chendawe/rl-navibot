git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git src/turtlebot3
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git src/turtlebot3_simulations


# 不装的话没有turtle3bot自己的teleop
apt install -y ros-humble-turtlebot3*

# 先设置模型（burger/waffle/waffle_pi）
export TURTLEBOT3_MODEL=burger
# 启动仿真世界
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
ros2 run turtlebot3_teleop teleop_keyboard




# 进入 turtlebot3_gazebo 的 worlds 目录
roscd turtlebot3_gazebo/worlds      # ROS1
# 或
cd $(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/worlds  # ROS2

ls *.world

export TURTLEBOT3_MODEL=burger   # 或 waffle / waffle_pi
ros2 launch turtlebot3_gazebo empty_world.launch.py
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
# ROS1 示例
roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch
roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
...
