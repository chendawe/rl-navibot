# Set ros2
source /opt/ros/humble/setup.sh

ros2 pkg create --build-type ament_cmake bots

ros2 pkg list


cd /home/chendawww/workspace/rl-navibot/src/execution/bot_world/src

sudo rosdep update

rosdepc install --from-paths src --ignore-src -y

colcon build --symlink-install

# pkg打包完改限，不然宿主机无法编辑打包目录
# sudo chown -R 1000:1000 /home/chendawww/workspace/rl-navibot



# 测试
# 打包重命名出现问题，或者代码改完发现build结果没变，或者突然依赖报奇怪的错
# 1. 清除所有旧的 ROS 和 Colcon 环境变量
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH

# 2. 重新加载 ROS2 基础环境
source /opt/ros/humble/setup.bash

# 3. 清理之前的编译缓存 (这一步很重要，防止旧的错误配置残留)
rm -rf build install log
# 如果你的工作空间已经编译成功，可以 source 它
source install/setup.bash


colcon test --packages-select robot_world
colcon test-result --verbose

# 格式纠错，根目录运行：
ament_uncrustify --reformat






# 运行
source install/setup.bash
ros2 launch robot_world robot_world_sim.launch.py


ros2 topic echo /cmd_vel
ros2 topic pub /rl_cmd geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.1}}" --once
