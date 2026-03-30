

# Pull from Github

```sh
git clone git@github.com:chendawe/rl-navibot.git
# 工作目录应当映射到docker的~/workspace/rl-navibot
```

# Build ros2_my docker for my Gazebo simulation

```sh
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop
docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop ros2

docker builder prune -y
docker build --no-cache -t ros2_my -f ~/workspace/rl-navibot/docker/Dockerfile .
docker run -it \
    --gpus all \
    --shm-size=8g --privileged \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/dev/dri:/dev/dri \
    --device=/dev/snd \
    --env="DISPLAY=$DISPLAY" \
    --name=ros2my  \
    -v /home/chendawww/workspace:/root/workspace \
    ros2_my

# --user $(id -u):$(id -g)
# = 强制让 Docker 里面的用户，和你宿主机完全一样（1000:1000）
# 效果：
# Docker 里创建文件 → 宿主机直接能改
# 宿主机修改 → Docker 里也能读
# 两边完全一致，永远不报错！

# 或者宿主机：sudo chown -R 1000:1000 ~/workspace/rl-navibot

```

# 启动robot_world节点（容器内）

```sh
# source /opt/ros/humble/setup.sh && \
# cd ~/workspace/rl-navibot && \
# source install/setup.sh && \
# cd ~/workspace/turtlebot3_ws && \
# source install/setup.sh && \
# cd ~/workspace/nitrobot_ws && \
# source install/setup.sh && \
# source ~/.bashrc

# 窗口1
ros2 launch robot_world robot_world_sim.launch.py

# 测试频道状态
# 窗口2
ros2 topic echo /cmd_vel
# 窗口3
ros2 topic pub /rl_cmd geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.1}}" --once
```

# 启动 Gazebo 和 TurtleBot3

```sh
# 共享内存报错，process died的话：
# df -h /dev/shm
# mount -o remount,size=2G /dev/shm
# apt update && apt install ros-humble-rmw-cyclonedds-cpp -y
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# 通过设置环境变量，强制 ROS2 使用网络传输而不是共享内存，虽然效率稍微低一点点，但能绕过内存大小限制。

# 临时强制软件渲染
export LIBGL_ALWAYS_SOFTWARE=1

# 设置型号为 burger
export TURTLEBOT3_MODEL=waffle
# export TURTLEBOT3_MODEL=burger
# 启动 Gazebo house世界 (或者用其他世界也可以)
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```



```sh
gzserver --verbose
ps aux | grep gz
pkill -f gz
pkill -f gazebo

# 2. 杀死所有 gazebo 相关父进程（清理僵尸）
pkill -9 -f "gzserver|gzclient|gazebo"

# 3. 再检查一遍（现在一定干净了）
ps aux | grep gz
```


# 遇到的离谱问题
## ！！！：gazebo关于端口占用的问题
触发原因：用`ctrl+c`而非`ctrl+c`直接退出仿真命令，导致默认端口`11345`一直被占用（可查看gazebo的log得到报错）；此时再新开命令则会触发`code252...processing failed`

1. 检查`11345`端口占用
```sh
# 方案 A（推荐）
lsof -i :11345 -P -n

# 方案 B
ss -tulnp | grep 11345

# 方案 C
netstat -tulnp | grep 11345

# 开杀
kill -9 <PID1> <PID2> ...
```

2. 用别的端口号测试占用是否成立
```sh
# 用11346尝试
export GAZEBO_MASTER_URI=http://localhost:11346
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo empty_world.launch.py

# 换回默认11345
unset GAZEBO_MASTER_URI
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo empty_world.launch.py

```
## ！！！：ros2插件找不到
在容器中执行以下命令：
```sh
# 1. 查找插件文件是否存在
find / -name "libgazebo_ros_factory.so" 2>/dev/null

# 2. 检查当前 GAZEBO_PLUGIN_PATH
echo $GAZEBO_PLUGIN_PATH

# 3. 设置正确的插件路径（根据 find 结果调整）
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}

# 4. 验证设置
echo $GAZEBO_PLUGIN_PATH
```

