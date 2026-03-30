

# Pull from Github

```sh
git clone git@github.com:chendawe/rl-navibot.git
# 工作目录应当映射到docker的~/workspace/rl-navibot
```

# Build ros2_my docker for my Gazebo simulation

```sh
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop
docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop ros2
docker run -it \
    --gpus all \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/dev/dri:/dev/dri \
    --device=/dev/snd \
    --env="DISPLAY=$DISPLAY" \
    --name=ros2  \
    -v /home/chendawww/workspace:/root/workspace \
    ros2

# --user $(id -u):$(id -g)
# = 强制让 Docker 里面的用户，和你宿主机完全一样（1000:1000）
# 效果：
# Docker 里创建文件 → 宿主机直接能改
# 宿主机修改 → Docker 里也能读
# 两边完全一致，永远不报错！

# 或者宿主机：sudo chown -R 1000:1000 ~/workspace/rl-navibot

docker build -t ros2_my -f /home/chendawww/workspace/rl-navi/Dockerfile .
```

# 启动robot_world节点（容器内）

```sh
# source /opt/ros/humble/setup.sh && \
# cd ~/workspace/rl-navibot && \
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