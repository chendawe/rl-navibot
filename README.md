# BUILD and RUN

## Build Python ROS2 environment
1. 设置语言环境：
```sh
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```
2. 添加ROS2 apt仓库：
```sh
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
3. 安装ROS2包：
```sh
sudo apt update
sudo apt install ros-humble-desktop  # 根据您的需求选择desktop或base
```

4. 环境变量配置（每次使用前需要执行）：
```sh
source /opt/ros/humble/setup.bash
# source ~/.bashrc
```


### 设置jupyter kernel的ros2的python和C库路径，让notebook能用到对应文件夹下的python库和c库
1. 先确认你当前用的是哪个 kernel：
```sh
jupyter kernelspec list
```

2. 创建并运行`wrapper.sh`文件，以用户权限级别在`~/.local/share/jupyter/kernels`创建和`/home/chendawww/Software/anaconda3/envs/ros2`一体的环境`ros2`，在启动后者的`ros2`环境前会先运行`~/.local/share/jupyter/kernels/ros2`中的`start.sh`来`source /opt/ros/humble/setup.bash`：
```
# wrapper.sh
mkdir -p ~/.local/share/jupyter/kernels/ros2

cat > ~/.local/share/jupyter/kernels/ros2/start.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
exec /home/chendawww/Software/anaconda3/envs/ros2/bin/python -m ipykernel_launcher "$@"
EOF
chmod +x ~/.local/share/jupyter/kernels/ros2/start.sh

cat > ~/.local/share/jupyter/kernels/ros2/kernel.json << 'EOF'
{
  "argv": ["/home/chendawww/.local/share/jupyter/kernels/ros2/start.sh", "-f", "{connection_file}"],
  "display_name": "ROS2 Humble",
  "language": "python",
  "metadata": {"debugger": true},
  "kernel_protocol_version": "5.5"
}
EOF
```

3. 运行 `jupyter kernelspec install --replace --user ~/.local/share/jupyter/kernels/ros2` → 把内核注册进 Jupyter

4. 重启 VSCode → 刷新内核列表

## Pull from Github

```sh
git clone git@github.com:chendawe/rl-navibot.git
# 工作目录应当映射到docker的~/workspace/rl-navibot
```

## Build `ros2my` docker for my Gazebo simulation

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
    --network host -e ROS_DOMAIN_ID=0 \
    --env="DISPLAY=$DISPLAY" \
    --name=ros2my  \
    -v /home/chendawww/workspace:/root/workspace \
    ros2_my

# --network host -e ROS_DOMAIN_ID=0，让wsl和docker的频道能够贯通

# --user $(id -u):$(id -g)
# = 强制让 Docker 里面的用户，和你宿主机完全一样（1000:1000）
# 效果：
# Docker 里创建文件 → 宿主机直接能改
# 宿主机修改 → Docker 里也能读
# 两边完全一致，永远不报错！

# 或者宿主机：sudo chown -R 1000:1000 ~/workspace/rl-navibot

```

## 启动robot_world节点（容器内）

```sh
# source /opt/ros/humble/setup.sh && \
# cd ~/workspace/rl-navibot && \
# source install/setup.sh && \
# cd ~/workspace/turtlebot3_ws && \
# source install/setup.sh && \
# cd ~/workspace/nitrobot_ws && \
# source install/setup.sh && \
# source ~/.bashrc

# export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_no_shm.xml
# export RMW_FASTRTPS_USE_QOS_FROM_XML=0

# 窗口1
ros2 launch robot_world robot_world_sim.launch.py

# 测试频道状态
# 窗口2
ros2 topic echo /cmd_vel
# 窗口3
ros2 topic pub /rl_cmd geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.1}}" --once
```

## 启动 Gazebo 和 TurtleBot3

1. 为`turtlebot3`的`House`世界增加`gazebo_ros` 功能包里的 `gazebo_ros_state`服务插件：

在`turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/worlds/turtlebot3_house.world`文件中添加这一项：
```xml
    <!-- 就是加上下面这段！！！ -->
    <plugin name="gaz1ebo_ros_state" filename="libgazebo_ros_state.so">
      <ros>
        <namespace>/</namespace>
      </ros>
    </plugin>
```
2. launch 仿真
```sh
# 共享内存报错，process died的话：
# df -h /dev/shm
# mount -o remount,size=2G /dev/shm
# apt update && apt install ros-humble-rmw-cyclonedds-cpp -y
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# 通过设置环境变量，强制 ROS2 使用网络传输而不是共享内存，虽然效率稍微低一点点，但能绕过内存大小限制。

# 临时强制软件渲染
export LIBGL_ALWAYS_SOFTWARE=1
# unset LIBGL_ALWAYS_SOFTWARE

# 设置型号为 burger
export TURTLEBOT3_MODEL=burger
# export TURTLEBOT3_MODEL=burger
# 启动 Gazebo house世界 (或者用其他世界也可以)
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```



## 启动网页app界面

### `Fastpi+Uvicorn` docker


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



# 项目设计

- 分三段
    - 点到点避障导航；
        - SAC训练
        - 基础MPC
        - 深度图 -> 24维障碍矢量；or 代价地图+CNN -> 特征矢量
    - 已知图，给定任务，基于完整结构化图，LLM规划路线行驶；
        - SLAM图 -> 结构化图
    - 临时建图，给定任务，基于即时结构图碎片，LLM规划路线行驶。
        - SLAM图，GroundingDINO -> 结构化图

r_distance：接近目标奖励。
r_collision：碰撞大惩罚。
r_smooth：动作平滑奖励（避免疯狂抖动）。


- Embedding Intelligent
    - Perception
    - Planning
    - Decision
    - Execution

- RL Agent
    - Body
        - Turtlebot3
    - Environment
        - Turtlebot3 house simulation

    - Reward
    - Algorithm
        - SAC
        - MPC
            - Asynchronous, Low-frequency
            - weighted
    - 

- Sim to Real
    - Ros2 + Gazebo
    - Domain Randomization



