# -2. Tech Stack


## Notes
- Tech note see
- Debug note see

# -1. Project Structure


# 0. Environment Building

## `Linux` and `Ros2` :
- `Ubuntu` version :
```sh
lsb_release -a
# No LSB modules are available.
# Distributor ID: Ubuntu
# Description:    Ubuntu 22.04.5 LTS
# Release:        22.04
# Codename:       jammy
```

- Corresponding Ros2 version :
```sh
echo $ROS_DISTRO
# humble
```

## Pull from Github
```sh
cd /path/to/workspace
git clone git@github.com:chendawe/rl-navibot.git
```

## Build `conda` :

- 创建conda环境+安装依赖：
```sh
# 1. 创建环境，名字=ros2，python=3.10
conda create -n ros2 python=3.10.20 -y

# 2. 激活环境
conda activate ros2

cd ~/workspace/rl-navibot
# 3. 用 environment.yml 安装 conda 依赖
conda env update -f environment.yml

# 4. 用 requirements.txt 安装 pip 依赖
pip install -r requirements.txt
```
---
- 在当前的`ros2`conda环境中绑定`Ros`包环境：
<!-- 
在 Jupyter 中手动添加一个名为 "ROS2 Humble" 的 Python 内核，使得你可以在 Jupyter Notebook / JupyterLab 中直接运行带有 ROS 2 环境的 Python 代码。 -->

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
---
build `rl-navibot` 的包：
```sh
cd ~/rl-navibot
colcon build
```
---
---
## Build `Docker` :
- build ros2-gazebo docker
```sh
# 从华为云拉ros2-humble的docker镜像
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop
docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop ros2

# ros2镜像基础上build必要的包库依赖（gazebo为主）
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
```
```
# --network host -e ROS_DOMAIN_ID=0，让wsl和docker的频道能够贯通
# --user $(id -u):$(id -g)
# = 强制让 Docker 里面的用户，和你宿主机完全一样（1000:1000）
# 效果：
# Docker 里创建文件 → 宿主机直接能改
# 宿主机修改 → Docker 里也能读
# 两边完全一致，永远不报错！
# 或者宿主机：sudo chown -R 1000:1000 ~/workspace/rl-navibot
```

docker内build turtlebot3的包see：https://github.com/ROBOTIS-GIT/turtlebot3


## env vars :
- 宿主机：
```sh
# 加载rl-navibot包的环境变量
cd ~/workspace/rl-navibot && \
source install/setup.sh && \

# 创建 FastDDS 配置文件，ros2通信方式由SHM替换为UDP，让宿主机可以监听到容器ros2发布的频道（要求容器启动时设置为host）
cat > ~/fastdds_no_shm.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>udp_transport</transport_id>
            <type>UDPv4</type>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="disable_shm_participant" is_default_profile="true">
        <rtps>
            <userTransports>
                <transport_id>udp_transport</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
        </rtps>
    </participant>
</profiles>
EOF
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_no_shm.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
```

- `ros2my`docker内：

```sh
# 加载turtlebot3包的环境变量
source /opt/ros/humble/setup.sh && \
cd ~/workspace/turtlebot3_ws && \
source install/setup.sh && \
source ~/.bashrc
```

## Required library and modules
- Nav2
```sh
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-cartographer ros-humble-cartographer-ros
```
# 1. Boot
## boot Linux
```sh
wsl
```

## boot conda
```sh
conda activate ros2
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_no_shm.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
# 忘了这两行会reset_world失败
```

## boot Docker

- start `gazebo` cmd :

启动容器：
```sh
docker exec -it ros2my bash
```
```sh
# 容器内配置环境变量：
source /opt/ros/humble/setup.sh && \
cd ~/workspace/turtlebot3_ws && \
source install/setup.sh && \
source ~/.bashrc
```

启动`waffle`in`house`gazebo仿真节点：
```sh
# export TURTLEBOT3_MODEL=burger
# ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

- strat `map` ndoe cmd :
新开一个容器命令行：
```sh
docker exec -it ros2my bash
```
```sh
# 容器内配置环境变量：
source /opt/ros/humble/setup.sh && \
cd ~/workspace/turtlebot3_ws && \
source install/setup.sh && \
source ~/.bashrc
```
启动`slam`建图的节点：
```sh
ros2 launch nav2_bringup slam_launch.py use_sim_time:=True
```
## boot web monitor via `uvicorn`
```sh
~/workspace/rl-navibot/app/start.sh
```

# 2. Train, Eval and Play
## RL strategy


# 3. Monitor on the Web

# 4. useful comand lines