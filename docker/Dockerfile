
# Use the base ROS2 humble image
FROM ros2
# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop

# # 会装到gazebo11，但是add-apt-repository网络连不上，因此选择下面rosdep-key配置ros2源
# RUN apt sudo apt update && \
#     apt install -y software-properties-common && \
#     add-apt-repository ppa:openrobotics/gazebo11-gz-cli \
#     apt update \
#     apt install gazebo
    
# 配rosdep源
RUN export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml && \
    echo 'export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml' >> ~/.bashrc && \
    source ~/.bashrc && \
    rm -rf ~/.ros/rosdep/sources.cache

# 通过rosdep key装gazebo11（尽管似乎报错说deprecated），gazebo11不行就改成gazebo
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" > /etc/apt/sources.list.d/gazebo-stable.list' && \
    wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add - && \
    apt update && \
    apt install -y gazebo11 libgazebo11-dev && \

RUN sudo apt update && \
    apt install \
        vim \
        pip \
        wget \
        rosdepc \
        ros-humble-simulation \
        ros-humble-robot \
        ros-humble-teleop-twist-keyboard

        # ros-humble-ros-gz \
        # ros-humble-gazebo-ros2-control \
        # ros-humble-xacro \
        # ros-humble-twist-mux \
        # ros-humble-robot-state-publisher \
        # ros-humble-joint-state-publisher \
        # ros-humble-joint-state-publisher-gui \
        # ros-humble-controller-manager \
        # ros-humble-ros2-control \
        # ros-humble-ros2-controllers

# 装ros2控制包
RUN apt install \
        ros-humble-diff-drive-controller \
        ros-humble-joint-state-broadcaster \
        ros-humble-controller-manager \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers

# 为了多开bash cmd方便，持久化ros2和rl-navibot工作环境
RUN source /opt/ros/humble/setup.sh && \
    cd ~/workspace/rl-navibot && \
    source install/setup.sh && \
    source ~/.bashrc

# 安装turtlebot3和nitrobot