
# Use the base ROS2 humble image
FROM ros2
# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/osrf/ros:humble-desktop

RUN apt sudo apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:openrobotics/gazebo11-gz-cli \
    apt update \
    apt install gazebo11

RUN sudo apt update && \
    apt install \
        vim \
        pip \
        wget \
        rospdec \
        ros-humble-ros-gz \
        ros-humble-gazebo-ros2-control \
        ros-humble-xacro \
        ros-humble-twist-mux \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-joint-state-publisher-gui \
        ros-humble-controller-manager \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers

        ros-humble-simulation \
        ros-humble-robot \
        ros-humble-teleop-twist-keyboard

    apt install \
        ros-humble-diff-drive-controller \
        ros-humble-joint-state-broadcaster \
        ros-humble-controller-manager \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers

        # 配rosdep源

RUN export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml && \
    echo 'export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml' >> ~/.bashrc && \
    source ~/.bashrc && \
    rm -rf ~/.ros/rosdep/sources.cache


# sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" > /etc/apt/sources.list.d/gazebo-stable.list'
# wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add -

# # 3. 更新并安装
# apt update
# apt install -y gazebo11 libgazebo11-dev



