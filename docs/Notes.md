
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

## FastRTPS共享内存出问题，切换dds实现设置udp发送指令
```sh
# 清理残留文件
rm -rf /dev/shm/fastrtps_*
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic pub /rl_cmd geometry_msgs/msg/Twist "{linear: {x: 2}, angular: {z: 0.1}}"

unset RMW_IMPLEMENTATION
echo $RMW_IMPLEMENTATION
```

## 宿主机到 Docker 容器的 DDS 通信问题（导致）
```
[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7411: open_and_lock_file failed
```
原因：FastDDS 共享内存传输跨容器失败：
Gazebo（容器内）和你的训练脚本（宿主机）使用的是 FastDDS，它默认优先尝试共享内存（SHM）传输数据。共享内存只能在同一台机器的同一个 OS 命名空间内使用，跨 Docker 容器边界是不通的。DDS 发现共享内存失败后，按理应该回退到 UDP，但在某些配置下会卡住不回退。

- 方案A：在宿主机禁用 FastDDS 共享内存传输
1. 创建 FastDDS 配置文件：
```sh
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
```
2. 在你的宿主机训练终端里，每次运行前设置环境变量：
```sh
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_no_shm.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
```
3. 宿主机和docker内测试命令：
```sh
ros2 service call /set_entity_state gazebo_msgs/srv/SetEntityState "{state: {name: 'waffle', pose: {position: {x: 0.0, y: 0.0, z: 0.1}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}, reference_frame: 'world'}}"

```

- 方案B：Python代码也在容器内跑


## 重置环境
在`/home/chendawww/workspace/turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/worlds/turtlebot3_house.world`中设置插件：

```xml
  <!-- world 文件里必须有 -->
  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
    <ros>
      <namespace>/</namespace>
    </ros>
  </plugin>
  <plugin name="gazebo_ros_factory" filename="libgazebo_ros_factory.so">
    <ros>
      <namespace>/</namespace>
    </ros>
  </plugin>
```
```sh
(ros2) (base) chendawww@cdws:~/workspace$ ros2 service list | grep entity
# /delete_entity
# /get_entity_state
# /set_entity_state
# /spawn_entity
```

## 添加SLAM节点

0. 检查节点和频道存在性
```sh
ros2 node list
ros2 topic list
```

1. 确认slam.launch位置
```sh
root@cdws:~/workspace/turtlebot3_ws# ls /opt/ros/humble/share/nav2_bringup/launch/
# bringup_launch.py                      localization_launch.py  rviz_launch.py  tb3_simulation_launch.py
# cloned_multi_tb3_simulation_launch.py  navigation_launch.py    slam_launch.py  unique_multi_tb3_simulation_launch.py
```
2. 启动节点
```sh
# 启动SLAM节点（需根据机器人配置调整参数）
ros2 launch nav2_bringup slam_launch.py use_sim_time:=True
```
3.启动后出现相应slam节点和频道
```sh
(ros2) chendawww@cdws:~/workspace/rl-navibot$ ros2 node list
# /lifecycle_manager_slam
# /map_saver
# /slam_toolbox
# /transform_listener_impl_6010c4b9e460
(ros2) chendawww@cdws:~/workspace/rl-navibot$ ros2 topic list
# /bond
# /clock
# /diagnostics
# /map
# /map_metadata
# /map_saver/transition_event
# /parameter_events
# /pose
# /rosout
# /scan
# /slam_toolbox/feedback
# /slam_toolbox/graph_visualization
# /slam_toolbox/scan_visualization
# /slam_toolbox/update
# /tf
# /tf_static
```
---
---
---
这是一份为你整理的排查笔记，你可以直接保存到你的博客、Obsidian 或笔记软件中：
ROS2 踩坑记录：Docker容器与WSL工作空间的“幽灵污染”
1. 问题现象
- bot仿真包在容器内build，navibot在wsl宿主机build
在 Docker 容器中启动原版的 TurtleBot3 Gazebo 仿真环境：
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
莫名报错：package 'perception' not found。
注：perception 是我自己写的包，属于 rl-navibot 工作空间，且该工作空间是在 WSL 本地编译的，理论上和 Docker 容器“八竿子打不着”。
2. 初始疑惑与反常现象
隔离性悖论：Docker 容器里的仿真，为什么会去寻找 WSL 里的包？
Launch 文件无误：使用 cat 查看 turtlebot3_world.launch.py，里面干干净净，根本没有调用 perception。
环境变量污染：在容器里哪怕只 source /opt/ros/humble/setup.bash 和 turtlebot3_ws，echo $AMENT_PREFIX_PATH 里依然会像“俄罗斯套娃”一样，把 nitrobot_ws 和 rl-navibot 的路径带出来。
3. 排查过程与“真凶”
通过 grep 命令直接搜查容器里 turtlebot3_ws 的底层文件：
grep -r "rl-navibot" /root/workspace/turtlebot3_ws/install/
发现大量 setup.sh 和 parent_prefix_path 文件中写死了 /root/workspace/rl-navibot/install 的路径。
真凶定位：Docker Volume 挂载 + Colcon 编译机制
物理层面没有隔离：我的 Docker 启动命令使用了 -v /home/chendawww/workspace:/root/workspace。这意味着 WSL 的 workspace 和 Docker 的 workspace 是同一块物理硬盘上的同一个文件夹。
Colcon 的“烙印”机制：Colcon 在编译时，会把当前终端的 COLCON_PREFIX_PATH（即已 source 的工作空间）作为父级依赖，写死到 install/setup.sh 中。
误操作发生：很可能是在前天或昨天，我在终端带着 rl-navibot 环境的情况下，不小心在容器里（或者在 WSL 的 workspace 根目录下）执行了 colcon build。这次“脏环境”下的编译，直接通过挂载目录覆盖了以前干净的 turtlebot3_ws 编译产物。
4. 解决步骤
核心思路：用绝对干净的环境覆盖掉被污染的产物。
第一步：清理 workspace 根目录的残留（如果是误在根目录 build 的）
# 检查根目录是否有编译产物
ls /root/workspace/install/
# 如果有，直接删除（不影响子目录里的包）
rm -rf /root/workspace/install /root/workspace/build /root/workspace/log
第二步：在容器内干净重编 TurtleBot3
# 1. 开启一个没有任何环境变量的纯净 bash
bash --norc --noprofile
# 2. 只加载 ROS2 基础环境
source /opt/ros/humble/setup.bash
# 3. 重新编译（这会生成干净的 setup.sh 覆盖掉被污染的）
cd /root/workspace/turtlebot3_ws
colcon build
# 4. 退出纯净 bash
exit
执行完后，再次正常 source 启动，问题解决，不再寻找 perception。
5. 经验总结（避坑指南）
💡 原则一：编译前必须“看一眼”
无论在 WSL 还是在 Docker 里执行 colcon build 之前，养成肌肉记忆，先看一眼当前环境：
echo $AMENT_PREFIX_PATH
如果里面出现了你当前不需要编译的无关工作空间路径，绝对不要直接 build。
💡 原则二：跨机/跨容器通信不靠“共享 Source”
WSL（跑算法）和 Docker（跑仿真）之间通信，靠的是 DDS 网络（已通过 --network host 和 ROS_DOMAIN_ID=0 实现），不需要在容器里 source 算法包的工作空间。容器里只需要 source 仿真相关的包。
💡 原则三：理解 Docker -v 挂载的“双刃剑”效应
挂载 WSL 目录到 Docker 可以省去重复拉取代码和重编环境的时间，但这打破了文件系统的隔离。A 侧的编译动作，会直接修改 B 侧看到的文件。在使用挂载目录时，必须明确当前的操作到底是在哪个“环境上下文”中执行。
💡 原则四：排错终极杀招 grep
当遇到“明明没写这个包，却报错找不到这个包”的玄学问题时，不要猜，直接用：
grep -r "包名" suspect_workspace/install/
顺藤摸瓜找 setup.sh 或 parent_prefix_path，百发百中。