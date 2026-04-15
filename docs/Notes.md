
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
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
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