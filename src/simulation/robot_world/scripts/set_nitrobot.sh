#!/bin/bash

WORKSPACE_DIR="~/workspace"
echo "workspace位置：$WORKSPACE_DIR"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "脚本位置：$SCRIPT_DIR"

mkdir -p "$WORKSPACE_DIR/nitrobot_ws/src"

cd "$WORKSPACE_DIR/nitrobot_ws/src"
git clone https://github.com/PAyush15/nitrobot-sim.git
    
cd "$WORKSPACE_DIR/nitrobot_ws"



# sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -y
# rosdepc init
# rosdepc update
# rosdepc install --from-paths src --ignore-src -y
colcon build

source $WORKSPACE_DIR/nitrobot_ws/install/setup.bash

ros2 control list_controllers
ros2 launch nitrobot launch_sim.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard


