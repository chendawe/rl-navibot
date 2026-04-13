#!/bin/bash

# docker 内：
# source /opt/ros/humble/setup.sh && \
# cd ~/workspace/rl-navibot && \
# source install/setup.sh && \
# cd ~/workspace/turtlebot3_ws && \
# source install/setup.sh && \
# cd ~/workspace/nitrobot_ws && \
# source install/setup.sh && \
# source ~/.bashrc

# export TURTLEBOT3_MODEL=burger
# ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py

# conda activate ros2
python /home/chendawww/workspace/rl-navibot/src/decision/rl_agent/rl_scripts/train.py \
    --algo_config /home/chendawww/workspace/rl-navibot/tries/burger_navi_in_house/config/transfer_learning.train.config.yaml \
    --env_config /home/chendawww/workspace/rl-navibot/src/decision/rl_agent/config/rl_env.house.config.yaml \
    --base_dir /home/chendawww/workspace/rl-navibot/tries/burger_navi_in_house \
    --model_prefix SAC_world2house