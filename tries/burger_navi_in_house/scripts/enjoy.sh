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
# ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

python /home/chendawww/workspace/rl-navibot/src/decision/rl_agent/rl_scripts/enjoy.py \
    --model_path /home/chendawww/workspace/rl-navibot/tries/burger_navi_in_house/saved_models/SAC_world2house/SAC_world2house_291000_steps.zip \
    --algo_config /home/chendawww/workspace/rl-navibot/tries/burger_navi_in_house/config/transfer_learning.train.config.yaml \
    --env_config /home/chendawww/workspace/rl-navibot/src/decision/rl_agent/config/rl_env.house.config.yaml \
    --episodes 10 --verbose