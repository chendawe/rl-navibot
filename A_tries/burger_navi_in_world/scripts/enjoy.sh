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

# export FASTRTPS_DEFAULT_PROFILES_FILE=/home/chendawww/workspace/rl-navibot/scripts/fastdds_no_shm.xml
# export RMW_FASTRTPS_USE_QOS_FROM_XML=0

#!/bin/bash

#!/bin/bash

# ========== 1. 路径推导 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRY_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# ========== 2. 参数赋值 ==========
script_path="$PROJECT_ROOT/src/decision/decision/rl/useful_scripts/enjoy.py"
model_path="$TRY_DIR/saved_models/SAC/sac_nav_model_136000_steps.zip"
algo_config="$TRY_DIR/configs/train.config.yaml"
env_config="$PROJECT_ROOT/src/decision/decision/rl/useful_env_configs/rl_env.world.config.yaml"
episodes=10

# ========== 3. 友好信息打印 ==========
echo "=================================================="
echo "🚀 RL Navigation Evaluator (Enjoy)"
echo "=================================================="
echo "📍 Python脚本 : $script_path"
echo "🤖 模型文件   : $(basename "$model_path")"
echo "   完整路径   : $model_path"
echo "⚙️  算法配置   : $algo_config"
echo "🌍 环境配置   : $env_config"
echo "🎯 测试局数   : $episodes"
echo "🔍 Verbose    : ON"
echo "=================================================="

# ========== 4. 关键文件预检 ==========
if [ ! -f "$script_path" ]; then
    echo "❌ 错误: 找不到 Python 脚本!"
    exit 1
fi

if [ ! -f "$model_path" ]; then
    echo "❌ 错误: 找不到模型权重文件!"
    exit 1
fi

# ========== 5. 启动执行 ==========
echo "⏳ 正在启动评估环境，请稍候...\n"

python "$script_path" \
    --model_path "$model_path" \
    --algo_config "$algo_config" \
    --env_config "$env_config" \
    --episodes "$episodes" \
    --verbose

# ========== 6. 结束状态捕获 ==========
exit_code=$?
echo ""
echo "=================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ 评估流程正常结束！"
else
    echo "❌ 评估流程异常退出 (错误码: $exit_code)"
fi
echo "=================================================="

exit $exit_code

