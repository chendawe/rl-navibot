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
#!/bin/bash

# ========== 1. 路径推导 ==========
# 假设本脚本位于: .../rl-navibot/A_tries/burger_navi_in_house/scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRY_DIR="$(dirname "$SCRIPT_DIR")"                                  # burger_navi_in_house/
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"    # rl-navibot/

# ========== 2. 参数赋值 ==========
script_path="$PROJECT_ROOT/src/decision/decision/rl/useful_scripts/train.py"
algo_config="$TRY_DIR/config/transfer_learning.train.config.yaml"
# 💡 注意：这里用的是 decision/decision/rl 路径下的环境配置
env_config="$PROJECT_ROOT/src/decision/decision/rl/useful_env_configs/rl_env.house.config.yaml"
base_dir="$TRY_DIR"
model_prefix="SAC_world2house"

# ========== 3. 友好信息打印 ==========
echo "=================================================="
echo "🏋️ RL Navigation Trainer (Transfer Learning)"
echo "=================================================="
echo "📍 Python脚本 : $script_path"
echo "⚙️  算法配置   : $algo_config"
echo "🌍 环境配置   : $env_config"
echo "💾 存储根目录 : $base_dir"
echo "🏷️  模型前缀   : $model_prefix"
echo "=================================================="

# ========== 4. 关键文件预检 ==========
if [ ! -f "$script_path" ]; then
    echo "❌ 错误: 找不到 Python 脚本!"
    exit 1
fi
if [ ! -f "$algo_config" ]; then
    echo "❌ 错误: 找不到算法配置文件!"
    exit 1
fi
if [ ! -f "$env_config" ]; then
    echo "❌ 错误: 找不到环境配置文件!"
    exit 1
fi

# ========== 5. 启动执行 ==========
echo "⏳ 正在启动训练流程，请稍候...\n"

python "$script_path" \
    --algo_config "$algo_config" \
    --env_config "$env_config" \
    --base_dir "$base_dir" \
    --model_prefix "$model_prefix"

# ========== 6. 结束状态捕获 ==========
exit_code=$?
echo ""
echo "=================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ 迁移学习训练流程正常结束！"
else
    echo "❌ 训练流程异常退出 (错误码: $exit_code)"
fi
echo "=================================================="

exit $exit_code
