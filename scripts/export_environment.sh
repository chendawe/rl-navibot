#!/bin/bash

echo "==================================="
echo "  正在导出 Conda 环境..."
echo "==================================="

cd /home/chendawww/workspace/rl-navibot

# 导出完整 conda 环境
conda env export > environment.yml

# 导出 pip 依赖
pip freeze > requirements.txt

# 导出 conda 显式列表（跨平台兼容）
conda list --explicit > conda_spec.txt

echo "✅ 导出完成！生成以下文件："
echo "  - environment.yml"
echo "  - requirements.txt"
echo "  - conda_spec.txt"