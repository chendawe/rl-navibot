#!/usr/bin/env python3
"""
最小化测试脚本：验证 Gazebo 的 /reset_world 服务是否正常响应。
使用方式：直接运行 python test_reset_world.py
若超时，请检查 Gazebo 是否已启动且服务可见（ros2 service list | grep reset_world）。
"""

import sys
import time
from pathlib import Path

# 确保项目模块可导入（根据你的实际路径调整）
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.ros2.master import Ros2Runtime
from core.ros2.simulators.gazebo import GazeboSimulator


def main():
    runtime = None
    sim = None

    try:
        print("[1] 初始化 Ros2Runtime ...")
        runtime = Ros2Runtime()
        print("     ✅ Runtime 已启动，后台线程运行中。")

        print("[2] 创建 GazeboSimulator 实例 ...")
        sim = GazeboSimulator(runtime, node_name="test_reset_sim")
        print("     ✅ Simulator 已创建，等待服务上线 ...")

        # 设置仿真服务名称（与你的环境配置保持一致）
        sim.setup(
            reset_world="/reset_world",
            set_entity="/set_entity_state",
            delete_entity="/delete_entity",
            spawn_entity="/spawn_entity",
            entity_name="turtlebot3_burger"  # 根据实际模型名修改
        )

        print("[3] 等待服务就绪 (timeout=5.0s) ...")
        sim.wait_for_services(timeout_sec=5.0)
        print("     ✅ 服务已就绪。")

        print("[4] 调用 reset_world() ...")
        start = time.time()
        sim.reset_world()  # 如果超时，会抛出 TimeoutError
        elapsed = time.time() - start
        print(f"     ✅ reset_world 成功！耗时 {elapsed:.3f} 秒。")

        print("\n🎉 测试通过：reset_world 服务正常工作。")

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if sim is not None:
            # GazeboSimulator 目前没有显式的 destroy，由 runtime 统一管理
            pass
        if runtime is not None:
            print("[5] 关闭 Runtime ...")
            runtime.shutdown()
            print("     ✅ Runtime 已关闭。")


if __name__ == "__main__":
    main()