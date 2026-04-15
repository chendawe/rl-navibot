#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# # ========== 1. 路径与项目导入 ==========
# # 🔥 动态推导：从当前文件往上找三级到达 src 目录
# _SRC_DIR = Path(__file__).resolve().parent.parent
# if str(_SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(_SRC_DIR))

from decision.rl.environments import TurtleBot3NavEnv, fetch_tb3_urdf
# 🔥 导入你的算法工厂模块，兼顾深度模型加载和规则基线实例化
from decision.rl.algorithms import RuleBasedModel
from stable_baselines3 import SAC, PPO

ALGO_MAP = {
    "sac": SAC,
    "ppo": PPO,
}

# ========== 2. 绘图子函数 ==========
def plot_evaluation_results(results, save_dir: Path, model_name: str):
    """根据 run_episodes 返回的 results 生成评估图表"""
    details = results["details"]
    num_episodes = len(details)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # 图 1：每局总奖励
    rewards = [d["reward"] for d in details]
    mean_reward = results["mean_reward"]
    axs[0].plot(rewards, color='blue', marker='.', linestyle='-', linewidth=1, markersize=4)
    axs[0].axhline(y=mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    axs[0].set_title(f"[{model_name}] Eval: Episode Reward (Avg: {mean_reward:.2f})")
    axs[0].set_ylabel("Total Reward")
    axs[0].grid(True, alpha=0.3); axs[0].legend()

    # 图 2：每局成功/失败
    successes = [1.0 if d["success"] else 0.0 for d in details]
    success_rate = results["success_rate"]
    axs[1].bar(range(num_episodes), successes, color=['green' if s else 'red' for s in successes])
    axs[1].axhline(y=success_rate/100, color='blue', linestyle='--', label=f'Rate: {success_rate:.1f}%')
    axs[1].set_title(f"[{model_name}] Eval: Success ({int(success_rate * num_episodes / 100)}/{num_episodes} = {success_rate:.1f}%)")
    axs[1].set_ylabel("Success (1/0)")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].grid(True, alpha=0.3); axs[1].legend()

    # 图 3：每局步数
    steps = [d["steps"] for d in details]
    mean_steps = results["mean_steps"]
    axs[2].plot(steps, color='purple', marker='.', linestyle='-', linewidth=1, markersize=4)
    axs[2].axhline(y=mean_steps, color='orange', linestyle='--', label=f'Mean: {mean_steps:.1f}')
    axs[2].set_title(f"[{model_name}] Eval: Steps per Episode (Avg: {mean_steps:.1f})")
    axs[2].set_xlabel("Episode"); axs[2].set_ylabel("Steps")
    axs[2].grid(True, alpha=0.3); axs[2].legend()

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"enjoy__{model_name}__results.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[Plot] 评估图表已保存至: {plot_path}")


# ========== 3. 核心主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="RL Navigation Evaluator (Enjoy)")
    parser.add_argument("--model_path", type=str, default=None, help="要测试的 .zip 模型绝对路径 (rule_baseline时可忽略)")
    parser.add_argument("--algo_config", type=str, required=True, help="算法 YAML 配置路径")
    parser.add_argument("--env_config", type=str, required=True, help="环境 YAML 配置路径")
    parser.add_argument("--episodes", type=int, default=50, help="测试局数")
    parser.add_argument("--verbose", action="store_true", help="是否打印每一帧的 obs 和 action (调试用，默认关闭)")
    parser.add_argument("--no_plot", action="store_true", help="禁止生成评估图表")
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.algo_config, "r") as f:
        algo_config = yaml.safe_load(f)
    with open(args.env_config, "r") as f:
        env_config = yaml.safe_load(f)

    algo_name = algo_config["algorithm"].lower()

    # 2. 初始化环境
    env = None
    try:
        my_robot_urdf = fetch_tb3_urdf()
        env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, env_config=env_config)

        # ========== 3. 模型加载分发 (核心修改) ==========
        if algo_name == "rule_baseline":
            print("=" * 50)
            print(f"Mode       : Evaluation (Rule Baseline)")
            print(f"World      : {env_config['world']['name']}")
            print(f"Bot        : {env_config['bot']['name']}")
            print("=" * 50)
            
            # 💡 兼容性处理：你的 RuleBasedPolicy 里读的是 config["environment"]，
            # 但重构后 env_config 已经拆成了 bot/world。这里临时拼一个给它吃。
            rule_cfg = {"environment": {**env_config.get("bot", {}), **env_config.get("world", {})}}
            model = RuleBasedModel(config=rule_cfg, env=env)
            model_file = Path("rule_baseline") # 伪路径，用于后续日志打印
            
        else:
            if algo_name not in ALGO_MAP:
                raise ValueError(f"不支持的算法: {algo_name}，支持: {list(ALGO_MAP.keys())} 或 rule_baseline")
            
            model_file = Path(args.model_path)
            if not model_file.exists():
                print(f"[Error] 找不到模型文件: {model_file}")
                return

            print("=" * 50)
            print(f"Mode       : Evaluation (Enjoy)")
            print(f"Algorithm  : {algo_name.upper()}")
            print(f"World      : {env_config['world']['name']}")
            print(f"Bot        : {env_config['bot']['name']}")
            print(f"Model      : {model_file.name}")
            print(f"Episodes   : {args.episodes}")
            print(f"Verbose    : {args.verbose}")
            print("=" * 50)

            print(f"\n[Load] 正在加载模型...")
            model = ALGO_MAP[algo_name].load(args.model_path, env=env, device="auto")
            print(f"[Load] 模型加载成功，训练步数: {model.num_timesteps}")

        # 4. 评估 (完全统一的接口)
        print(f"\n[Eval] 开始评估 ({args.episodes} episodes)...")
        results = env.run_episodes(
            model=model,
            num_episodes=args.episodes,
            deterministic=True,
            verbose=args.verbose
        )

        # 5. 汇总
        details = results["details"]
        print("\n" + "=" * 50)
        print("=== 评估汇总 ===")
        print(f"总测试局数 : {results['num_episodes']}")
        print(f"成功局数   : {int(sum(1 for d in details if d['success']))}/{results['num_episodes']}")
        print(f"成功率     : {results['success_rate']:.1f}%")
        print(f"平均总奖励 : {results['mean_reward']:.2f}")
        print(f"平均步数   : {results['mean_steps']:.1f}")
        print("=" * 50)

        # 6. 绘图 (规则基线不画图，因为没保存路径意义不大)
        if not args.no_plot and algo_name != "rule_baseline":
            plot_dir = model_file.parent / "plots"
            plot_evaluation_results(results, plot_dir, model_file.stem)

    finally:
        if env is not None:
            env.close()
        print("[Env] 资源已释放。")

if __name__ == "__main__":
    main()
