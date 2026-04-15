import sys
import os
import re
import argparse
import numpy as np
import yaml
import rclpy
from pathlib import Path

# ========== 1. 路径与项目导入 ==========
sys.path.append("/home/chendawww/workspace/rl-navibot/src")
from decision.rl_agent.rl_agent.rl.env import TurtleBot3NavEnv, fetch_tb3_urdf
from decision.rl_agent.rl_agent.rl.algorithms import get_algorithm

# ========== 2. 导入算法与 Callback ==========
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ========== 3. 通用 Callback 定义 ==========
class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        self._current_rewards = {}
        self._current_successes = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i in range(len(dones)):
            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + rewards[i]
            if infos[i].get('goal_reached', False):
                self._current_successes[i] = True

            if dones[i]:
                self.episode_rewards.append(self._current_rewards[i])
                self.episode_successes.append(float(self._current_successes.get(i, False)))
                del self._current_rewards[i]
                self._current_successes.pop(i, None)

        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
            recent = self.episode_successes[-10:]
            avg_r = np.mean(self.episode_rewards[-10:])
            rate = np.mean(recent) * 100
            print(f"[Episodes: {len(self.episode_rewards)}] SR: {rate:.0f}% | Avg Reward: {avg_r:.1f}")
        return True

class GazeboShutdownCallback(BaseCallback):
    def __init__(self, container_name="ros2my", stop_container=False, verbose=0):
        super().__init__(verbose)
        self.container_name = container_name
        self.stop_container = stop_container

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.stop_container:
            os.system(f"docker stop {self.container_name}")
        else:
            os.system(f"docker exec {self.container_name} pkill -9 gzserver 2>/dev/null")
        print("[Shutdown] Gazebo terminated.")

# ========== 4. 核心逻辑：动态构建模型 ==========
def build_model(algo_name, env, algo_config, train_config, log_dir):
    device = train_config.get("device", "cuda")
    
    if algo_name == "SAC":
        params = algo_config["sac_params"]
        return SAC(
            "MlpPolicy", env, device=device,
            learning_rate=params.get("learning_rate", 3e-4),
            buffer_size=params.get("buffer_size", 100000),
            learning_starts=params.get("learning_starts", 1000),
            batch_size=params.get("batch_size", 256),
            tau=params.get("tau", 0.005),
            gamma=params.get("gamma", 0.99),
            ent_coef=params.get("ent_coef", "auto"),
            policy_kwargs=dict(net_arch=params.get("net_arch", [256, 256])),
            verbose=1,
            tensorboard_log=log_dir,
        )
    elif algo_name == "PPO":
        params = algo_config["ppo_params"]
        return PPO(
            "MlpPolicy", env, device=device,
            learning_rate=params.get("learning_rate", 3e-4),
            n_steps=params.get("n_steps", 2048),
            batch_size=params.get("batch_size", 64),
            n_epochs=params.get("n_epochs", 10),
            gamma=params.get("gamma", 0.99),
            gae_lambda=params.get("gae_lambda", 0.95),
            clip_range=params.get("clip_range", 0.2),
            ent_coef=params.get("ent_coef", 0.01),
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 0.5),
            policy_kwargs=dict(net_arch=dict(
                pi=params.get("net_arch_pi", [256, 256]),
                vf=params.get("net_arch_vf", [256, 256])
            )),
            verbose=1,
            tensorboard_log=log_dir,
        )
    else:
        raise ValueError(f"不支持的算法: {algo_name}")

# ========== 5. 断点续训查找逻辑 ==========
# 通过文件名找
# def find_latest_checkpoint(model_dir, prefix):
#     checkpoint_files = [f for f in model_dir.glob(f"{prefix}_*.zip") if "final" not in f.stem]
#     if not checkpoint_files: return None, 0
#     def extract_steps(filename):
#         match = re.search(r"_(\d+)_steps", filename.stem)
#         return int(match.group(1)) if match else 0
#     latest = max(checkpoint_files, key=extract_steps)
#     return latest, extract_steps(latest)

# 直接通过断点文件找
def find_latest_checkpoint(model_dir, prefix):
    checkpoint_files = [f for f in model_dir.glob(f"{prefix}_*.zip") if "final" not in f.stem]
    if not checkpoint_files:
        return None, 0
    # 按文件修改时间排序，取最新的
    latest = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
    return latest, 0  # 步数后面从模型里读


def main():
    # 1. 接收命令行参数 (新增 base_dir 和 model_prefix)
    parser = argparse.ArgumentParser(description="RL Navigation Trainer")
    parser.add_argument("--algo_config", type=str, required=True, help="算法 YAML 配置路径")
    parser.add_argument("--env_config", type=str, required=True, help="环境 YAML 配置路径")
    parser.add_argument("--base_dir", type=str, default=None, help="覆盖 YAML 中的基础存储路径")
    parser.add_argument("--model_prefix", type=str, default=None, help="覆盖 YAML 中的模型名前缀")
    args = parser.parse_args()

    # 2. 加载配置
    with open(args.algo_config, "r") as f:
        algo_config = yaml.safe_load(f)
    with open(args.env_config, "r") as f:
        env_config = yaml.safe_load(f)

    algo_name = algo_config["algorithm"]
    train_cfg = algo_config["training"]
    paths_cfg = algo_config.get("paths", {})
    
    # 🔥 核心逻辑：优先取命令行参数，没有则取 YAML，都没有则兜底
    BASE_DIR = Path(args.base_dir if args.base_dir else paths_cfg.get("base_dir", "."))
    MODEL_PREFIX = args.model_prefix if args.model_prefix else paths_cfg.get("model_prefix", algo_name.lower())

    # 🔥 严格构建存储树
    MODEL_DIR = BASE_DIR / "saved_models" / MODEL_PREFIX
    LOG_DIR = MODEL_DIR / "log"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    seed = train_cfg.get("seed", 42)
    
    # 打印确认，一眼看清存在哪
    print("="*50)
    print(f"Algorithm  : {algo_name.upper()}")
    print(f"World      : {env_config['world']['name']}")
    print(f"Bot        : {env_config['bot']['name']}")
    print(f"Seed       : {seed}")
    print(f"Base Dir   : {BASE_DIR}")
    print(f"Model Prefix: {MODEL_PREFIX}")
    print(f"Save Path  : {MODEL_DIR}")
    print(f"TB Log Path: {LOG_DIR}")
    print("="*50)

    # 3. 初始化环境
    rclpy.init()
    env = None
    my_robot_urdf = fetch_tb3_urdf()
    env = TurtleBot3NavEnv(robot_urdf=my_robot_urdf, env_config=env_config)

    # 4. 调用工厂函数，把 LOG_DIR 塞进去
    model = get_algorithm(
        algo_name=algo_name,
        env=env,
        config=algo_config,
        log_dir=str(LOG_DIR)
    )

    # 5. 断点续训 (根据 MODEL_PREFIX 找 zip)
    latest_ckpt, _ = find_latest_checkpoint(MODEL_DIR, MODEL_PREFIX)
    trained_steps = 0
    continue_steps = train_cfg["total_timesteps"]

    if latest_ckpt:
        # 用类方法 load，返回的是加载好数据的新模型对象
        algo_class = type(model)  # 拿到 SAC / PPO 等具体的算法类
        model = algo_class.load(latest_ckpt, env=env, reset_num_timesteps=False)
        model.set_random_seed(seed)

        model.tensorboard_log = str(LOG_DIR)   # 强制指定 log 路径
        model._logger = None                   # 杀掉可能存在的僵尸 logger
        
        trained_steps = model.num_timesteps
        continue_steps = train_cfg["total_timesteps"] - trained_steps
        
        if continue_steps > 0:
            print(f"\n[Resume] 找到断点: {latest_ckpt.name} (已训 {trained_steps} 步)")
            print(f"\n[Resume] 找到断点: {latest_ckpt.name}")
            print(f"  ├─ 算法: {algo_class.__name__}")
            print(f"  ├─ 已训步数: {trained_steps}")
            print(f"  ├─ 目标总步数: {train_cfg['total_timesteps']}")
            print(f"  ├─ 剩余需训练: {continue_steps} 步")
            print(f"  ├─ 已完成: {trained_steps / train_cfg['total_timesteps'] * 100:.1f}%")
            print(f"  ├─ 网络结构: {model.policy.net_arch}")
            print(f"  ├─ 学习率: {model.learning_rate}")
            print(f"  ├─ γ (discount): {model.gamma}")
            if hasattr(model, 'ent_coef'):
                print(f"  ├─ 熵系数: {model.ent_coef}")
            print(f"  └─ 缓冲区大小: {model.buffer_size}")
        else:
            print(f"\n[Done] 已达到目标步数 {train_cfg['total_timesteps']}，无需继续训练。退出。")
            env.close()
            rclpy.shutdown()
            return
    else:
        model.set_random_seed(seed)
        print(f"\n[New] 未找到断点，从头开始训练 {train_cfg['total_timesteps']} 步...")


    # 6. 训练
    try:
        callbacks = [
            CheckpointCallback(
                save_freq=train_cfg["save_freq"], 
                save_path=MODEL_DIR,         # 模型存这
                name_prefix=MODEL_PREFIX,    # 文件前缀用你指定的
                save_replay_buffer=True,
            ),
            SuccessRateCallback(),
        ]
        if train_cfg.get("auto_kill_gazebo"):
            callbacks.append(GazeboShutdownCallback(container_name=train_cfg.get("docker_container", "ros2my")))
            
        model.learn(
            total_timesteps=continue_steps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False
        )
    finally:
        model.save(MODEL_DIR / f"{MODEL_PREFIX}_final")
        print(f"\n[Save] 最终模型已保存至: {MODEL_DIR / f'{MODEL_PREFIX}_final.zip'}")
        env.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()