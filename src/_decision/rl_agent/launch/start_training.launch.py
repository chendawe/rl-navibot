import gym
from ..rl_agent.rl.algorithms import get_algorithm
from ..rl_agent.rl.env import MyGymEnv
import yaml

# 1. 加载配置
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. 实例化环境
env = MyGymEnv(config=config)

# 3. 从配置文件读取算法名称 (比如 config 里写了 algo: "rule_baseline")
#    你可以随时改成 "ppo" 或 "sac"，下面的代码一行都不用改！
ALGO_NAME = config.get("train", {}).get("algorithm", "rule_baseline")

# 4. 获取算法实例 (不管是规则还是神经网络，现在它们是同一种东西)
model = get_algorithm(ALGO_NAME, env, config)

# 5. 统一训练/测试流程
# 如果是 rule_baseline，这里的 learn 只是跑一遍数据收集；
# 如果是 ppo/sac，这里就是真正的梯度更新训练。
model.learn(total_timesteps=10000)

# 6. 保存 (虽然 rule_baseline 保存没有意义，但接口兼容，不会报错)
model.save(f"models/{ALGO_NAME}_model")
