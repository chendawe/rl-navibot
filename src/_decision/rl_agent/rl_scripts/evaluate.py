from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import your_env

env = your_env.make_env()  # 你的原始环境
# env = VecNormalize.load("saved_models/SAC/vec_normalize.pkl", env)  # 如果有归一化
model = SAC.load("saved_models/SAC/sac_nav_model_200000_steps.zip", env=env)

# 在模拟环境中评估
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # 观察 reward 是否正常（比如大于 0）