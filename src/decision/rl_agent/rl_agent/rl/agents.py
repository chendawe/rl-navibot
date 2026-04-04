import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from collections import deque
from abc import ABC, abstractmethod
from torch.distributions import Normal

# ==========================================
# 0. 基础组件 (所有算法共用)
# ==========================================
class ReplayBuffer:
    """Off-policy 专用：随机采样池"""
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(args)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(lambda x: torch.FloatTensor(np.array(x)).to('cpu'), zip(*batch))
    def __len__(self): return len(self.buffer)

class RolloutBuffer:
    """On-policy 专用：顺序存取轨迹"""
    def __init__(self): self.clear()
    def push(self, *args): 
        for i, arg in enumerate(args): self.buffers[i].append(arg)
    def clear(self): self.buffers = [[] for _ in range(5)] # [s, a, log_p, r, d]
    def get(self):
        return map(lambda x: torch.FloatTensor(np.array(x)), self.buffers)

# ==========================================
# 1. 第一层：基础抽象类 (模拟 BaseAlgorithm)
# ==========================================
class BaseAlgorithm(ABC):
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

    @abstractmethod
    def select_action(self, state):
        """根据状态选择动作"""
        pass

    @abstractmethod
    def _update(self, *args):
        """核心算法逻辑：计算 Loss 并反向传播"""
        pass

# ==========================================
# 2. 第二层：Off-Policy 抽象类 (模拟 OffPolicyAlgorithm)
# ==========================================
class OffPolicy(BaseAlgorithm):
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=64, learn_freq=4, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.learn_freq = learn_freq
        self.step_counter = 0

    def learn(self, env, total_steps):
        state, _ = env.reset()
        for step in range(1, total_steps + 1):
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # 1. 存入经验池
            self.buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            self.step_counter += 1
            
            # 2. 按频率更新 (Off-policy 特征)
            if self.step_counter % self.learn_freq == 0 and len(self.buffer) > self.batch_size:
                self._update(self.buffer.sample(self.batch_size))
            
            if done: state, _ = env.reset()

# ==========================================
# 3. 第二层：On-Policy 抽象类 (模拟 OnPolicyAlgorithm)
# ==========================================
class OnPolicy(BaseAlgorithm):
    def __init__(self, state_dim, action_dim, update_epochs=10, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.buffer = RolloutBuffer()
        self.update_epochs = update_epochs

    # def learn(self, env, total_episodes):
    #     for episode in range(1, total_episodes + 1):
    #         state, _ = env.reset()
    #         episode_reward = 0
            
    #         # 1. 收集一整条轨迹 (On-policy 特征)
    #         while True:
    #             action, log_prob = self.select_action(state) # On-policy 需返回 log_prob
    #             next_state, reward, done, truncated, _ = env.step(action)
    #             done = done or truncated
    #             self.buffer.push(state, action, log_prob, reward, float(done))
    #             state = next_state
    #             episode_reward += reward
    #             if done: break
            
    #         # 2. 用这批数据反复训练 N 次
    #         self._update(self.buffer.get(), self.update_epochs)
    #         self.buffer.clear() # 3. 训练完立刻丢弃
            
    #         print(f"Episode {episode} | Reward: {episode_reward:.2f}")
    def learn(self, env, total_episodes):
        for episode in range(1, total_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                # 【关键修改】统一接收三元组：环境动作、Buffer动作、对数概率
                env_action, buffer_action, log_prob = self.select_action(state)
                
                # 1. 环境只接收处理好的动作
                next_state, reward, done, truncated, _ = env.step(env_action)
                done = done or truncated
                
                # 2. Buffer 只存储原始的计算数据 (5个元素，完美匹配)
                self.buffer.push(state, buffer_action, log_prob, reward, float(done))
                
                state = next_state
                episode_reward += reward
                if done: break
            
            # 3. 更新网络
            self._update(self.buffer.get(), self.update_epochs)
            self.buffer.clear()
            
            print(f"Episode {episode} | Reward: {episode_reward:.2f}")

# ==========================================
# 4. 第三层：具体算法实现 (只关心网络和 Loss)
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class DDQN(OffPolicy):
    """DDQN 只需要写自己的网络和 _update"""
    # def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, **kwargs):
    #     super().__init__(state_dim, action_dim, **kwargs)
    #     self.gamma = gamma
    #     self.q_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)).to(self.device)
    #     self.target_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)).to(self.device)
    #     self.target_net.load_state_dict(self.q_net.state_dict())
    #     self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
    #     self.epsilon = 1.0
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.005, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.gamma, self.tau = gamma, tau
        self.q_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)).to(self.device)
        self.target_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.epsilon = 1.0


    def select_action(self, state):
        if random.random() < self.epsilon: return random.randrange(self.action_dim)
        with torch.no_grad(): return self.q_net(torch.FloatTensor(state)).argmax(0).item()

    def _update(self, sampled_batch):
        s, a, r, s_, d = [x.to(self.device) for x in sampled_batch]
        a = a.long()
        next_actions = self.q_net(s_).argmax(1)
        q_target = r + self.gamma * (1 - d) * self.target_net(s_).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1), q_target)
        
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        
        self.target_net.load_state_dict(self.q_net.state_dict()) # 简化版硬更新
        # # 【优化】改为软更新，取代粗暴的硬更新，训练更稳定
        # for p, pt in zip(self.q_net.parameters(), self.target_net.parameters()):
        #     pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            
        self.epsilon = max(0.01, self.epsilon * 0.995)


# class PPO(OnPolicy):
#     """PPO 只需要写自己的 Actor-Critic 和 _update"""
#     def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, **kwargs):
#         super().__init__(state_dim, action_dim, **kwargs)
#         self.gamma, self.max_action = gamma, max_action
#         self.lam, self.eps_clip = lam, eps_clip
        
#         # 拆分 Actor 和 Critic 方便操作
#         self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim * 2)).to(self.device)
#         self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)).to(self.device)
#         self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

#     def select_action(self, state):
#         s = torch.FloatTensor(state).to(self.device)
#         out = self.actor(s)
#         mean, log_std = out.chunk(2, dim=-1)
#         dist = Normal(mean, log_std.exp())
#         action = dist.rsample()
#         # 注意：这里存入 buffer 的 log_prob 是未经过 tanh 修正的（为了简化父类传递）
#         # 修正会在 _update 中重新计算
#         log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
#         return (torch.tanh(action) * self.max_action).cpu().item(), log_prob.cpu().item()

#     def _update(self, rollout_data, epochs):
#         # 1. 解包并转换数据到 Tensor
#         states, actions, old_log_probs, rewards, dones = rollout_data
#         states = torch.FloatTensor(np.array(states)).to(self.device)
#         actions = torch.FloatTensor(np.array(actions)).unsqueeze(1).to(self.device)
#         old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1).to(self.device)
#         rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
#         dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

#         # 2. 计算 GAE (广义优势估计)
#         with torch.no_grad():
#             values = self.critic(states)
#             # 拼接最后一步的 value (为0)
#             next_values = torch.cat([values[1:], torch.zeros(1, 1).to(self.device)])
            
#             deltas = rewards + self.gamma * next_values * (1 - dones) - values
#             advantages = []
#             gae = 0
#             for delta in reversed(deltas):
#                 gae = delta + self.gamma * self.lam * (1 - dones[0]) * gae # 简化处理 done
#                 advantages.insert(0, gae)
#             advantages = torch.FloatTensor(advantages).to(self.device)
#             returns = advantages + values
#             # 标准化优势
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         # 3. PPO 核心更新逻辑 (反复利用这批数据训练 epochs 次)
#         for _ in range(epochs):
#             out = self.actor(states)
#             mean, log_std = out.chunk(2, dim=-1)
#             dist = Normal(mean, log_std.exp())
            
#             # 重新采样并计算 带有 tanh 修正的 new_log_prob
#             action_sample = dist.rsample()
#             new_log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
#             # Tanh 压缩修正公式
#             new_log_prob -= torch.log(1 - torch.tanh(action_sample).pow(2) + 1e-8)
            
#             # 为了和 actions 对应，我们其实不需要 action_sample，直接用原来的 actions 算 log_prob 即可
#             # 这里直接用原 actions 评估概率：
#             new_log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
#             new_log_prob -= torch.log(1 - torch.tanh(actions).pow(2) + 1e-8)

#             # 计算 Ratio
#             ratio = torch.exp(new_log_prob - old_log_probs)

#             # 截断损失
#             surr1 = ratio * advantages
#             surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#             actor_loss = -torch.min(surr1, surr2).mean()

#             # 价值损失
#             new_values = self.critic(states)
#             critic_loss = F.mse_loss(new_values, returns)

#             # 总损失反向传播
#             loss = actor_loss + 0.5 * critic_loss
#             self.optimizer.zero_grad()
#             loss.backward()
#             # 可选：梯度裁剪 (SB3 默认做了)
#             nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=0.5)
#             self.optimizer.step()

class PPO(OnPolicy):
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.gamma, self.max_action = gamma, max_action
        self.lam, self.eps_clip = lam, eps_clip
        
        self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim * 2)).to(self.device)
        self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    # def select_action(self, state):
    #     s = torch.FloatTensor(state).to(self.device)
    #     out = self.actor(s)
    #     mean, log_std = out.chunk(2, dim=-1)
    #     dist = Normal(mean, log_std.exp())
        
    #     # 1. 采样原始动作 (未经过 tanh)
    #     raw_action = dist.rsample()
    #     # 2. 计算 raw_action 的 log_prob (无需修正)
    #     log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        
    #     # 3. 压缩动作给环境执行
    #     tanh_action = torch.tanh(raw_action) * self.max_action
        
    #     # 【修复】返回 tanh_action 给环境，返回 raw_action 和 log_prob 存入 buffer
    #     return tanh_action.cpu().item(), (raw_action.cpu().item(), log_prob.cpu().item())

    def select_action(self, state):
        s = torch.FloatTensor(state).to(self.device)
        out = self.actor(s)
        mean, log_std = out.chunk(2, dim=-1)
        dist = Normal(mean, log_std.exp())
        
        raw_action = dist.rsample()
        
        # 【关键修复】在存入 Buffer 前，直接算出带 tanh 修正的 log_prob
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob_corrected = log_prob_raw - torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-8)
        
        env_action = torch.tanh(raw_action) * self.max_action
        
        # 存入的是修正后的 log_prob_corrected
        return env_action.cpu().item(), raw_action.cpu().item(), log_prob_corrected.cpu().item()

    def _update(self, rollout_data, epochs):
        # 【优化】去除冗余的 np.array 和 unsqueeze，直接转 tensor
        states, raw_actions, old_log_probs, rewards, dones = rollout_data
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        raw_actions = torch.tensor(raw_actions, dtype=torch.float32).unsqueeze(1).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            values = self.critic(states)
            next_values = torch.cat([values[1:], torch.zeros(1, 1).to(self.device)])
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            # 【修复】GAE 计算：使用当前步 t 的 done，而不是 dones[0]
            advantages = []
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
                advantages.insert(0, gae)
            advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            out = self.actor(states)
            mean, log_std = out.chunk(2, dim=-1)
            dist = Normal(mean, log_std.exp())
            
            # 【修复】直接用当前分布评估 buffer 里的 raw_actions，然后加上 tanh 修正项
            new_log_prob = dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
            new_log_prob -= torch.log(1 - torch.tanh(raw_actions).pow(2) + 1e-8)

            ratio = torch.exp(new_log_prob - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(self.critic(states), returns)
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=0.5)
            self.optimizer.step()


# ==========================================
# 补全：SAC 完整实现
# ==========================================
class SAC(OffPolicy):
    """SAC 继承 OffPolicy，处理连续空间"""
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.max_action, self.gamma, self.tau = max_action, gamma, tau

        # 自动熵调节参数
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        ).to(self.device)
        self.mean_head = nn.Linear(256, action_dim).to(self.device)
        self.log_std_head = nn.Linear(256, action_dim).to(self.device)

        # 双 Q 网络 (Critic)
        self.q1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.q2 = self._build_critic(state_dim, action_dim).to(self.device)
        
        # 目标 Q 网络
        self.q1_target = self._build_critic(state_dim, action_dim).to(self.device)
        self.q2_target = self._build_critic(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # 优化器
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.actor_opt = optim.Adam(list(self.actor.parameters()) + list(self.mean_head.parameters()) + list(self.log_std_head.parameters()), lr=lr)

    def _build_critic(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    # def select_action(self, state):
    #     # Off-policy 的 select_action 只和环境交互，不需要返回 log_prob
    #     s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     feat = self.actor(s)
    #     mean = self.mean_head(feat)
    #     log_std = self.log_std_head(feat).clamp(-20, 2)
        
    #     # 推理时直接输出均值 (不加噪声) 以获得确定性动作，或者加一点噪声
    #     # 这里加一点噪声模拟探索
    #     std = log_std.exp()
    #     dist = Normal(mean, std)
    #     action = dist.rsample()
    #     return (torch.tanh(action) * self.max_action).cpu().numpy()[0]

    # def select_action(self, state):
    #     s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     feat = self.actor(s)
    #     # 【优化】推理时直接取均值，不加噪声。SAC的探索完全由经验池和历史随机性保证
    #     mean = self.mean_head(feat)
    #     return (torch.tanh(mean) * self.max_action).cpu().numpy()[0]

    def select_action(self, state):
        """训练时调用：必须带噪声探索"""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        feat = self.actor(s)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(-20, 2)
        
        # 【关键修复】训练时，使用 rsample 引入高斯噪声
        dist = Normal(mean, log_std.exp())
        raw_action = dist.rsample()
        action = torch.tanh(raw_action) * self.max_action
        
        return action.cpu().numpy()[0]

    def predict(self, state):
        """测试/评估时调用：输出确定性均值动作"""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.actor(s)
            mean = self.mean_head(feat)
            return (torch.tanh(mean) * self.max_action).cpu().numpy()[0]

    def _update(self, sampled_batch):
        # 1. 解包 Off-policy 采样的数据
        s, a, r, s_, d = [x.to(self.device) for x in sampled_batch]
        r = r.unsqueeze(1)
        d = d.unsqueeze(1)
        a = a.unsqueeze(1) if a.dim() == 1 else a # 确保维度正确

        # # ----------------- 2. 更新 Q 网络 -----------------
        # with torch.no_grad():
        #     # 算下一步的动作和 log_prob (带 tanh 修正)
        #     feat_ = self.actor(s_)
        #     mean_, log_std_ = self.mean_head(feat_), self.log_std_head(feat_).clamp(-20, 2)
        #     dist_ = Normal(mean_, log_std_.exp())
        #     a_ = dist_.rsample()
        #     log_p_ = dist_.log_prob(a_).sum(dim=-1, keepdim=True)
        #     log_p_ -= torch.log(1 - torch.tanh(a_).pow(2) + 1e-8)
        #     a_ = torch.tanh(a_) * self.max_action

        #     # 取双 Q 的最小值 (防止过估计)
        #     q1_target_val = self.q1_target(torch.cat([s_, a_], dim=-1))
        #     q2_target_val = self.q2_target(torch.cat([s_, a_], dim=-1))
        #     min_q_target = torch.min(q1_target_val, q2_target_val)
            
        #     # SAC 核心：目标 Q = R + γ * (Q_min - α * log_π)
        #     q_target = r + self.gamma * (1 - d) * (min_q_target - self.alpha * log_p_)

        # q1_val = self.q1(torch.cat([s, a], dim=-1))
        # q2_val = self.q2(torch.cat([s, a], dim=-1))
        # q_loss = F.mse_loss(q1_val, q_target) + F.mse_loss(q2_val, q_target)

        # self.q_opt.zero_grad()
        # q_loss.backward()
        # self.q_opt.step()

        # # ----------------- 3. 更新 Actor 网络 -----------------
        # feat = self.actor(s)
        # mean = self.mean_head(feat)
        # log_std = self.log_std_head(feat).clamp(-20, 2)
        # dist = Normal(mean, log_std.exp())
        # a_new = dist.rsample()
        # log_p = dist.log_prob(a_new).sum(dim=-1, keepdim=True)
        # log_p -= torch.log(1 - torch.tanh(a_new).pow(2) + 1e-8)
        # a_new = torch.tanh(a_new) * self.max_action

        # q1_new = self.q1(torch.cat([s, a_new], dim=-1))
        # q2_new = self.q2(torch.cat([s, a_new], dim=-1))
        # min_q_new = torch.min(q1_new, q2_new)
        
        # # SAC 核心：Actor 目标是最大化 Q - α * log_π
        # actor_loss = (self.alpha * log_p - min_q_new).mean()

        # self.actor_opt.zero_grad()
        # actor_loss.backward()
        # self.actor_opt.step()
        
       # ----------------- 2. 更新 Q 网络 -----------------
        with torch.no_grad():
            feat_ = self.actor(s_)
            mean_, log_std_ = self.mean_head(feat_), self.log_std_head(feat_).clamp(-20, 2)
            dist_ = Normal(mean_, log_std_.exp())
            
            # 注意：_update 内部算目标 Q 时，仍然需要 rsample 加噪声
            a_ = dist_.rsample()
            log_p_ = dist_.log_prob(a_).sum(dim=-1, keepdim=True)
            log_p_ -= torch.log(1 - torch.tanh(a_).pow(2) + 1e-8)
            a_ = torch.tanh(a_) * self.max_action

            q1_target_val = self.q1_target(torch.cat([s_, a_], dim=-1))
            q2_target_val = self.q2_target(torch.cat([s_, a_], dim=-1))
            min_q_target = torch.min(q1_target_val, q2_target_val)
            q_target = r + self.gamma * (1 - d) * (min_q_target - self.alpha * log_p_)

        q1_val = self.q1(torch.cat([s, a], dim=-1))
        q2_val = self.q2(torch.cat([s, a], dim=-1))
        q_loss = F.mse_loss(q1_val, q_target) + F.mse_loss(q2_val, q_target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # ----------------- 3. 更新 Actor 网络 -----------------
        feat = self.actor(s)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(-20, 2)
        dist = Normal(mean, log_std.exp())
        
        # 同理，Actor 更新时需要重参数化求梯度
        a_new = dist.rsample()
        log_p = dist.log_prob(a_new).sum(dim=-1, keepdim=True)
        log_p -= torch.log(1 - torch.tanh(a_new).pow(2) + 1e-8)
        a_new = torch.tanh(a_new) * self.max_action

        q1_new = self.q1(torch.cat([s, a_new], dim=-1))
        q2_new = self.q2(torch.cat([s, a_new], dim=-1))
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_p - min_q_new).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()


        # ----------------- 4. 更新温度系数 α -----------------
        # 目标是让 log_π 逼近目标熵
        alpha_loss = -(self.log_alpha.exp() * (log_p.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ----------------- 5. 软更新目标网络 -----------------
        # θ_target = τ * θ + (1 - τ) * θ_target
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)


# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    # 离散环境跑 DDQN
    env_discrete = gym.make('CartPole-v1')
    model = DDQN(state_dim=4, action_dim=2, device='cpu')
    model.learn(env_discrete, total_steps=10000)

    # 连续环境跑 PPO
    env_cont = gym.make('Pendulum-v1')
    # model_ppo = PPO(state_dim=3, action_dim=1, max_action=2.0, device='cpu')
    # model_ppo.learn(env_cont, total_episodes=100)
