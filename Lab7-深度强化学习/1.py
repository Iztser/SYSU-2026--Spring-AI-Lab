import os
import gymnasium as gym
import random
import numpy as np
from collections import deque

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ==========================
# Q Network
# ==========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ==========================
# DQN Agent
# ==========================
class DQNAgent:
    def __init__(self, state_dim, action_dim):

        self.action_dim = action_dim

        self.gamma = 0.99
        self.lr = 1e-3

        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = QNetwork(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=self.lr
        )

        self.memory = ReplayBuffer(10000)

    # ε-greedy策略
    def select_action(self, state):

        # TODO 1:
        # 实现 ε-greedy 动作选择

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state)\
            .unsqueeze(0)\
            .to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax().item()

    # 网络训练
    def train(self):

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前Q值
        current_q = self.q_net(states)\
            .gather(1, actions)

        # TODO 2:
        # 根据Bellman方程计算目标Q值

        with torch.no_grad():

            next_q = self.target_net(next_states)\
                .max(1)[0]\
                .unsqueeze(1)

            target_q = rewards + \
                       self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 更新目标网络
    def update_target(self):

        # TODO 3:
        # 将在线网络参数复制到目标网络

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )


# ==========================
# 训练结果可视化
# ==========================
def plot_training_results(rewards, window=20, save_dir=None):
    """
    绘制并保存训练结果图像。

    参数:
        rewards (list[float]): 每轮的累计奖励
        window (int): 移动平均的窗口大小
        save_dir (str): 保存目录，默认为脚本所在目录
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)

    # 计算移动平均
    avg_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    avg_episodes = np.arange(window, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # 原始奖励 — 半透明折线，展示波动
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue",
            linewidth=0.8, label="Episode Reward")

    # 移动平均 — 实线，展示整体趋势
    ax.plot(avg_episodes, avg_rewards, color="darkorange",
            linewidth=2.0, label=f"Moving Average (w={window})")

    # 180 分解决阈值线
    ax.axhline(y=180, color="green", linestyle="--", linewidth=1.2,
               label="Solved Threshold (180)")

    # 最终平均标记
    if len(avg_rewards) > 0:
        final_avg = avg_rewards[-1]
        ax.axhline(y=final_avg, color="red", linestyle=":", linewidth=1.0,
                   alpha=0.7, label=f"Final Avg: {final_avg:.1f}")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("DQN on CartPole-v1 — Training Progress", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 在右下角显示统计信息
    textstr = (
        f"Episodes: {len(rewards)}\n"
        f"Max Reward: {rewards.max():.1f}\n"
        f"Final 20-avg: {rewards[-20:].mean():.1f}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
    ax.text(0.98, 0.04, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props)

    fig.tight_layout()

    save_path = os.path.join(save_dir, "training_rewards.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"\n📈 训练曲线已保存至: {save_path}")


# ==========================
# Main
# ==========================
def main():

    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim
    )

    episodes = 500

    rewards_history = []

    for episode in range(episodes):

        state, _ = env.reset()

        episode_reward = 0

        while True:

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.train()

            state = next_state

            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)

        # 每10轮同步一次目标网络
        if episode % 10 == 0:
            agent.update_target()

        avg_reward = np.mean(
            rewards_history[-20:]
        )

        print(
            f"Episode:{episode:4d} "
            f"Reward:{episode_reward:6.1f} "
            f"Avg:{avg_reward:6.1f}"
        )

        if avg_reward >= 300:
            print("\nSolved!")
            break

    env.close()

    # 绘制并保存训练曲线
    plot_training_results(rewards_history, window=20)


if __name__ == "__main__":
    main()