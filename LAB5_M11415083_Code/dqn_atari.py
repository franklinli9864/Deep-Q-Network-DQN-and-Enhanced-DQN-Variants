# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Task 2: Vanilla DQN on Atari Pong-v5

import os
import random
import argparse
from collections import deque

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


gym.register_envs(ale_py)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class AtariPreprocessor:
    """
    Preprocess Atari RGB frames:
    1. RGB -> grayscale
    2. resize to 84 x 84
    3. stack last 4 frames
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)],
            maxlen=self.frame_stack
        )
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class ReplayBuffer:
    """
    Uniform replay buffer for vanilla DQN.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.uint8)
        next_states = np.array(next_states, dtype=np.uint8)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    CNN Q-network for Atari Pong.

    Input shape:
        batch x 4 x 84 x 84

    Output:
        Q-values for each action
    """

    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),

            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # x is uint8 image tensor in [0, 255]
        x = x.float() / 255.0
        return self.network(x)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.args = args

        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")

        self.num_actions = self.env.action_space.n
        print("Number of actions:", self.num_actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)
        self.test_preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        
        if args.load_model is not None and os.path.exists(args.load_model):
            print(f"Loading model from: {args.load_model}")
            self.q_net.load_state_dict(torch.load(args.load_model, map_location=self.device))
        else:
            print("Training from scratch.")

        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = ReplayBuffer(args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor

        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = args.start_env_steps
        self.train_count = 0

        # Pong score range is usually -21 to 21.
        self.best_reward = -21.0

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(
            state,
            dtype=torch.uint8,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def train(self):
        if len(self.memory) < self.args.replay_start_size:
            return

        if len(self.memory) < self.batch_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.train_count += 1

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.uint8, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.train_count % self.args.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % self.args.log_interval == 0:
            print(
                f"[Train] Env Steps: {self.env_count} | "
                f"Updates: {self.train_count} | "
                f"Loss: {loss.item():.4f} | "
                f"Q mean: {q_values.mean().item():.4f} | "
                f"Epsilon: {self.epsilon:.4f}"
            )

            wandb.log(
                {
                    "Loss": loss.item(),
                    "Q Mean": q_values.mean().item(),
                    "Q Max": q_values.max().item(),
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                }
            )

    def run(self):
        episode = 0

        while self.env_count < self.args.total_steps:
            obs, _ = self.env.reset(seed=self.args.seed + episode)
            state = self.preprocessor.reset(obs)

            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done and self.env_count < self.args.total_steps:
                action = self.select_action(state)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)

                self.memory.add(
                    state,
                    action,
                    reward,
                    next_state,
                    float(done)
                )

                for _ in range(self.args.train_per_step):
                    self.train()

                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.env_count += 1

                if self.env_count % self.args.eval_interval_steps == 0:
                    eval_reward = self.evaluate(num_episodes=self.args.eval_episodes)

                    print(
                        f"[Eval] Env Steps: {self.env_count} | "
                        f"Average Reward: {eval_reward:.2f} | "
                        f"Best Reward: {self.best_reward:.2f}"
                    )

                    wandb.log(
                        {
                            "Eval Reward": eval_reward,
                            "Best Reward": self.best_reward,
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                        }
                    )

                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        model_path = os.path.join(self.save_dir, "best_model.pt")
                        torch.save(self.q_net.state_dict(), model_path)

                        print(
                            f"Saved new best model to {model_path} "
                            f"with average reward {eval_reward:.2f}"
                        )

                if self.env_count % self.args.save_interval_steps == 0:
                    model_path = os.path.join(
                        self.save_dir,
                        f"model_step{self.env_count}.pt"
                    )
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved checkpoint to {model_path}")

            print(
                f"[Episode] Ep: {episode:4d} | "
                f"Reward: {episode_reward:6.2f} | "
                f"Episode Steps: {episode_steps:5d} | "
                f"Env Steps: {self.env_count:7d} | "
                f"Updates: {self.train_count:7d} | "
                f"Epsilon: {self.epsilon:.4f}"
            )

            wandb.log(
                {
                    "Episode": episode,
                    "Train Episode Reward": episode_reward,
                    "Train Episode Steps": episode_steps,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                }
            )

            episode += 1

        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        torch.save(self.q_net.state_dict(), final_model_path)

        print(f"Saved final model to {final_model_path}")
        print(f"Best evaluation reward: {self.best_reward:.2f}")
        print(f"Best model path: {os.path.join(self.save_dir, 'best_model.pt')}")

    def evaluate(self, num_episodes=1):
        rewards = []

        for ep in range(num_episodes):
            obs, _ = self.test_env.reset(seed=ep)
            state = self.test_preprocessor.reset(obs)

            done = False
            total_reward = 0.0

            while not done:
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.uint8,
                    device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax(dim=1).item()

                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated

                state = self.test_preprocessor.step(next_obs)
                total_reward += reward

            rewards.append(total_reward)

        return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")

    parser.add_argument("--save-dir", type=str, default="./results/task2_3M_finetune")
    parser.add_argument("--wandb-run-name", type=str, default="task2_pong_3M_finetune")


    # 一開始只跑 100 steps，確認程式能不能動。
    parser.add_argument("--total-steps", type=int, default=3000000)
    parser.add_argument("--start-env-steps", type=int, default=1000000)
    
    parser.add_argument("--load-model", type=str, default="./results/task2_1M/best_model.pt")
    
    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)

    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.02)

    # 測試用先設小一點，不然 100 steps 內不會 train。
    
    parser.add_argument("--replay-start-size", type=int, default=10000)

    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--train-per-step", type=int, default=1)

    parser.add_argument("--eval-interval-steps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--save-interval-steps", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(
        project="DLP-Lab5-DQN-Pong",
        name=args.wandb_run_name,
        config=vars(args),
        save_code=True,
        mode="offline",
    )

    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run()

    wandb.finish()


if __name__ == "__main__":
    main()