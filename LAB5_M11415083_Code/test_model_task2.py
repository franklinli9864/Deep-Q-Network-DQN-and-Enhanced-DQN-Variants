import argparse
import random
from collections import deque

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn


gym.register_envs(ale_py)


class AtariPreprocessor:
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


class DQN(nn.Module):
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
        x = x.float() / 255.0
        return self.network(x)


def evaluate(model_path, episodes=20, seed_start=0, render=False):
    render_mode = "human" if render else "rgb_array"

    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Number of actions:", num_actions)
    print("Model path:", model_path)

    q_net = DQN(num_actions).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    preprocessor = AtariPreprocessor(frame_stack=4)

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        state = preprocessor.reset(obs)

        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.uint8,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                action = q_net(state_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = preprocessor.step(next_obs)
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)

        print(
            f"Seed {seed_start + ep:02d}: "
            f"Reward = {total_reward:.2f}, "
            f"Steps = {step_count}"
        )

    env.close()

    rewards = np.array(rewards, dtype=np.float32)

    print("-" * 50)
    print(f"Mean reward over {episodes} episodes: {rewards.mean():.2f}")
    print(f"Std reward: {rewards.std():.2f}")
    print(f"Min reward: {rewards.min():.2f}")
    print(f"Max reward: {rewards.max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="./results/task2_1M/best_model.pt"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed_start)
    np.random.seed(args.seed_start)
    torch.manual_seed(args.seed_start)

    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        seed_start=args.seed_start,
        render=args.render
    )