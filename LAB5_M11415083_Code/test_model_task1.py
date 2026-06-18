import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model_path, episodes=20, seed_start=0):
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = DQN(state_dim, num_actions).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                action = q_net(state_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_obs
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    print("Evaluation rewards:")
    for i, r in enumerate(rewards):
        print(f"Seed {seed_start + i:02d}: {r:.2f}")

    print("-" * 40)
    print(f"Mean reward over {episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./results/task1/best_model.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed_start)
    np.random.seed(args.seed_start)
    torch.manual_seed(args.seed_start)

    evaluate(args.model_path, args.episodes, args.seed_start)