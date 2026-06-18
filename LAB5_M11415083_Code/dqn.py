# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Task 1: Vanilla DQN on CartPole-v1

import os
import random
import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
    Vanilla DQN network for CartPole-v1.

    CartPole state dimension: 4
    CartPole action dimension: 2
    """

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


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.args = args

        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)

        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.state_dim, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)

        self.target_net = DQN(self.state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = deque(maxlen=args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor

        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0

        self.best_reward = 0.0

        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def train(self):
        """
        One DQN update step.
        """

        if len(self.memory) < self.replay_start_size:
            return

        if len(self.memory) < self.batch_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.train_count += 1

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            actions, dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )

        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 100 == 0:
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Q Mean": q_values.mean().item(),
                    "Q Max": q_values.max().item(),
                    "Update Count": self.train_count,
                    "Env Step Count": self.env_count,
                    "Epsilon": self.epsilon,
                }
            )

    def run(self, episodes):
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.args.seed + ep)
            state = obs

            done = False
            total_reward = 0.0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = next_obs

                self.memory.append(
                    (state, action, reward, next_state, float(done))
                )

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

            print(
                f"[Train] Ep: {ep:4d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Env Steps: {self.env_count:7d} | "
                f"Updates: {self.train_count:7d} | "
                f"Epsilon: {self.epsilon:.4f}"
            )

            wandb.log(
                {
                    "Episode": ep,
                    "Train Episode Reward": total_reward,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                }
            )

            if ep % self.args.eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=20)

                print(
                    f"[Eval] Ep: {ep:4d} | "
                    f"Average Reward: {eval_reward:.2f} | "
                    f"Best Reward: {self.best_reward:.2f}"
                )

                wandb.log(
                    {
                        "Episode": ep,
                        "Eval Reward": eval_reward,
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

            if ep % self.args.save_interval == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved checkpoint to {model_path}")

        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        torch.save(self.q_net.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")

    def evaluate(self, num_episodes=20):
        """
        Evaluate with greedy policy.
        The assignment grades Task 1 using average evaluation score over 20 episodes.
        """

        rewards = []

        for ep in range(num_episodes):
            obs, _ = self.test_env.reset(seed=ep)
            state = obs

            done = False
            total_reward = 0.0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax(dim=1).item()

                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated

                state = next_obs
                total_reward += reward
                step_count += 1

            rewards.append(total_reward)

        return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, default="./results/task1")
    parser.add_argument("--wandb-run-name", type=str, default="task1_cartpole_dqn")

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=50000)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)

    parser.add_argument("--target-update-frequency", type=int, default=500)
    parser.add_argument("--replay-start-size", type=int, default=1000)

    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--train-per-step", type=int, default=1)

    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(
        project="DLP-Lab5-DQN-CartPole",
        name=args.wandb_run_name,
        config=vars(args),
        save_code=True,
    )

    agent = DQNAgent(env_name="CartPole-v1", args=args)
    agent.run(episodes=args.episodes)

    wandb.finish()


if __name__ == "__main__":
    main()