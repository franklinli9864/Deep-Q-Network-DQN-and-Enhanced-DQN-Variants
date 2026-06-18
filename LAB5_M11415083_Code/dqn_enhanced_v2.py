# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Task 3 v2: More Stable Enhanced DQN on Atari Pong-v5
#
# Enhancements:
# 1. Double DQN
# 2. Prioritized Experience Replay
# 3. Multi-Step Return

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
            maxlen=self.frame_stack,
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


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity,
        alpha=0.4,
        beta_start=0.4,
        beta_frames=2500000,
        eps=1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.pos = 0
        self.frame = 1

    def beta_by_frame(self):
        beta = self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        return min(1.0, beta)

    def add(self, transition, priority=None):
        if priority is None:
            if len(self.buffer) == 0:
                priority = 1.0
            else:
                priority = self.priorities[:len(self.buffer)].max()
                if priority <= 0:
                    priority = 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        scaled_priorities = priorities ** self.alpha
        total_priority = scaled_priorities.sum()

        if total_priority <= 0:
            probs = np.ones_like(scaled_priorities) / len(scaled_priorities)
        else:
            probs = scaled_priorities / total_priority

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame()
        self.frame += 1

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones, gammas = zip(*samples)

        states = np.array(states, dtype=np.uint8)
        next_states = np.array(next_states, dtype=np.uint8)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        gammas = np.array(gammas, dtype=np.float32)

        return states, actions, rewards, next_states, dones, gammas, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + self.eps

        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = float(error)

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    def __init__(self, n_step=2, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque()

    def append(self, transition):
        self.buffer.append(transition)

    def can_pop(self):
        return len(self.buffer) >= self.n_step

    def pop_n_step_transition(self):
        reward_sum = 0.0
        next_state = self.buffer[0][3]
        done = self.buffer[0][4]
        gamma_power = 1.0

        state = self.buffer[0][0]
        action = self.buffer[0][1]

        for idx, (_, _, reward, n_state, d) in enumerate(self.buffer):
            reward_sum += (self.gamma ** idx) * reward
            next_state = n_state
            done = d
            gamma_power = self.gamma ** (idx + 1)

            if idx + 1 >= self.n_step:
                break

            if d:
                break

        self.buffer.popleft()

        return state, action, reward_sum, next_state, float(done), gamma_power

    def flush(self):
        transitions = []

        while len(self.buffer) > 0:
            transitions.append(self.pop_n_step_transition())

        return transitions

    def clear(self):
        self.buffer.clear()


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

        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=args.per_alpha,
            beta_start=args.per_beta_start,
            beta_frames=args.per_beta_frames,
            eps=args.per_eps,
        )

        self.n_step_buffer = NStepBuffer(
            n_step=args.n_step,
            gamma=args.discount_factor,
        )

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor

        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0

        self.best_reward = -float("inf")
        self.first_reach_19_step = None

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.snapshot_steps = set(args.snapshot_steps)
        self.saved_snapshot_steps = set()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(
            state,
            dtype=torch.uint8,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def add_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if self.n_step_buffer.can_pop():
            n_step_transition = self.n_step_buffer.pop_n_step_transition()
            self.memory.add(n_step_transition)

        if done:
            flushed_transitions = self.n_step_buffer.flush()

            for flushed_transition in flushed_transitions:
                self.memory.add(flushed_transition)

            self.n_step_buffer.clear()

    def train(self):
        if len(self.memory) < self.args.replay_start_size:
            return

        if len(self.memory) < self.batch_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.train_count += 1

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            gammas,
            indices,
            weights,
        ) = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.uint8, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.uint8, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        gammas = torch.tensor(gammas, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q_values_all = self.q_net(states)
        q_values = q_values_all.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # Double DQN:
            # Online network selects action.
            next_actions = self.q_net(next_states).argmax(dim=1)

            # Target network evaluates selected action.
            next_q_values = self.target_net(next_states).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)

            targets = rewards + gammas * next_q_values * (1.0 - dones)

        td_errors = targets - q_values

        elementwise_loss = nn.SmoothL1Loss(reduction="none")(q_values, targets)
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.memory.update_priorities(
            indices,
            td_errors.detach().abs().cpu().numpy(),
        )

        if self.train_count % self.args.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % self.args.log_interval == 0:
            print(
                f"[Train] Env Steps: {self.env_count} | "
                f"Updates: {self.train_count} | "
                f"Loss: {loss.item():.4f} | "
                f"TD mean: {td_errors.abs().mean().item():.4f} | "
                f"Q mean: {q_values.mean().item():.4f} | "
                f"Epsilon: {self.epsilon:.4f}"
            )

            wandb.log(
                {
                    "Loss": loss.item(),
                    "TD Error Mean": td_errors.abs().mean().item(),
                    "Q Mean": q_values.mean().item(),
                    "Q Max": q_values.max().item(),
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                }
            )

    def save_required_snapshots(self):
        for step in sorted(self.snapshot_steps):
            if self.env_count >= step and step not in self.saved_snapshot_steps:
                model_path = os.path.join(self.save_dir, f"model_step{step}.pt")
                torch.save(self.q_net.state_dict(), model_path)

                self.saved_snapshot_steps.add(step)

                print(f"Saved required snapshot to {model_path}")

    def run(self):
        episode = 0

        while self.env_count < self.args.total_steps:
            obs, _ = self.env.reset(seed=self.args.seed + episode)
            state = self.preprocessor.reset(obs)
            self.n_step_buffer.clear()

            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done and self.env_count < self.args.total_steps:
                action = self.select_action(state)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)

                self.add_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(done),
                )

                for _ in range(self.args.train_per_step):
                    self.train()

                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.env_count += 1

                self.save_required_snapshots()

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

                    if eval_reward >= 19.0 and self.first_reach_19_step is None:
                        self.first_reach_19_step = self.env_count
                        print(
                            f"First reached score 19 at env step: "
                            f"{self.first_reach_19_step}"
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
                        f"checkpoint_step{self.env_count}.pt",
                    )
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved checkpoint to {model_path}")

            print(
                f"[Episode] Ep: {episode:4d} | "
                f"Reward: {episode_reward:6.2f} | "
                f"Episode Steps: {episode_steps:5d} | "
                f"Env Steps: {self.env_count:8d} | "
                f"Updates: {self.train_count:8d} | "
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

        if self.first_reach_19_step is not None:
            print(f"First reached score 19 at env step: {self.first_reach_19_step}")
        else:
            print("Did not reach score 19 during this run.")

    def evaluate(self, num_episodes=5):
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
                    device=self.device,
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

    parser.add_argument("--save-dir", type=str, default="./results/task3_v2")
    parser.add_argument("--wandb-run-name", type=str, default="task3_enhanced_dqn_v2")

    parser.add_argument("--total-steps", type=int, default=2500000)

    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--discount-factor", type=float, default=0.99)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)

    parser.add_argument("--replay-start-size", type=int, default=10000)

    parser.add_argument("--target-update-frequency", type=int, default=5000)
    parser.add_argument("--train-per-step", type=int, default=1)

    parser.add_argument("--per-alpha", type=float, default=0.4)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=2500000)
    parser.add_argument("--per-eps", type=float, default=1e-6)

    parser.add_argument("--n-step", type=int, default=2)

    parser.add_argument("--eval-interval-steps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--save-interval-steps", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=1000)

    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=[600000, 1000000, 1500000, 2000000, 2500000],
    )

    parser.add_argument("--seed", type=int, default=43)

    args = parser.parse_args()

    set_seed(args.seed)

    wandb.init(
        project="DLP-Lab5-Enhanced-DQN-Pong",
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