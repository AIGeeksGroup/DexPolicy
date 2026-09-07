"""Small CPU-only API check; not a manipulation experiment."""
import argparse

import gym
import numpy as np
import torch
from stable_baselines3 import PPO

from algos.grpo import TrajectoryGRPO
from hand_imitation.utils.phase_schedule import PhaseScheduleCallback


class SmokeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def seed(self, seed=None):
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        return np.zeros(4, dtype=np.float32), -float(np.square(action).sum()), self.steps >= 8, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), default="ppo")
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cls = PPO if args.algorithm == "ppo" else TrajectoryGRPO
    model = cls("MlpPolicy", SmokeEnv(), n_steps=32, batch_size=16,
                n_epochs=1, seed=0, device="cpu", verbose=0)
    preset = "no_mid_jump_5m" if args.algorithm == "ppo" else "grpo_hold_010_5m"
    model.learn(args.steps, callback=PhaseScheduleCallback("std_only", preset))
    print(f"{args.algorithm}: completed {model.num_timesteps} transitions")


if __name__ == "__main__":
    main()
