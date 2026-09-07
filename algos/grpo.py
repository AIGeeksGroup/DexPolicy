import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import PPO


def trajectory_group_advantages(rewards, episode_starts, final_episode_starts):
    """Assign each complete trajectory its group-normalized episodic return."""
    rewards = np.asarray(rewards)
    episode_starts = np.asarray(episode_starts, dtype=bool)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    segments = []

    for env_index in range(rewards.shape[1]):
        starts = np.flatnonzero(episode_starts[:, env_index]).tolist()
        if final_episode_starts[env_index]:
            starts.append(rewards.shape[0])
        for start, end in zip(starts[:-1], starts[1:]):
            segments.append((start, end, env_index, rewards[start:end, env_index].sum()))

    returns = np.asarray([segment[3] for segment in segments], dtype=np.float32)
    if len(returns) >= 2:
        normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
        for (start, end, env_index, _), advantage in zip(segments, normalized):
            advantages[start:end, env_index] = advantage
    return advantages, len(segments)


class TrajectoryGRPO(PPO):
    """Critic-free clipped policy update with trajectory-group advantages."""

    def train(self):
        advantages, trajectory_count = trajectory_group_advantages(
            self.rollout_buffer.rewards,
            self.rollout_buffer.episode_starts,
            self._last_episode_starts,
        )
        self.rollout_buffer.advantages = advantages

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses = []
        policy_losses = []
        clip_fractions = []
        approx_kl_divs = []

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                _, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -th.min(
                    rollout_data.advantages * ratio,
                    rollout_data.advantages
                    * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                entropy_loss = -th.mean(entropy) if entropy is not None else th.mean(log_prob)
                loss = policy_loss + self.ent_coef * entropy_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kl_divs.append(approx_kl)
                clip_fractions.append(clip_fraction)

        self._n_updates += self.n_epochs
        self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/complete_trajectories", trajectory_count)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
