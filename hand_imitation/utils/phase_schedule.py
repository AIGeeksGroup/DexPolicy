import math

from stable_baselines3.common.callbacks import BaseCallback

from hand_imitation.utils.phase_schedule_core import PRESETS, progress_clock, schedule_values


class PhaseScheduleCallback(BaseCallback):
    def __init__(self, variant, preset, progress_trigger=False, progress_threshold=0.95, verbose=0):
        super().__init__(verbose)
        valid_variants = {"baseline", "std_only", "std_centered", "optimizer_only", "full"}
        if variant not in valid_variants:
            raise ValueError(f"Unknown schedule variant: {variant}")
        if preset not in PRESETS:
            raise ValueError(f"Unknown schedule preset: {preset}")
        self.variant = variant
        self.preset = preset
        self.control_std = variant in {"std_only", "std_centered", "full"}
        self.center_std = variant == "std_centered"
        self.control_optimizer = variant in {"optimizer_only", "full"}
        self.progress_trigger = bool(progress_trigger)
        self.progress_threshold = float(progress_threshold)
        self.progress_start_step = None
        self.current = None
        self.std_grad_hook = None

    def _on_training_start(self):
        if self.variant == "baseline":
            return
        if self.control_std:
            if self.center_std:
                self.model.policy.log_std.requires_grad_(True)
            elif self.model.__class__.__name__ == "TRPO":
                self.std_grad_hook = self.model.policy.log_std.register_hook(lambda grad: grad * 0)
            else:
                self.model.policy.log_std.requires_grad_(False)
        self._apply()

    def _on_rollout_start(self):
        if self.variant != "baseline":
            self._apply()

    def _on_step(self):
        return True

    def _apply(self):
        if self.progress_trigger and self.progress_start_step is None:
            successes = [
                info["pregrasp_success"]
                for info in self.model.ep_info_buffer
                if "pregrasp_success" in info
            ]
            self.progress_start_step, _ = progress_clock(
                self.num_timesteps,
                self.progress_start_step,
                successes,
                self.progress_threshold,
            )
        schedule_step = self.num_timesteps
        if self.progress_trigger:
            _, schedule_step = progress_clock(
                self.num_timesteps,
                self.progress_start_step,
                [],
                self.progress_threshold,
            )
        values = schedule_values(self.preset, schedule_step)
        self.current = values
        if self.control_std:
            if self.center_std:
                target_std = values["action_std"]
                current_std = self.model.policy.log_std.data.exp().mean().item()
                self.model.policy.log_std.data.add_(math.log(target_std / current_std))
            else:
                self.model.policy.log_std.requires_grad_(values["learn_std"])
            if not self.center_std and not values["learn_std"]:
                log_std = math.log(values["action_std"])
                self.model.policy.log_std.data.fill_(log_std)
        if self.control_optimizer:
            learning_rate = values["learning_rate"]
            clip_range = values["clip_range"]
            self.model.learning_rate = learning_rate
            self.model.lr_schedule = lambda _: learning_rate
            self.model.clip_range = lambda _: clip_range
            self.model.n_epochs = values["n_epochs"]
            self.model.max_grad_norm = values["max_grad_norm"]
            for group in self.model.policy.optimizer.param_groups:
                group["lr"] = learning_rate
        self.logger.record("schedule/phase_index", values["phase_index"])
        self.logger.record("schedule/progress_triggered", float(self.progress_start_step is not None))
        self.logger.record("schedule/effective_step", int(schedule_step))
        self.logger.record("schedule/action_std", float(self.model.policy.log_std.exp().mean().item()))
        self.logger.record("schedule/std_learned", float(self.center_std or values["learn_std"]))
        self.logger.record("schedule/learning_rate", float(self.model.policy.optimizer.param_groups[0]["lr"]))
        if hasattr(self.model, "clip_range"):
            clip_value = self.model.clip_range(1.0) if callable(self.model.clip_range) else self.model.clip_range
            self.logger.record("schedule/clip_range", float(clip_value))
        if hasattr(self.model, "n_epochs"):
            self.logger.record("schedule/n_epochs", int(self.model.n_epochs))
        self.logger.record("schedule/max_grad_norm", float(self.model.max_grad_norm))
