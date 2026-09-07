import numpy as np


SCREEN_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 4_000_000,
        "std_start": 0.08,
        "std_end": 0.05,
        "learning_rate": 3.0e-6,
        "clip_range": 0.10,
        "n_epochs": 3,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 4_000_000,
        "end_step": 5_000_000,
        "std_start": 0.04,
        "std_end": 0.04,
        "learning_rate": 1.0e-6,
        "clip_range": 0.05,
        "n_epochs": 2,
        "max_grad_norm": 0.25,
    },
)


LINEAR_5M = (
    {
        "start_step": 0,
        "end_step": 5_000_000,
        "std_start": 0.20,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


FRONTLOADED_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.08,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 4_000_000,
        "std_start": 0.08,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 4_000_000,
        "end_step": 5_000_000,
        "std_start": 0.05,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


NO_FINAL_DROP_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 4_000_000,
        "std_start": 0.08,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 4_000_000,
        "end_step": 5_000_000,
        "std_start": 0.05,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


NO_MID_JUMP_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 4_000_000,
        "std_start": 0.10,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 4_000_000,
        "end_step": 5_000_000,
        "std_start": 0.04,
        "std_end": 0.04,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


FIXED_020_5M = (
    {
        "start_step": 0,
        "end_step": 5_000_000,
        "std_start": 0.20,
        "std_end": 0.20,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


FIXED_010_5M = (
    {
        "start_step": 0,
        "end_step": 5_000_000,
        "std_start": 0.10,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


FIXED_004_5M = (
    {
        "start_step": 0,
        "end_step": 5_000_000,
        "std_start": 0.04,
        "std_end": 0.04,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


A2C_HOLD_010_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 7.0e-4,
        "clip_range": 0.20,
        "n_epochs": 1,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 5_000_000,
        "std_start": 0.10,
        "std_end": 0.10,
        "learning_rate": 7.0e-4,
        "clip_range": 0.20,
        "n_epochs": 1,
        "max_grad_norm": 0.50,
    },
)


A2C_HOLD_015_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.15,
        "learning_rate": 7.0e-4,
        "clip_range": 0.20,
        "n_epochs": 1,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 5_000_000,
        "std_start": 0.15,
        "std_end": 0.15,
        "learning_rate": 7.0e-4,
        "clip_range": 0.20,
        "n_epochs": 1,
        "max_grad_norm": 0.50,
    },
)


GRPO_HOLD_010_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 5_000_000,
        "std_start": 0.10,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


GRPO_REEXPAND_018_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 3_000_000,
        "std_start": 0.10,
        "std_end": 0.18,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 3_000_000,
        "end_step": 5_000_000,
        "std_start": 0.18,
        "std_end": 0.18,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


GRPO_CONTRACT_RELEASE_5M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 5_000_000,
        "std_start": 0.10,
        "std_end": 0.10,
        "learn_std": True,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


GRPO_EARLY_RELEASE_5M = (
    {
        "start_step": 0,
        "end_step": 1_500_000,
        "std_start": 0.20,
        "std_end": 0.125,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 1_500_000,
        "end_step": 5_000_000,
        "std_start": 0.125,
        "std_end": 0.125,
        "learn_std": True,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


SELECTED_10M = (
    {
        "start_step": 0,
        "end_step": 2_000_000,
        "std_start": 0.20,
        "std_end": 0.10,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 2_000_000,
        "end_step": 4_000_000,
        "std_start": 0.10,
        "std_end": 0.05,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 4_000_000,
        "end_step": 10_000_000,
        "std_start": 0.04,
        "std_end": 0.04,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
)


FULL_20M = (
    {
        "start_step": 0,
        "end_step": 12_000_000,
        "std_start": 0.20,
        "std_end": 0.05,
        "std_decay_end_step": 5_000_000,
        "learning_rate": 1.0e-5,
        "clip_range": 0.20,
        "n_epochs": 5,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 12_000_000,
        "end_step": 13_000_000,
        "std_start": 0.05,
        "std_end": 0.05,
        "learning_rate": 3.0e-6,
        "clip_range": 0.10,
        "n_epochs": 4,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 13_000_000,
        "end_step": 14_000_000,
        "std_start": 0.045,
        "std_end": 0.03,
        "learning_rate": 1.0e-6,
        "clip_range": 0.06,
        "n_epochs": 3,
        "max_grad_norm": 0.50,
    },
    {
        "start_step": 14_000_000,
        "end_step": 19_000_000,
        "std_start": 0.03,
        "std_end": 0.022,
        "learning_rate": 2.0e-7,
        "clip_range": 0.03,
        "n_epochs": 2,
        "max_grad_norm": 0.25,
    },
    {
        "start_step": 19_000_000,
        "end_step": 20_000_000,
        "std_start": 0.022,
        "std_end": 0.022,
        "learning_rate": 2.0e-7,
        "clip_range": 0.025,
        "n_epochs": 1,
        "max_grad_norm": 0.10,
    },
)


PRESETS = {
    "screen_5m": SCREEN_5M,
    "linear_5m": LINEAR_5M,
    "frontloaded_5m": FRONTLOADED_5M,
    "no_final_drop_5m": NO_FINAL_DROP_5M,
    "no_mid_jump_5m": NO_MID_JUMP_5M,
    "fixed_020_5m": FIXED_020_5M,
    "fixed_010_5m": FIXED_010_5M,
    "fixed_004_5m": FIXED_004_5M,
    "a2c_hold_010_5m": A2C_HOLD_010_5M,
    "a2c_hold_015_5m": A2C_HOLD_015_5M,
    "grpo_hold_010_5m": GRPO_HOLD_010_5M,
    "grpo_reexpand_018_5m": GRPO_REEXPAND_018_5M,
    "grpo_contract_release_5m": GRPO_CONTRACT_RELEASE_5M,
    "grpo_early_release_5m": GRPO_EARLY_RELEASE_5M,
    "selected_10m": SELECTED_10M,
    "full_20m": FULL_20M,
}


def progress_clock(num_timesteps, start_step, success_values, threshold=0.95):
    if start_step is None and success_values and float(np.mean(success_values)) >= threshold:
        start_step = int(num_timesteps)
    effective_step = 0 if start_step is None else max(int(num_timesteps) - start_step, 0)
    return start_step, effective_step


def schedule_values(preset, step):
    if preset not in PRESETS:
        raise ValueError(f"Unknown PPO schedule preset: {preset}")
    phases = PRESETS[preset]
    phase_index = len(phases) - 1
    for index, phase in enumerate(phases):
        if step < phase["end_step"]:
            phase_index = index
            break
    phase = phases[phase_index]
    decay_end = phase.get("std_decay_end_step", phase["end_step"])
    denominator = max(decay_end - phase["start_step"], 1)
    progress = np.clip((step - phase["start_step"]) / denominator, 0.0, 1.0)
    action_std = phase["std_start"] + progress * (phase["std_end"] - phase["std_start"])
    return {
        "phase_index": phase_index,
        "action_std": float(action_std),
        "learn_std": bool(phase.get("learn_std", False)),
        "learning_rate": float(phase["learning_rate"]),
        "clip_range": float(phase["clip_range"]),
        "n_epochs": int(phase["n_epochs"]),
        "max_grad_norm": float(phase["max_grad_norm"]),
    }
