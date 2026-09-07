# DexPolicy: Scheduled Exploration for Policy Optimization in Dexterous Manipulation

Method code for scheduled exploration with Gaussian PPO, trajectory-group GRPO,
and a flow-generated Gaussian action mean.

## Release Contents

- `hand_imitation/utils/phase_schedule_core.py`: schedule definitions and ablations.
- `hand_imitation/utils/phase_schedule.py`: Stable-Baselines3 callback.
- `algos/grpo.py`: critic-free trajectory-group clipped policy updates.
- `algos/rl/fpo_core.py`: flow action head and objective utilities.
- `algos/rl/fpo_trainer.py`: flow-head training implementation.
- `algos/rl/config/agent/`: research configurations.
- `tests/`: schedule, group-advantage, and flow-head checks.

This is a method-core release, not a standalone simulation benchmark package.
Simulator environments, object meshes, motion datasets, pretrained weights,
robot drivers, and real-robot data are not bundled. Training on the manipulation
tasks requires separately supplied environments and licensed assets.

## Environment

The research code uses Python 3.10, PyTorch 2.4.1, NumPy below 1.24,
Gym 0.25.2, and Stable-Baselines3 1.1.0. It uses the legacy Gym API, not
Gymnasium. The original environment combines these Gym/SB3 versions despite
their package metadata constraints; use an existing compatible environment
rather than silently upgrading SB3. PyYAML, OmegaConf, and pytest are also
needed for configurations and tests.

From the repository root:

```bash
python -m pytest tests -q
python -m examples.smoke --algorithm ppo --steps 128
python -m examples.smoke --algorithm grpo --steps 128
```

The smoke commands use a synthetic continuous-control environment only to
check the training API. They do not reproduce manipulation results.

## Schedule Integration

Pass `PhaseScheduleCallback` to the model's `learn(..., callback=...)` call.
Use `variant="baseline"` for the unchanged Gaussian policy and
`variant="std_only", preset="no_mid_jump_5m"` for the selected PPO schedule.
That schedule contracts std from 0.20 to 0.10 over 0--2M transitions, to 0.05
over 2--4M, then holds 0.04. Use total environment transitions for the clock,
not the number of vector-environment calls. The schedule changes exploration;
the `std_only` variant does not change optimizer hyperparameters.

For the GRPO plateau experiment use `preset="grpo_hold_010_5m"`.
Other presets are research ablations, not additional recommended methods.
The supplied trajectory-group implementation normalizes returns across complete
episodes within a rollout; incomplete boundary fragments receive zero advantage.
It is not a prompt-conditioned language-model GRPO implementation.

## Flow-Head PPO

Legacy `fpo_*` filenames are retained to preserve internal imports. They do not
identify this experiment with the official FPO or FPO++ algorithms. Set
`actor_objective=gaussian_ppo`, `trust_region_mode=ppo`, and
`action_head_mode=flow_residual`. With no base-policy checkpoint, this mode
returns the flow-generated mean directly: there is no pretrained PPO residual.

The current paired experiment compares fixed std 0.10 with linear decay
0.20 to 0.05 over 5M transitions (`gaussian_action_std=0.20`,
`gaussian_action_std_decay_steps=5000000`,
`gaussian_action_std_min_scale=0.25`). This is distinct from the PPO stagewise
schedule above. Ongoing runs are not published as completed evidence.

## Attribution

The research implementation builds on code by Zerui Chen and collaborators
(2025; https://arxiv.org/abs/2404.15709). The original MIT copyright and
license are preserved in `LICENSE`. PPO infrastructure is provided by
Stable-Baselines3 (https://github.com/DLR-RM/stable-baselines3).
Redistributed dependencies retain their own licenses; no third-party datasets
or robot assets are included here.
