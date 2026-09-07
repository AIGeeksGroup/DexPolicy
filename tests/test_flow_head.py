import torch

from algos.rl.fpo_core import FPOPolicyConfig, FPOStatePolicy


def test_flow_head_is_differentiable_and_deterministic():
    torch.manual_seed(0)
    policy = FPOStatePolicy(FPOPolicyConfig(
        obs_dim=4, action_dim=2, actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8), timestep_embed_dim=8, sampling_steps=2,
    ))
    obs = torch.randn(3, 4)
    action = policy.act(obs, deterministic=True)
    assert action.shape == (3, 2)
    assert torch.isfinite(action).all()
    assert torch.equal(action, policy.act(obs, deterministic=True))
    action.sum().backward()
    assert any(p.grad is not None for p in policy.actor.parameters())
