import pytest

from hand_imitation.utils.phase_schedule_core import schedule_values


def test_screen_schedule_boundaries():
    assert schedule_values("screen_5m", 0)["action_std"] == pytest.approx(0.20)
    assert schedule_values("screen_5m", 1_000_000)["action_std"] == pytest.approx(0.15)
    assert schedule_values("screen_5m", 2_000_000)["action_std"] == pytest.approx(0.08)
    assert schedule_values("screen_5m", 4_000_000)["learning_rate"] == pytest.approx(1.0e-6)


def test_full_schedule_holds_std_after_decay():
    assert schedule_values("full_20m", 5_000_000)["action_std"] == pytest.approx(0.05)
    assert schedule_values("full_20m", 10_000_000)["action_std"] == pytest.approx(0.05)
    assert schedule_values("full_20m", 19_500_000)["n_epochs"] == 1


def test_a2c_hold_015_contracts_once_then_holds():
    assert schedule_values("a2c_hold_015_5m", 0)["action_std"] == pytest.approx(0.20)
    assert schedule_values("a2c_hold_015_5m", 2_000_000)["action_std"] == pytest.approx(0.15)
    assert schedule_values("a2c_hold_015_5m", 5_000_000)["action_std"] == pytest.approx(0.15)


def test_linear_schedule_is_smooth_and_keeps_ppo_update_fixed():
    start = schedule_values("linear_5m", 0)
    midpoint = schedule_values("linear_5m", 2_500_000)
    end = schedule_values("linear_5m", 5_000_000)

    assert start["action_std"] == pytest.approx(0.20)
    assert midpoint["action_std"] == pytest.approx(0.125)
    assert end["action_std"] == pytest.approx(0.05)
    assert end["learning_rate"] == pytest.approx(1.0e-5)
    assert end["clip_range"] == pytest.approx(0.20)
    assert end["n_epochs"] == 5
    assert end["max_grad_norm"] == pytest.approx(0.50)


def test_frontloaded_schedule_removes_boundary_jumps():
    before_2m = schedule_values("frontloaded_5m", 1_999_999)
    at_2m = schedule_values("frontloaded_5m", 2_000_000)
    before_4m = schedule_values("frontloaded_5m", 3_999_999)
    at_4m = schedule_values("frontloaded_5m", 4_000_000)
    end = schedule_values("frontloaded_5m", 5_000_000)

    assert before_2m["action_std"] == pytest.approx(0.08, abs=1e-6)
    assert at_2m["action_std"] == pytest.approx(0.08)
    assert before_4m["action_std"] == pytest.approx(0.05, abs=1e-6)
    assert at_4m["action_std"] == pytest.approx(0.05)
    assert end["action_std"] == pytest.approx(0.05)
    assert end["learning_rate"] == pytest.approx(1.0e-5)
    assert end["clip_range"] == pytest.approx(0.20)


def test_component_ablation_schedules_change_only_one_boundary():
    no_final_before_2m = schedule_values("no_final_drop_5m", 1_999_999)
    no_final_at_2m = schedule_values("no_final_drop_5m", 2_000_000)
    no_final_at_4m = schedule_values("no_final_drop_5m", 4_000_000)
    no_mid_before_2m = schedule_values("no_mid_jump_5m", 1_999_999)
    no_mid_at_2m = schedule_values("no_mid_jump_5m", 2_000_000)
    no_mid_at_4m = schedule_values("no_mid_jump_5m", 4_000_000)

    assert no_final_before_2m["action_std"] == pytest.approx(0.10, abs=1e-6)
    assert no_final_at_2m["action_std"] == pytest.approx(0.08)
    assert no_final_at_4m["action_std"] == pytest.approx(0.05)
    assert no_mid_before_2m["action_std"] == pytest.approx(0.10, abs=1e-6)
    assert no_mid_at_2m["action_std"] == pytest.approx(0.10)
    assert no_mid_at_4m["action_std"] == pytest.approx(0.04)


def test_selected_10m_extends_only_the_final_std_plateau():
    before_2m = schedule_values("selected_10m", 1_999_999)
    at_2m = schedule_values("selected_10m", 2_000_000)
    before_4m = schedule_values("selected_10m", 3_999_999)
    at_4m = schedule_values("selected_10m", 4_000_000)
    at_10m = schedule_values("selected_10m", 10_000_000)

    assert before_2m["action_std"] == pytest.approx(0.10, abs=1e-6)
    assert at_2m["action_std"] == pytest.approx(0.10)
    assert before_4m["action_std"] == pytest.approx(0.05, abs=1e-6)
    assert at_4m["action_std"] == pytest.approx(0.04)
    assert at_10m["action_std"] == pytest.approx(0.04)
    assert at_10m["learning_rate"] == pytest.approx(1.0e-5)
    assert at_10m["clip_range"] == pytest.approx(0.20)
    assert at_10m["n_epochs"] == 5
    assert at_10m["max_grad_norm"] == pytest.approx(0.50)


@pytest.mark.parametrize(
    ("preset", "expected_std"),
    (("fixed_020_5m", 0.20), ("fixed_010_5m", 0.10), ("fixed_004_5m", 0.04)),
)
def test_fixed_std_controls_remain_constant(preset, expected_std):
    for step in (0, 1_000_000, 2_500_000, 4_999_999, 5_000_000):
        values = schedule_values(preset, step)
        assert values["action_std"] == pytest.approx(expected_std)
        assert values["learn_std"] is False


def test_a2c_schedule_stops_contracting_at_point_one():
    start = schedule_values("a2c_hold_010_5m", 0)
    midpoint = schedule_values("a2c_hold_010_5m", 1_000_000)
    at_2m = schedule_values("a2c_hold_010_5m", 2_000_000)
    at_5m = schedule_values("a2c_hold_010_5m", 5_000_000)

    assert start["action_std"] == pytest.approx(0.20)
    assert midpoint["action_std"] == pytest.approx(0.15)
    assert at_2m["action_std"] == pytest.approx(0.10)
    assert at_5m["action_std"] == pytest.approx(0.10)
    assert at_5m["learning_rate"] == pytest.approx(7.0e-4)


def test_grpo_plateau_stops_contracting_at_point_one():
    start = schedule_values("grpo_hold_010_5m", 0)
    midpoint = schedule_values("grpo_hold_010_5m", 1_000_000)
    at_2m = schedule_values("grpo_hold_010_5m", 2_000_000)
    at_5m = schedule_values("grpo_hold_010_5m", 5_000_000)

    assert start["action_std"] == pytest.approx(0.20)
    assert midpoint["action_std"] == pytest.approx(0.15)
    assert at_2m["action_std"] == pytest.approx(0.10)
    assert at_5m["action_std"] == pytest.approx(0.10)
    assert at_5m["learning_rate"] == pytest.approx(1.0e-5)


def test_grpo_reexpansion_restores_exploration_after_contraction():
    start = schedule_values("grpo_reexpand_018_5m", 0)
    at_2m = schedule_values("grpo_reexpand_018_5m", 2_000_000)
    midpoint = schedule_values("grpo_reexpand_018_5m", 2_500_000)
    at_3m = schedule_values("grpo_reexpand_018_5m", 3_000_000)
    at_5m = schedule_values("grpo_reexpand_018_5m", 5_000_000)

    assert start["action_std"] == pytest.approx(0.20)
    assert at_2m["action_std"] == pytest.approx(0.10)
    assert midpoint["action_std"] == pytest.approx(0.14)
    assert at_3m["action_std"] == pytest.approx(0.18)
    assert at_5m["action_std"] == pytest.approx(0.18)
    assert at_5m["learning_rate"] == pytest.approx(1.0e-5)


def test_grpo_contract_release_switches_std_back_to_learning():
    before_2m = schedule_values("grpo_contract_release_5m", 1_999_999)
    at_2m = schedule_values("grpo_contract_release_5m", 2_000_000)
    at_5m = schedule_values("grpo_contract_release_5m", 5_000_000)

    assert before_2m["action_std"] == pytest.approx(0.10, abs=1e-6)
    assert before_2m["learn_std"] is False
    assert at_2m["action_std"] == pytest.approx(0.10)
    assert at_2m["learn_std"] is True
    assert at_5m["learn_std"] is True


def test_grpo_early_release_preserves_more_exploration():
    before_release = schedule_values("grpo_early_release_5m", 1_499_999)
    at_release = schedule_values("grpo_early_release_5m", 1_500_000)
    at_5m = schedule_values("grpo_early_release_5m", 5_000_000)

    assert before_release["action_std"] == pytest.approx(0.125, abs=1e-6)
    assert before_release["learn_std"] is False
    assert at_release["action_std"] == pytest.approx(0.125)
    assert at_release["learn_std"] is True
    assert at_5m["learn_std"] is True


def test_unknown_preset_fails():
    with pytest.raises(ValueError):
        schedule_values("missing", 0)


def test_progress_trigger_holds_schedule_clock_until_pregrasp_gate():
    from hand_imitation.utils.phase_schedule_core import progress_clock

    start, effective = progress_clock(1_000_000, None, [0.90, 0.94], 0.95)
    assert start is None
    assert effective == 0

    start, effective = progress_clock(1_200_000, start, [0.96, 0.95], 0.95)
    assert start == 1_200_000
    assert effective == 0

    start, effective = progress_clock(1_500_000, start, [0.20], 0.95)
    assert start == 1_200_000
    assert effective == 300_000
