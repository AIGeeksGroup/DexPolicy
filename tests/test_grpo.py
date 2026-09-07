import numpy as np

from algos.grpo import trajectory_group_advantages


def test_trajectory_group_advantages_use_only_complete_episodes():
    rewards = np.array(
        [[1, 9], [1, 9], [2, 3], [2, 3], [4, 1], [4, 1]], dtype=np.float32
    )
    starts = np.array(
        [[1, 0], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0]], dtype=bool
    )
    advantages, count = trajectory_group_advantages(rewards, starts, [True, True])

    assert count == 5
    assert np.all(advantages[:2, 1] == 0)
    assert advantages[0, 0] == advantages[1, 0]
    assert advantages[2, 1] == advantages[3, 1]
    assert advantages[4, 0] > advantages[4, 1]
