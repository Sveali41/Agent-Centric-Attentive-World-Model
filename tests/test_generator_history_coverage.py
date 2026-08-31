import numpy as np
import pytest
import torch

from generator.history_encoder import HistoryEncoder
from modelBased.world_model.AttentionWM import AttentionWorldModel


def test_coverage_distinguishes_unobserved_from_zero_loss():
    accumulator = [[[] for _ in range(3)] for _ in range(3)]
    accumulator[1][1] = [0.0]
    accumulator[1][2] = [1.0, 3.0]

    loss_map, coverage_map = AttentionWorldModel.average_loss_and_coverage_maps(
        accumulator
    )

    assert loss_map[0, 0] == 0.0
    assert coverage_map[0, 0] == 0.0
    assert loss_map[1, 1] == 0.0
    assert coverage_map[1, 1] == 1.0
    assert loss_map[1, 2] == 2.0
    assert coverage_map[1, 2] == 1.0
    assert loss_map.dtype == np.float32
    assert coverage_map.dtype == np.float32


def test_minigrid_history_accepts_loss_and_coverage_channels():
    encoder = HistoryEncoder(context_dim=16, emb_dim=4, env_type="minigrid")
    state = torch.zeros((2, 3, 8, 8), dtype=torch.long)
    state[:, 0] = 1
    state[:, 0, 4, 4] = 10
    feedback = torch.zeros((2, 2, 8, 8), dtype=torch.float32)
    feedback[:, 1] = 1.0

    context = encoder(state, feedback)

    assert context.shape == (2, 16)
    assert torch.isfinite(context).all()

    with pytest.raises(ValueError, match="expects 2 spatial feedback"):
        encoder(state, feedback[:, :1])


def test_minigrid_history_is_translation_invariant_for_local_failure_pattern():
    torch.manual_seed(3)
    encoder = HistoryEncoder(context_dim=16, emb_dim=4, env_type="minigrid")
    encoder.eval()

    state = torch.zeros((2, 3, 12, 12), dtype=torch.long)
    feedback = torch.zeros((2, 2, 12, 12), dtype=torch.float32)
    for batch_idx, offset in enumerate((0, 2)):
        y, x = 4 + offset, 4 + offset
        state[batch_idx, 0, y, x] = 4
        state[batch_idx, 0, y, x + 1] = 5
        state[batch_idx, 1, y, x] = 1
        state[batch_idx, 2, y, x + 1] = 1
        feedback[batch_idx, 0, y, x] = 2.0
        feedback[batch_idx, 1, y, x] = 1.0

    with torch.no_grad():
        context = encoder(state, feedback)

    assert encoder.net[0].in_channels == 4 * 3 + 2
    assert torch.allclose(context[0], context[1], atol=1e-5, rtol=1e-5)
