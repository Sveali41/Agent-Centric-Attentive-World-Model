import numpy as np
import torch

from modelBased.continue_learning.reservoir_buffer import ReservoirReplayBuffer
from modelBased.world_model.crafter_dynamics import balanced_categorical_effect_loss


def test_focal_is_per_cell_and_gamma_zero_matches_ce():
    logits = torch.tensor(
        [[[[6.0]], [[0.0]], [[0.0]]], [[[0.0]], [[1.0]], [[0.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0]], [[1]]])
    natural = balanced_categorical_effect_loss(logits, target, reduction="mean", focal_gamma=0.0)
    focal = balanced_categorical_effect_loss(logits, target, reduction="mean", focal_gamma=1.0)
    assert torch.isfinite(focal)
    assert focal < natural
    focal.backward()
    assert torch.isfinite(logits.grad).all()


def test_reservoir_keeps_aligned_transitions_and_seen_count():
    samples = {
        "obs": np.arange(20).reshape(10, 2),
        "obs_next": np.arange(20, 40).reshape(10, 2),
        "act": np.arange(10),
        "rew": np.arange(10, dtype=np.float32),
        "done": np.zeros(10, dtype=bool),
        "info": np.asarray([{"id": i} for i in range(10)], dtype=object),
    }
    buffer = ReservoirReplayBuffer(max_size=4, seed=11)
    buffer.add_from_batch(samples)
    assert len(buffer) == 4
    assert buffer.num_seen == 10
    exported = buffer.export_dict()
    for index in range(len(exported["obs"])):
        source_id = int(exported["act"][index])
        assert np.array_equal(exported["obs"][index], samples["obs"][source_id])
        assert np.array_equal(exported["obs_next"][index], samples["obs_next"][source_id])
        assert exported["info"][index]["id"] == source_id
