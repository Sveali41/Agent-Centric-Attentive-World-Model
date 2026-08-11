import numpy as np
import torch
from omegaconf import OmegaConf

from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.exploration.p2e import CrafterP2EExplorer, P2EEnsemble
from modelBased.policy_training.PPO import PPO


class _FakeCrafterWM(torch.nn.Module):
    mask_size = 5

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode(self, state):
        return torch.zeros((state.shape[0], 1, 8), device=state.device) + self.anchor

    def forward(self, state, action, info, inv=None):
        logits = torch.zeros(
            (state.shape[0], 27, state.shape[2], state.shape[3]),
            device=state.device,
        ) + self.anchor
        return logits, None, torch.zeros_like(inv)


def _cfg():
    return OmegaConf.create(
        {
            "domains": {
                "crafter": {
                    "inventory_dim": 16,
                    "observation_schema": [
                        {"name": "inventory", "target_mode": "delta"}
                    ],
                }
            },
            "attention_model": {"embed_dim": 8, "action_norm_values": 3},
            "p2e": {
                "inventory_scale": 9.0,
                "intrinsic_reward_scale": 1.0,
                "imagined_horizon": 2,
                "imagined_batch_size": 4,
                "imagined_batches_per_cycle": 1,
            },
        }
    )


def _batch(count=8):
    images = np.zeros((count, 15, 15, 2), dtype=np.int64)
    images[:, 7, 7, 0] = 13
    images[:, 7, 7, 1] = 1
    inventories = np.full((count, 16), 5.0, dtype=np.float32)
    return images, inventories


def test_imagined_actor_updates_on_latest_world_model():
    cfg = _cfg()
    world_model = _FakeCrafterWM()
    ensemble = P2EEnsemble(24, 3, num_models=3, device="cpu")
    ppo = PPO(24, 3, 3e-4, 1e-3, 0.99, 2, 0.2, False, normalize_returns=False)
    explorer = CrafterP2EExplorer(ppo, world_model, ensemble, cfg)
    images, inventories = _batch()

    metrics = explorer.adapt_actor_imagined(images, inventories)

    assert metrics["imagined_transitions"] == 8
    assert metrics["imagined_updates"] == 1
    assert metrics["imagined_parameter_delta"] > 0.0
    assert not any(parameter.grad is not None for parameter in world_model.parameters())


def test_replay_saliency_is_layout_invariant():
    images, _ = _batch(2)
    images[:, 7, 8, 0] = 6
    buffer = FisherReplayBuffer(10)
    nhwc = buffer.get_agent_near_elements_mask(torch.as_tensor(images))
    nchw = buffer.get_agent_near_elements_mask(
        torch.as_tensor(images).permute(0, 3, 1, 2)
    )
    assert torch.equal(nhwc, nchw)


def test_task_aware_intrinsic_rewards_progress_and_penalizes_death():
    cfg = _cfg()
    world_model = _FakeCrafterWM()
    ensemble = P2EEnsemble(24, 3, num_models=3, device="cpu")
    ppo = PPO(24, 3, 3e-4, 1e-3, 0.99, 2, 0.2, False, normalize_returns=False)
    explorer = CrafterP2EExplorer(ppo, world_model, ensemble, cfg)
    images, inventories = _batch(1)
    observation = {"image": images[0], "inventory": inventories[0]}

    gained = inventories[0].copy()
    gained[5] += 1.0  # stone
    progressed = {"image": images[0], "inventory": gained}
    assert explorer.compute_intrinsic_reward(observation, 0, progressed) > 0.0

    dead = inventories[0].copy()
    dead[0] = 0.0  # health
    terminal = {"image": images[0], "inventory": dead}
    assert explorer.compute_intrinsic_reward(observation, 0, terminal) < 0.0
