from types import SimpleNamespace

import numpy as np
import torch

from generator.generator_interface import GeneratorInterface
from generator.generator_network import MapEditorActorCritic
from modelBased.world_model.AttentionWM_support import AttentionModule


def _pair_distance(mask):
    positions = torch.nonzero(mask, as_tuple=False).float()
    distances = torch.cdist(positions, positions, p=1)
    return distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)].mean()


def test_minigrid_dpp_selects_editable_deterministic_spread_positions():
    policy = MapEditorActorCritic(
        num_actions=13, context_dim=8, env_type="minigrid", spatial_dpp_sigma=1.5
    )
    logits = torch.zeros(1, 13, 8, 8)
    logits[:, 1:, 3:6, 3:6] = 4.0
    immutable = torch.zeros(1, 1, 8, 8)
    immutable[:, :, 0, :] = 1
    immutable[:, :, -1, :] = 1
    immutable[:, :, :, 0] = 1
    immutable[:, :, :, -1] = 1
    first = policy._get_topk_mask(logits, 0.25, immutable)
    second = policy._get_topk_mask(logits, 0.25, immutable)

    assert torch.equal(first, second)
    assert int(first.sum()) == round(0.25 * 36)
    assert not first[0][immutable[0, 0].bool()].any()
    assert _pair_distance(first[0]) > 0


def test_minigrid_history_changes_types_but_not_placement_logits():
    torch.manual_seed(5)
    policy = MapEditorActorCritic(
        num_actions=13, context_dim=8, env_type="minigrid", spatial_dpp_sigma=1.5
    )
    policy.eval()
    map_vec = torch.zeros(1, 3, 8, 8)
    context_a = torch.zeros(1, 8)
    context_b = torch.ones(1, 8)
    immutable = torch.zeros(1, 1, 8, 8)
    immutable[:, :, 0, :] = 1
    immutable[:, :, -1, :] = 1
    immutable[:, :, :, 0] = 1
    immutable[:, :, :, -1] = 1

    with torch.no_grad():
        features_a = policy.forward_features(map_vec, context_a)
        features_b = policy.forward_features(map_vec, context_b)
        placement_a, type_logits_a, _ = policy._minigrid_logits(
            features_a, context_a
        )
        placement_b, type_logits_b, _ = policy._minigrid_logits(
            features_b, context_b
        )
        mask_a = policy._get_topk_mask(placement_a, 0.25, immutable)
        mask_b = policy._get_topk_mask(placement_b, 0.25, immutable)

    assert torch.equal(features_a, features_b)
    assert torch.equal(placement_a, placement_b)
    assert torch.equal(mask_a, mask_b)
    assert not torch.allclose(type_logits_a[:, 1:], type_logits_b[:, 1:])


def test_minigrid_history_conditioned_act_and_evaluate_are_finite():
    torch.manual_seed(9)
    policy = MapEditorActorCritic(
        num_actions=13, context_dim=8, env_type="minigrid", spatial_dpp_sigma=1.5
    )
    map_vec = torch.zeros(2, 3, 8, 8)
    context = torch.randn(2, 8)
    immutable = torch.zeros(2, 1, 8, 8)
    immutable[:, :, 0, :] = 1
    immutable[:, :, -1, :] = 1
    immutable[:, :, :, 0] = 1
    immutable[:, :, :, -1] = 1

    action, stats_action, _, _, _, topk_mask, stats_topk_mask = policy.act(
        map_vec, context, immutable, max_edits=0.25
    )
    logp, stats_logp, value, entropy = policy.evaluate(
        map_vec,
        context,
        (action, stats_action),
        immutable,
        target_topk_mask=topk_mask,
        target_stats_topk_mask=stats_topk_mask,
    )

    assert torch.isfinite(logp).all()
    assert torch.isfinite(stats_logp).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(entropy)


def test_world_model_map_features_have_stable_shape_and_finite_values():
    model = AttentionModule(
        data_type="discrete",
        grid_shape=(3, 8, 8),
        mask_size=5,
        embed_dim=16,
        num_heads=4,
        env_type="minigrid",
        minigrid_transition_mode="effect",
    )
    states = torch.zeros(2, 3, 8, 8, dtype=torch.long)
    states[1, 0, 2, 3] = 10
    features = model.encode_map_features(states)
    assert features.shape == (2, 32)
    assert torch.isfinite(features).all()


class _FakeWM:
    mask_size = 3

    def encode_map_features(self, states):
        # The first cell is enough to make identical maps identical and
        # distinct maps different while preserving the public WM interface.
        return states.flatten(1).float()[:, :4] + 1e-3

    def encode_state_action_features(self, local_states, actions, inventory):
        features = torch.cat(
            [local_states.flatten(1).float(), actions.float().unsqueeze(1), inventory.float().unsqueeze(1)],
            dim=1,
        )
        return torch.nn.functional.normalize(features, p=2, dim=1)


def _fake_interface():
    interface = GeneratorInterface.__new__(GeneratorInterface)
    interface.is_minigrid = True
    interface.map_archive = []
    interface.map_archive_size = 8
    interface.map_knn_k = 2
    interface.map_batch_archive_mix = 0.5
    interface.latent_kernel_temperature = 0.2
    interface.state_action_archive = []
    interface.state_action_archive_size = 8
    interface.state_action_knn_k = 2
    interface.state_action_novelty_samples = 128
    interface.state_action_batch_archive_mix = 0.5
    interface.device = torch.device("cpu")
    interface.wm = _FakeWM()
    interface.OBJ_EMPTY = 1
    interface.OBJ_START = 10
    interface.OBJ_GOAL = 8
    interface.minigrid_reward_cfg = SimpleNamespace(
        loss=3.0, map_novelty=0.0, state_action_novelty=3.0,
        reach=2.0, dist=2.0, solvable=2.0, bias=0.0,
        loss_transform="log1p", loss_scale=1000.0, clip=60.0,
    )
    interface.ablation_type = "none"
    interface._minigrid_loss_ema = None
    return interface


def test_map_archive_scores_before_batch_update_without_order_bias():
    interface = _fake_interface()
    same = np.zeros((3, 2, 2), dtype=np.int64)
    different = same.copy()
    different[0, 0, 0] = 1
    scores, logdet = interface._minigrid_map_novelty([same, different])

    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))
    assert len(interface.map_archive) == 2
    assert np.isfinite(logdet)

    interface = _fake_interface()
    identical_scores, _ = interface._minigrid_map_novelty([same.copy(), same.copy()])
    assert np.max(identical_scores) < 1e-6


def test_state_action_novelty_uses_agent_centric_pairs_and_fifo_archive():
    interface = _fake_interface()
    obs = torch.zeros(4, 3, 2, 2, dtype=torch.long)
    obs[:, 0, 0, 0] = 10
    trajectory = {
        "obs": obs,
        "act": torch.zeros(4, dtype=torch.long),
    }
    pairs = interface._minigrid_state_action_pairs(trajectory)
    assert pairs["local"].shape == (4, 3, 3, 3)
    scores, _ = interface._minigrid_state_action_novelty_batch([pairs, pairs])
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))
    assert len(interface.state_action_archive) == 8


def test_warmup_reward_ignores_world_model_loss():
    interface = _fake_interface()
    warm = interface._minigrid_reward_components(
        raw_loss=0.001, solved=True, is_warmup=True,
        map_novelty=0.0, state_action_novelty=0.0,
        minigrid_reach_ratio=1.0, minigrid_norm_dist=1.0,
    )
    formal = interface._minigrid_reward_components(
        raw_loss=0.001, solved=True, is_warmup=False,
        map_novelty=0.0, state_action_novelty=0.0,
        minigrid_reach_ratio=1.0, minigrid_norm_dist=1.0,
    )
    assert warm["wm_loss"] == 0.0
    assert formal["wm_loss"] > 0.0
