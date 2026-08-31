from types import SimpleNamespace

import torch

from modelBased.policy_training.mpc_planner import WorldModelMPC
from modelBased.policy_training.mcts_planner import WorldModelMCTS


class ForwardOnlyWorldModel(torch.nn.Module):
    """Small absolute-transition WM for validating MPC without an environment."""

    minigrid_transition_mode = "absolute"

    def forward(self, state, actions, info=None, inv=None):
        batch, _, rows, columns = state.shape
        object_ids = state[:, 0].long().clone()
        color_ids = state[:, 1].long().clone()
        state_ids = state[:, 2].long().clone()
        center_y, center_x = rows // 2, columns // 2
        forward = actions.eq(2)
        object_ids[forward, center_y, center_x] = 1
        object_ids[forward, center_y, center_x + 1] = 10
        state_ids[forward, center_y, center_x] = 0
        state_ids[forward, center_y, center_x + 1] = 0

        logits = torch.full((batch, 21, rows, columns), -30.0, device=state.device)
        for offset, values, classes in ((0, object_ids, 11), (11, color_ids, 6), (17, state_ids, 4)):
            logits[:, offset : offset + classes].scatter_(1, values.unsqueeze(1), 30.0)
        inventory_logits = torch.full((batch, 7), -30.0, device=state.device)
        inventory_logits.scatter_(1, inv.long().unsqueeze(1), 30.0)
        return logits, None, inventory_logits
def test_mpc_uses_only_wm_rollouts_and_selects_forward_progress():
    torch.manual_seed(3)
    config = SimpleNamespace(
        horizon=2,
        population=256,
        elite_count=32,
        iterations=3,
        gamma=0.99,
        goal_reward=100.0,
        progress_reward=1.0,
        step_penalty=0.01,
        invalid_action_penalty=0.1,
        uncertainty_penalty=0.0,
        fallback_action=0,
    )
    state = torch.zeros(3, 5, 5)
    state[0, :, :] = 1
    state[0, 2, 1] = 10
    state[0, 2, 3] = 8
    planner = WorldModelMPC(ForwardOnlyWorldModel(), attention_mask_size=3, config=config)

    plan = planner.plan(state, inventory_token=0, goal_yx=(2, 3))

    assert plan.actions[0] == 2
    assert plan.diagnostics["imagined_transitions"] == 256 * 2 * 3
    assert plan.predicted_next_inventory == 0
    assert tuple(torch.nonzero(plan.predicted_next_state[0] == 10)[0].tolist()) == (2, 2)


def test_mcts_expands_only_wm_states_and_selects_forward_progress():
    torch.manual_seed(3)
    config = SimpleNamespace(
        simulations=64,
        max_depth=4,
        rollout_depth=0,
        exploration_constant=1.25,
        gamma=0.99,
        goal_reward=100.0,
        progress_reward=1.0,
        inventory_change_bonus=0.0,
        door_open_bonus=0.0,
        step_penalty=0.01,
        invalid_action_penalty=0.1,
        uncertainty_penalty=0.0,
        heuristic_weight=0.05,
        fallback_action=0,
    )
    state = torch.zeros(3, 5, 5)
    state[0, :, :] = 1
    state[0, 2, 1] = 10
    state[0, 2, 3] = 8
    planner = WorldModelMCTS(ForwardOnlyWorldModel(), attention_mask_size=3, config=config)

    plan = planner.plan(state, inventory_token=0, goal_yx=(2, 3))

    assert plan.actions[0] == 2
    assert plan.diagnostics["root_visits"] == 64
    assert plan.diagnostics["tree_nodes"] >= 7
    assert plan.diagnostics["imagined_transitions"] >= 6
    assert plan.predicted_next_inventory == 0
    assert tuple(torch.nonzero(plan.predicted_next_state[0] == 10)[0].tolist()) == (2, 2)


def test_mcts_uses_native_goal_reward_and_lava_terminal():
    config = SimpleNamespace(
        simulations=1,
        max_depth=1,
        rollout_depth=0,
        exploration_constant=1.25,
        gamma=0.99,
        fallback_action=0,
    )
    planner = WorldModelMCTS(ForwardOnlyWorldModel(), attention_mask_size=3, config=config)
    state = torch.zeros(3, 5, 5)
    state[0, :, :] = 1
    state[0, 2, 2] = 10
    state[2, 2, 2] = 0  # facing right
    state[0, 2, 3] = 9  # MiniGrid lava object ID

    assert planner._native_goal_reward(10, 100) == 0.91
    assert planner._forward_enters_lava(state, action=2)
    assert not planner._forward_enters_lava(state, action=0)
