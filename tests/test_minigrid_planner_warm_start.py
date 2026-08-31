import json

import numpy as np
from minigrid.wrappers import FullyObsWrapper

from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from modelBased.policy_training.dijkstra_planner import (
    GraphPlanner,
    _json_default,
    plan_exact_minigrid,
    replay_actions_in_real_env,
    run_receding_horizon_episode,
)


LAYOUT = """
WWWWW
WSOGW
WWWWW
"""

COLORS = """
WWWWW
WERGW
WWWWW
"""


def test_six_action_astar_path_executes_and_reaches_goal():
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            layout_str=LAYOUT,
            color_str=COLORS,
            agent_start_dir=0,
            max_steps=32,
        )
    )
    env.reset(seed=0)

    actions, diagnostics = plan_exact_minigrid(env)

    assert diagnostics["found"]
    assert actions == [4, 2, 2]
    assert all(0 <= action < 6 for action in actions)

    env.reset(seed=0)
    replay = replay_actions_in_real_env(env, actions)
    assert replay["reached_goal"]
    assert len(replay["states"]) == len(actions)
    assert replay["states"][0].shape == (3, 3, 5)


class _ScriptedReplanningPlanner:
    def __init__(self):
        self.plans = [[4, 2, 2], [2, 2], [2]]
        self.calls = 0
        self.last_plan_diagnostics = {}

    @staticmethod
    def _state_key(state, inventory_token=0):
        positions = np.argwhere(np.asarray(state)[0] == 10)
        y, x = map(int, positions[0])
        direction = int(np.asarray(state)[2, y, x])
        return inventory_token, y, x, direction

    def plan(self, *_args, **_kwargs):
        plan = self.plans[self.calls] if self.calls < len(self.plans) else []
        self.calls += 1
        self.last_plan_diagnostics = {
            "found": bool(plan),
            "path_length": len(plan),
            "expansions": 1,
            "visited_states": 2,
        }
        return plan

    @staticmethod
    def predict_transition(full_obs, inventory_token, _action):
        return np.asarray(full_obs).copy(), int(inventory_token)


def test_receding_horizon_executes_one_action_per_plan(tmp_path):
    layout_path = tmp_path / "tiny_replanning.txt"
    layout_path.write_text(f"{LAYOUT.strip()}\n\n{COLORS.strip()}\n", encoding="utf-8")
    planner = _ScriptedReplanningPlanner()

    result, trace = run_receding_horizon_episode(
        planner=planner,
        layout_path=str(layout_path),
        direction=0,
        seed=0,
        max_steps=8,
        max_depth=8,
        max_expansions=100,
        progress_every_expansions=0,
        gif_path=None,
        gif_fps=10,
        print_every_steps=0,
    )

    assert result["real_reached_goal"]
    assert result["real_executed_steps"] == 3
    assert result["replans"] == 3
    assert [item["action"] for item in trace] == [4, 2, 2]
    assert planner.calls == 3


def test_planner_json_supports_numpy_scalars():
    payload = {"position": [np.int64(2), np.int64(3)]}
    assert json.loads(json.dumps(payload, default=_json_default)) == {
        "position": [2, 3]
    }


def test_world_model_astar_recognizes_lava_under_agent_token():
    initial = np.ones((3, 3, 4), dtype=np.int16)
    initial[:, 1, 1] = (10, 0, 0)
    initial[:, 1, 2] = (9, 0, 0)
    planner = GraphPlanner(None, 6, 3, [], [], [])
    planner._set_reference_layout(initial)

    predicted = initial.copy()
    predicted[:, 1, 1] = (1, 0, 0)
    predicted[:, 1, 2] = (10, 0, 0)

    assert planner._is_terminal_failure_state(predicted)
    assert not planner._is_terminal_failure_state(initial)
