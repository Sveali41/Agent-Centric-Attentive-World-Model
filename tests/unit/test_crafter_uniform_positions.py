"""Focused tests for legal and exhaustive Crafter uniform spawn scheduling."""

from pathlib import Path

import numpy as np

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from modelBased.data.data_collect import (
    _crafter_uniform_reset_interval,
    _crafter_uniform_reset_kwargs,
    _crafter_uniform_spawn_positions,
)


_LAYOUT = (
    Path(__file__).resolve().parents[3]
    / "trainer/level/crafter/target_tasks/crafter_target_task_1.txt"
)


def test_uniform_spawns_are_native_legal_and_teleportable():
    env = CustomCrafterEnv(
        txt_file_path=str(_LAYOUT), max_steps=20, seed=7, ai_enabled=False
    )
    try:
        env.reset()
        player = env.env._player
        world = env.env._world
        default_spawn = tuple(map(int, player.pos))
        spawn_points = _crafter_uniform_spawn_positions(env)

        assert default_spawn in spawn_points
        assert spawn_points
        for point in spawn_points:
            material, occupant = world[point]
            assert material in player.walkable
            assert occupant is None or point == default_spawn

        for point in (default_spawn, spawn_points[0], spawn_points[-1]):
            env.reset(**_crafter_uniform_reset_kwargs(point, default_spawn))
            assert tuple(map(int, env.env._player.pos)) == point
    finally:
        env.close()


def test_25000_transition_budget_completes_a_legal_spawn_pass():
    spawn_count = 803
    interval = _crafter_uniform_reset_interval(25000, spawn_count, 5000)

    assert interval == 31
    assert int(np.ceil(25000 / interval)) >= spawn_count
