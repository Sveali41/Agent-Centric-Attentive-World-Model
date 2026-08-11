import numpy as np
import torch

from domain.crafter.crafter_custom_env import CustomCrafterEnv
from domain.crafter.crafter_reward import (
    ACHIEVEMENT_NAMES,
    CrafterAchievementTracker,
    native_reward_batch,
)


def _life_state():
    return {
        "hunger": torch.zeros(1),
        "thirst": torch.zeros(1),
        "fatigue": torch.zeros(1),
        "recover": torch.zeros(1),
        "sleeping": torch.zeros(1, dtype=torch.bool),
    }


def test_tracker_matches_wake_and_interaction_on_one_native_step():
    image = np.zeros((3, 3, 2), dtype=np.int32)
    image[1, 1] = (13, 4)  # player facing right
    image[1, 2, 0] = 1  # water
    next_image = image.copy()
    inventory = np.zeros(16, dtype=np.float32)
    inventory[:4] = (9, 9, 0, 9)
    next_inventory = inventory.copy()
    next_inventory[2] = 1

    tracker = CrafterAchievementTracker(sleeping=True)
    result = tracker.update(image, inventory, 5, next_image, next_inventory)

    assert set(result["events"]) == {"collect_drink", "wake_up"}
    assert set(result["newly_unlocked"]) == {"collect_drink", "wake_up"}
    assert result["reward"] == 1.0


def test_native_batch_tracks_hidden_zombie_hp_when_grid_is_static():
    layout = "GGGGG\nGZGGG\nGGAGG\nGGGGG"
    env = CustomCrafterEnv(
        layout_str=layout,
        max_steps=100,
        seed=1,
        initial_inventory={"wood_sword": 1},
    )
    observation, _ = env.reset(agent_pos=(2, 1), agent_dir=(-1, 0))
    states = torch.as_tensor(
        np.transpose(observation["image"], (2, 0, 1))
    ).unsqueeze(0)
    inventories = torch.as_tensor(observation["inventory"]).unsqueeze(0)
    unlocked = torch.zeros((1, len(ACHIEVEMENT_NAMES)), dtype=torch.bool)
    life = _life_state()
    entity_hp = torch.full_like(states[:, 0], -1.0)
    entity_hp[states[:, 0] == 15] = 5.0

    rewards = []
    for _ in range(3):
        next_observation, native_reward, _, _, info = env.step(5)
        next_states = torch.as_tensor(
            np.transpose(next_observation["image"], (2, 0, 1))
        ).unsqueeze(0)
        next_inventories = torch.as_tensor(
            next_observation["inventory"]
        ).unsqueeze(0)
        batch = native_reward_batch(
            states,
            inventories,
            torch.tensor([5]),
            next_states,
            next_inventories,
            unlocked,
            life["sleeping"],
            entity_hp,
            life,
        )
        rewards.append((float(native_reward), float(batch[0][0])))
        states, inventories, unlocked, life, entity_hp = (
            next_states,
            next_inventories,
            batch[2],
            batch[3],
            batch[4],
        )

    env.close()
    assert rewards == [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
    assert info["newly_unlocked"] == ["defeat_zombie"]
