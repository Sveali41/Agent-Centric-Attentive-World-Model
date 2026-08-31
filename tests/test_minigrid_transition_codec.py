import torch

from domain.minigrid.transition_codec import (
    MINIGRID_FIELDS,
    canonical_minigrid_schema,
    decode_minigrid_transition,
    effect_target,
    minigrid_contract,
    minigrid_effect_cell_nll,
    validate_schema,
)


def _perfect_effect_logits(current, following):
    contract = minigrid_contract("effect")
    logits = torch.full(
        (current.shape[0], contract.output_channels, current.shape[2], current.shape[3]),
        -20.0,
    )
    for name, target_index, _ in MINIGRID_FIELDS:
        start, stop = contract.spatial_slices[name]
        labels = effect_target(current[:, target_index], following[:, target_index])
        logits[:, start:stop].scatter_(1, labels.unsqueeze(1), 20.0)
    return logits


def test_schema_and_contract_are_canonical():
    validate_schema(canonical_minigrid_schema("effect"), "effect")
    natural_schema = canonical_minigrid_schema("effect", "mean")
    validate_schema(natural_schema, "effect", "mean")
    assert all(field["effect_reduction"] == "mean" for field in natural_schema)
    assert minigrid_contract("effect").output_channels == 24
    assert minigrid_contract("effect").inventory_output_classes == 8
    assert minigrid_contract("absolute").output_channels == 21


def test_effect_round_trip_including_agent_and_inventory():
    current = torch.zeros(2, 3, 3, 3, dtype=torch.long)
    following = current.clone()
    current[:, 0, 1, 1] = 10
    following[:, 0, 1, 1] = 1
    following[:, 0, 1, 2] = 10
    following[:, 1, 1, 2] = 4
    logits = _perfect_effect_logits(current, following)
    inventory_logits = torch.full((2, 8), -20.0)
    inventory_logits[:, 1] = 20.0  # SET_TO_EMPTY
    decoded, decoded_inventory, diagnostics = decode_minigrid_transition(
        logits,
        current,
        inventory_logits,
        torch.ones(2, dtype=torch.long),
        mode="effect",
        constrain_agent=True,
    )
    assert torch.equal(decoded, following)
    assert torch.equal(decoded_inventory, torch.zeros(2, dtype=torch.long))
    assert torch.equal(diagnostics["raw_agent_count"], torch.ones(2, dtype=torch.long))


def test_effect_cell_nll_is_finite_and_spatial():
    current = torch.zeros(1, 3, 3, 3, dtype=torch.long)
    following = current.clone()
    current[:, 0, 1, 1] = 10
    following[:, 0, 1, 2] = 10
    logits = _perfect_effect_logits(current, following)
    cell_map, field_maps = minigrid_effect_cell_nll(
        logits, current, following, mode="effect"
    )
    assert cell_map.shape == (1, 3, 3)
    assert set(field_maps) == {"object", "color", "state"}
    assert torch.isfinite(cell_map).all()


def test_constrained_decode_repairs_missing_and_duplicate_agent():
    current = torch.zeros(1, 3, 3, 3, dtype=torch.long)
    current[:, 0, 1, 1] = 10
    contract = minigrid_contract("effect")
    logits = torch.full((1, contract.output_channels, 3, 3), -10.0)
    start, stop = contract.spatial_slices["object"]
    # Make two cells prefer SET_TO_AGENT and no cell retain a strong agent.
    logits[:, start + 11, 0, 0] = 10.0
    logits[:, start + 11, 2, 2] = 9.0
    decoded, _, diagnostics = decode_minigrid_transition(
        logits,
        current,
        torch.zeros(1, 8),
        torch.zeros(1, dtype=torch.long),
        mode="effect",
        constrain_agent=True,
    )
    assert int((decoded[:, 0] == 10).sum()) == 1
    assert diagnostics["agent_correction"].item()
