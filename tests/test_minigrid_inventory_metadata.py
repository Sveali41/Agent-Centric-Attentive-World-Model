import numpy as np
import pytest
from torch.utils.data._utils.collate import default_collate

from modelBased.data.datamodule import WMRLDataset


def test_inventory_uses_raw_current_and_next_tokens_from_same_transition():
    info = np.asarray(
        [
            {
                "current_carrying_token": 0,
                "carrying_token": 5,
                "next_carrying_token": 5,
            },
            {
                "current_carrying_token": 5,
                "carrying_token": 0,
                "next_carrying_token": 0,
            },
            {
                "current_carrying_token": 3,
                "carrying_token": 3,
                "next_carrying_token": 3,
            },
        ],
        dtype=object,
    )

    current, following = WMRLDataset._minigrid_inventory_from_info(info)

    np.testing.assert_array_equal(current, np.asarray([0, 5, 3]))
    np.testing.assert_array_equal(following, np.asarray([5, 0, 3]))


def test_inventory_rejects_legacy_metadata_instead_of_shifting_rows():
    legacy_info = np.asarray(
        [{"carrying_key": False}, {"carrying_key": True}], dtype=object
    )

    with pytest.raises(ValueError, match="recollect this dataset"):
        WMRLDataset._minigrid_inventory_from_info(
            legacy_info, done=np.asarray([False, True])
        )


def test_optional_uniform_reset_is_normalized_before_batch_collation():
    raw_info = np.asarray(
        [
            {
                "current_carrying_token": 0,
                "next_carrying_token": 0,
            },
            {
                "current_carrying_token": 0,
                "next_carrying_token": 4,
                "uniform_reset": True,
            },
        ],
        dtype=object,
    )

    normalized = WMRLDataset._normalize_minigrid_info_for_batch(raw_info)
    batch = default_collate(list(normalized))

    assert set(normalized[0]) == set(normalized[1])
    np.testing.assert_array_equal(
        batch["uniform_reset"].numpy(), np.asarray([False, True])
    )
    np.testing.assert_array_equal(
        batch["next_carrying_token"].numpy(), np.asarray([0, 4])
    )
