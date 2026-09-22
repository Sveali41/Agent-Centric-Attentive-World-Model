"""Focused tests for Crafter's opt-in transition-aware replay."""
import unittest
import numpy as np
from types import SimpleNamespace

from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.data.datamodule import WMRLDataset, WMRLDataModule


def _samples(kinds, nhwc=False):
    n, channels, height, width = len(kinds), 3, 8, 8
    obs = np.zeros((n, channels, height, width), dtype=np.int16)
    nxt = obs.copy()
    obs[:, 0, 1, 1] = 13
    nxt[:, 0, 1, 1] = 13
    inv = np.zeros((n, 16), dtype=np.int16)
    inv_next = inv.copy()
    for i, kind in enumerate(kinds):
        if kind == "move":
            nxt[i, 0, 1, 1] = 0
            nxt[i, 0, 1, 2] = 13
        elif kind == "turn":
            nxt[i, 1, 1, 1] = 1
        elif kind == "map":
            obs[i, 0, 0, 0] = 6
            nxt[i, 0, 0, 0] = 0
        elif kind == "inventory":
            inv_next[i, 4] = 1
            # Acquisition may also remove a resource; inventory has priority.
            obs[i, 0, 0, 0] = 6
            nxt[i, 0, 0, 0] = 0
        elif kind == "survival":
            inv_next[i, 0] = 1
    if nhwc:
        obs, nxt = np.moveaxis(obs, 1, -1), np.moveaxis(nxt, 1, -1)
    return {"obs": obs, "obs_next": nxt, "act": np.arange(n) % 17,
            "inv": inv, "inv_next": inv_next}


class CrafterTransitionReplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"enabled": True, "balance_exponent": 0.5, "include_action": True}

    def test_classification_priority_and_layouts(self):
        data = _samples(["static", "move", "turn", "map", "inventory", "survival"])
        types = FisherReplayBuffer.classify_crafter_transitions(data)
        self.assertEqual(types.tolist(), [3, 2, 2, 1, 0, 3])
        nhwc_types = FisherReplayBuffer.classify_crafter_transitions(
            _samples(["map", "inventory"], nhwc=True)
        )
        self.assertEqual(nhwc_types.tolist(), [1, 0])

    def test_critical_admission_metadata_and_alignment(self):
        data = _samples(["inventory", "map", "move", "static", "static", "move"])
        buffer = FisherReplayBuffer(100, crafter_transition_replay=self.cfg, seed=7)
        buffer.add_from_batch(data, current_sample_ratio=0.5)
        exported = buffer.export_dict()
        self.assertEqual(len(exported["obs"]), 3)
        self.assertTrue({0, 1}.issubset(set(exported["_replay_transition_type"].tolist())))
        self.assertTrue(np.array_equal(
            exported["_replay_bucket"],
            exported["_replay_transition_type"].astype(np.int64) * 17 + exported["act"],
        ))
        restored = FisherReplayBuffer(100, crafter_transition_replay=self.cfg, seed=7)
        restored.load_from_dict(exported)
        self.assertEqual(restored.export_dict()["act"].tolist(), exported["act"].tolist())

    def test_enabled_load_rebuilds_old_replay_metadata(self):
        data = _samples(["inventory", "map", "move", "static"])
        old_buffer = FisherReplayBuffer(100, seed=4)
        old_buffer.add_from_batch(data, current_sample_ratio=1.0)
        legacy_data = old_buffer.export_dict()
        self.assertNotIn("_replay_transition_type", legacy_data)
        enabled = FisherReplayBuffer(100, crafter_transition_replay=self.cfg, seed=4)
        enabled.load_from_dict(legacy_data)
        rebuilt = enabled.export_dict()
        self.assertEqual(sorted(rebuilt["_replay_transition_type"].tolist()), [0, 1, 2, 3])
        self.assertTrue(np.array_equal(
            rebuilt["_replay_bucket"], rebuilt["_replay_transition_type"] * 17 + rebuilt["act"]
        ))

    def test_capacity_retention_is_soft_balanced_and_seeded(self):
        data = _samples(["static"] * 100 + ["inventory"] * 10)
        first = FisherReplayBuffer(10, crafter_transition_replay=self.cfg, seed=9)
        second = FisherReplayBuffer(10, crafter_transition_replay=self.cfg, seed=9)
        first.add_from_batch(data, current_sample_ratio=1.0)
        second.add_from_batch(data, current_sample_ratio=1.0)
        one, two = first.export_dict(), second.export_dict()
        self.assertEqual(len(one["obs"]), 10)
        self.assertTrue(np.array_equal(one["_replay_bucket"], two["_replay_bucket"]))
        self.assertGreater(np.sum(one["_replay_transition_type"] == 0), 0)

    def test_seeded_soft_sampling_and_legacy_mode(self):
        kinds = ["static"] * 100 + ["inventory"] * 4
        data = _samples(kinds)
        one = FisherReplayBuffer(200, crafter_transition_replay=self.cfg, seed=3)
        two = FisherReplayBuffer(200, crafter_transition_replay=self.cfg, seed=3)
        types = one.classify_crafter_transitions(data)
        buckets = one._crafter_buckets(types, data["act"])
        picked_one = one._soft_balanced_indices(np.arange(len(types)), buckets, 20)
        picked_two = two._soft_balanced_indices(np.arange(len(types)), buckets, 20)
        self.assertTrue(np.array_equal(picked_one, picked_two))
        # Natural expectation is <1 rare event; n^-0.5 raises rare-event inclusion.
        self.assertGreater(np.sum(types[picked_one] == 0), 0)
        legacy = FisherReplayBuffer(100, seed=3)
        legacy.add_from_batch(data, current_sample_ratio=0.2)
        self.assertFalse(any("_replay_transition_type" in item for item in legacy.buffer))

    def test_datamodule_uses_persisted_joint_buckets(self):
        data = _samples(["static"] * 100 + ["inventory"] * 4)
        types = FisherReplayBuffer.classify_crafter_transitions(data)
        replay = dict(data)
        replay["_replay_transition_type"] = types
        replay["_replay_bucket"] = types.astype(np.int64) * 17 + replay["act"]
        dataset = WMRLDataset.__new__(WMRLDataset)
        dataset.hparams = SimpleNamespace(
            crafter_transition_replay={"balance_exponent": 0.5, "include_action": True}
        )
        picked = dataset._crafter_soft_replay_indices(replay, 20, np.random.default_rng(3))
        self.assertGreater(np.sum(types[picked] == 0), 0)
        self.assertEqual(sum(v["count"] for v in dataset.replay_sampling_stats.values()), 20)

    def test_inventory_replay_fraction_reserves_real_changes(self):
        data = _samples(["static"] * 100 + ["inventory"] * 20)
        types = FisherReplayBuffer.classify_crafter_transitions(data)
        replay = dict(data)
        replay["_replay_transition_type"] = types
        replay["_replay_bucket"] = types.astype(np.int64) * 17 + replay["act"]
        dataset = WMRLDataset.__new__(WMRLDataset)
        dataset.hparams = SimpleNamespace(crafter_transition_replay={
            "balance_exponent": 0.35,
            "include_action": True,
            "inventory_replay_fraction": 0.25,
        })
        picked = dataset._crafter_soft_replay_indices(
            replay, 40, np.random.default_rng(11)
        )
        # ceil(40 * .25) inventory transitions are reserved before filling
        # the rest from the usual bucket-balanced distribution.
        self.assertGreaterEqual(int(np.sum(types[picked] == 0)), 10)
        self.assertEqual(len(np.unique(picked)), 40)
        self.assertEqual(sum(v["count"] for v in dataset.replay_sampling_stats.values()), 40)

    def test_inventory_replay_fraction_validates_range(self):
        dataset = WMRLDataset.__new__(WMRLDataset)
        dataset.hparams = SimpleNamespace(crafter_transition_replay={
            "inventory_replay_fraction": 1.1,
        })
        data = _samples(["static", "inventory"])
        types = FisherReplayBuffer.classify_crafter_transitions(data)
        replay = dict(data)
        replay["_replay_transition_type"] = types
        replay["_replay_bucket"] = types.astype(np.int64) * 17 + replay["act"]
        with self.assertRaisesRegex(ValueError, "inventory_replay_fraction"):
            dataset._crafter_soft_replay_indices(replay, 1, np.random.default_rng(0))

    def test_missing_inventory_fails_clearly(self):
        data = _samples(["static"])
        del data["inv_next"]
        with self.assertRaisesRegex(ValueError, "inv_next"):
            FisherReplayBuffer.classify_crafter_transitions(data)

    def test_changed_slot_signature_distinguishes_same_action(self):
        data = _samples(["inventory", "inventory", "inventory"])
        data["act"][:] = 5
        # Item slots are inventory indices 4..15: wood=4, diamond=8.
        data["inv_next"][:] = data["inv"]
        data["inv_next"][0, 4] = 1
        data["inv_next"][1, 8] = 1
        data["inv_next"][2, 4] = 1
        data["inv_next"][2, 8] = 1
        cfg = {**self.cfg, "include_changed_slot": True}
        buffer = FisherReplayBuffer(100, crafter_transition_replay=cfg, seed=2)
        buffer.add_from_batch(data, current_sample_ratio=1.0)
        exported = buffer.export_dict()
        self.assertEqual(sorted(exported["_replay_changed_slot_signature"].tolist()), [1, 16, 17])
        self.assertEqual(len(set(exported["_replay_bucket"].tolist())), 3)
        self.assertTrue(np.all(exported["_replay_transition_type"] == 0))

    def test_survival_only_signature_is_zero(self):
        data = _samples(["survival"])
        self.assertEqual(
            FisherReplayBuffer.crafter_changed_slot_signatures(data).tolist(), [0]
        )

    def test_slot_mode_rebuilds_legacy_metadata(self):
        data = _samples(["inventory", "static"])
        cfg = {**self.cfg, "include_changed_slot": True}
        buffer = FisherReplayBuffer(100, crafter_transition_replay=cfg, seed=1)
        buffer.load_from_dict(data)
        exported = buffer.export_dict()
        self.assertIn("_replay_changed_slot_signature", exported)
        self.assertEqual(len(exported["_replay_bucket"]), 2)

    def test_slot_aware_sampling_raises_rare_slot_probability(self):
        data = _samples(["inventory"] * 101)
        data["act"][:] = 5
        data["inv_next"][:] = data["inv"]
        data["inv_next"][:100, 4] = 1  # common wood
        data["inv_next"][100, 8] = 1  # rare diamond
        cfg = {**self.cfg, "include_changed_slot": True}
        first = FisherReplayBuffer(200, crafter_transition_replay=cfg, seed=5)
        second = FisherReplayBuffer(200, crafter_transition_replay=cfg, seed=5)
        types = first.classify_crafter_transitions(data)
        signatures = first.crafter_changed_slot_signatures(data)
        buckets = first._crafter_buckets_with_signatures(types, data["act"], signatures)
        picked_one = first._crafter_soft_balanced_indices(
            np.arange(101), types, data["act"], signatures, buckets, 20
        )
        picked_two = second._crafter_soft_balanced_indices(
            np.arange(101), types, data["act"], signatures, buckets, 20
        )
        self.assertTrue(np.array_equal(picked_one, picked_two))
        self.assertIn(100, picked_one.tolist())

        replay = dict(data)
        replay["_replay_transition_type"] = types
        replay["_replay_changed_slot_signature"] = signatures
        replay["_replay_bucket"] = buckets
        dataset = WMRLDataset.__new__(WMRLDataset)
        dataset.hparams = SimpleNamespace(crafter_transition_replay=cfg)
        dataset._crafter_soft_replay_indices(replay, 20, np.random.default_rng(5))
        sampled_slots = dataset.replay_sampling_stats["inventory_change"]["changed_slots"]
        self.assertIn(8, sampled_slots)  # Absolute inventory index for diamond.
        self.assertEqual(
            sum(value["count"] for value in dataset.replay_sampling_stats.values()), 20
        )

    def test_protected_replay_reuses_multislot_examples_and_is_seeded(self):
        data = _samples(["inventory"] * 5)
        data["act"][:] = 1
        data["inv_next"][:] = data["inv"]
        # One transition covers both rare slots; each slot only needs two.
        data["inv_next"][0, 4] = data["inv_next"][0, 8] = 1
        data["inv_next"][1, 4] = 1
        data["inv_next"][2, 8] = 1
        cfg = {**self.cfg, "include_changed_slot": True,
               "protected_slot_replay": {"enabled": True, "min_samples_per_observed_slot": 2}}
        replay = dict(data)
        types = FisherReplayBuffer.classify_crafter_transitions(data)
        signatures = FisherReplayBuffer.crafter_changed_slot_signatures(data)
        replay.update({"_replay_transition_type": types, "_replay_changed_slot_signature": signatures,
                       "_replay_bucket": types.astype(np.int64) * 17 * (1 << 12) + signatures})
        picks = []
        for seed in (4, 4):
            dataset = WMRLDataset.__new__(WMRLDataset)
            dataset.hparams = SimpleNamespace(crafter_transition_replay=cfg)
            picked = dataset._crafter_soft_replay_indices(replay, 3, np.random.default_rng(seed))
            picks.append(picked)
            self.assertEqual(len(set(picked.tolist())), 3)
            self.assertGreaterEqual(dataset.protected_replay_slot_counts[4], 2)
            self.assertGreaterEqual(dataset.protected_replay_slot_counts[8], 2)
        self.assertTrue(np.array_equal(picks[0], picks[1]))

    def test_fisher_loader_counts_final_unique_samples(self):
        n = 40
        dataset = WMRLDataset.__new__(WMRLDataset)
        dataset.data = {"obs": np.zeros((n, 2, 2, 2)), "obs_next": np.zeros((n, 2, 2, 2)),
                        "act": np.arange(n), "inv": np.zeros((n, 16), dtype=np.int64),
                        "inv_next": np.zeros((n, 16), dtype=np.int64)}
        dataset.data["inv_next"][:2, 4] = 1
        dataset.data["inv_next"][2:4, 8] = 1
        dataset.__getitem__ = lambda index: {key: value[index] for key, value in dataset.data.items()}
        dataset.__len__ = lambda: n
        module = WMRLDataModule.__new__(WMRLDataModule)
        module.cfg = SimpleNamespace(env_type="crafter", seed=2, batch_size=5, n_cpu=0,
            fisher_slot_stratified={"enabled": True, "min_samples_per_observed_slot": 2})
        import torch
        module.data_train = torch.utils.data.Subset(dataset, range(n))
        loader = module.fisher_dataloader(10)
        self.assertEqual(sum(len(batch["act"]) for batch in loader), 10)
        self.assertGreaterEqual(module.fisher_sampling_stats[4], 2)
        self.assertGreaterEqual(module.fisher_sampling_stats[8], 2)


if __name__ == "__main__":
    unittest.main()
