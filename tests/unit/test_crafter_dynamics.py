"""Focused tests for Crafter's structured inventory transition objective."""

from __future__ import annotations

import unittest
import numpy as np
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from modelBased.world_model.crafter_dynamics import (
    ITEM_DELTA_CLASSES,
    INVENTORY_SLOTS,
    INVENTORY_VALUES,
    balanced_categorical_effect_loss,
    crafter_inventory_gate_loss,
    crafter_inventory_effect_loss,
    categorical_inventory_effect_target,
    decode_crafter_inventory,
    imagined_crafter_step_batch,
    apply_crafter_pose_projection,
    crafter_pose_target,
    decode_crafter_pose_logits,
    categorical_crafter_inventory_event_target,
    CRAFTER_CANONICAL_EVENT_CODEBOOK,
    crafter_event_codebook,
    crafter_inventory_event_residual_loss,
    crafter_player_counts,
)
from modelBased.world_model.AttentionWM import AttentionWorldModel, crafter_selection_loss
from trainer.target_baseline_experiment import _ensure_baseline_csv
from modelBased.world_model.AttentionWM_support import AttentionModule
from domain.crafter.crafter_support import (
    PLAYER_ID,
    crafter_reconstruct_from_logits,
)


def _prediction(
    batch_size: int, *, gate_keep: bool = True, requires_grad: bool = False,
    value_classes: int = INVENTORY_VALUES,
):
    # Survival has KEEP plus SET_TO values 0..9.  All generated current and
    # following survival slots initially stay at zero, so KEEP is correct.
    survival = torch.full((batch_size, 11, 4), -8.0, requires_grad=requires_grad)
    gate = torch.full((batch_size, 2, 12), -8.0, requires_grad=requires_grad)
    value = torch.full((batch_size, value_classes, 12), -8.0, requires_grad=requires_grad)
    with torch.no_grad():
        survival[:, 0] = 8.0
        gate[:, 0 if gate_keep else 1] = 8.0
        value[:, 0] = 8.0
    return {
        "survival_effect_logits": survival,
        "item_gate_logits": gate,
        "item_value_logits": value,
    }


class CrafterInventoryGateLossTest(unittest.TestCase):
    def test_event_codebook_encodes_saturated_crafts(self):
        current = torch.ones(3, 12, dtype=torch.long)
        following = current.clone()
        following[0, 0] -= 1
        following[1, :2] -= 1
        following[2, [0, 2, 3]] -= 1

        target = categorical_crafter_inventory_event_target(current, following)
        self.assertEqual(target.tolist(), [17, 18, 19])
        self.assertEqual(len(CRAFTER_CANONICAL_EVENT_CODEBOOK), 20)

    def test_event_codebook_round_trip_and_invalid_context_mask(self):
        current = torch.zeros(2, 12, dtype=torch.long)
        following = current.clone()
        following[1, 0] = 1
        target = categorical_crafter_inventory_event_target(current, following)
        self.assertEqual(target.tolist(), [0, 16])
        # Event -2 wood is invalid when wood is zero and must never decode.
        from modelBased.world_model.crafter_dynamics import (
            crafter_event_scores_from_slot_logits, decode_crafter_event_scores,
        )
        logits = torch.full((1, 5, 12), -10.0)
        logits[:, 1, 0] = 10.0
        scores, invalid = crafter_event_scores_from_slot_logits(logits, current[:1])
        self.assertTrue(bool(invalid[0, 1]))
        items, event = decode_crafter_event_scores(scores, current[:1])
        self.assertEqual(int(event[0]), 0)
        self.assertTrue(torch.equal(items, current[:1]))

    def test_event_residual_zero_init_and_hard_rows(self):
        module = AttentionModule(
            "discrete", (2, 5, 5), 5, 8, 1, env_type="crafter",
            crafter_output_mode="effect", crafter_inventory_output_mode="categorical_effect",
            crafter_inventory_event_residual_enabled=True,
        )
        self.assertEqual(float(module.crafter_inventory_event_residual[-1].weight.abs().sum()), 0.0)
        current = torch.zeros(2, 16, dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        prediction = {
            "item_event_codebook": crafter_event_codebook(),
            "item_event_change_bias": 1.875,
            "item_event_base_scores": torch.zeros(2, len(CRAFTER_CANONICAL_EVENT_CODEBOOK)),
            "item_event_residual_logits": torch.zeros(
                2, len(CRAFTER_CANONICAL_EVENT_CODEBOOK), requires_grad=True
            ),
        }
        # The second KEEP is a false positive under the deployed decoder.
        prediction["item_event_base_scores"][1, 1] = 3.0
        loss, parts = crafter_inventory_event_residual_loss(prediction, current, following)
        self.assertEqual(int(parts["event_residual_hard_rows"]), 2)
        loss.backward()
        self.assertGreater(float(prediction["item_event_residual_logits"].grad.abs().sum()), 0.0)

    def test_event_residual_trains_on_bias_induced_false_change(self):
        current = torch.zeros(1, 16, dtype=torch.long)
        base_scores = torch.full((1, len(CRAFTER_CANONICAL_EVENT_CODEBOOK)), -5.0)
        base_scores[0, 0] = 2.0
        wood_gain = CRAFTER_CANONICAL_EVENT_CODEBOOK.index(
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
        base_scores[0, wood_gain] = 1.0
        residual = torch.zeros_like(base_scores, requires_grad=True)
        prediction = {
            "item_event_codebook": crafter_event_codebook(),
            "item_event_base_scores": base_scores,
            "item_event_residual_logits": residual,
            "item_event_change_bias": 1.875,
        }
        loss, parts = crafter_inventory_event_residual_loss(
            prediction, current, current
        )
        self.assertEqual(int(parts["event_residual_hard_rows"]), 1)
        self.assertEqual(int(parts["event_residual_false_positive_keep_rows"]), 1)
        loss.backward()
        self.assertGreater(float(residual.grad.abs().sum()), 0.0)

        prediction["item_event_change_bias"] = 0.0
        _, parts = crafter_inventory_event_residual_loss(
            prediction, current, current
        )
        self.assertEqual(int(parts["event_residual_hard_rows"]), 0)

    def test_event_residual_correct_keep_sampling_is_deterministic_and_capped(self):
        current = torch.zeros(12, 16, dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        base_scores = torch.zeros(12, len(CRAFTER_CANONICAL_EVENT_CODEBOOK))
        base_scores[1, 1] = 3.0  # Incorrect change on a true KEEP row.
        residual = torch.zeros_like(base_scores, requires_grad=True)
        prediction = {
            "item_event_codebook": crafter_event_codebook(),
            "item_event_base_scores": base_scores,
            "item_event_residual_logits": residual,
            "item_event_change_bias": 0.0,
        }
        default_loss, default_parts = crafter_inventory_event_residual_loss(
            prediction, current, following
        )
        zero_loss, zero_parts = crafter_inventory_event_residual_loss(
            prediction, current, following, correct_keep_fraction=0.0
        )
        self.assertTrue(torch.equal(default_loss, zero_loss))
        self.assertEqual(int(default_parts["event_residual_hard_rows"]), 2)
        self.assertEqual(int(default_parts["event_residual_selected_correct_keep_rows"]), 0)
        self.assertEqual(int(zero_parts["event_residual_selected_correct_keep_rows"]), 0)

        sampled_loss, sampled_parts = crafter_inventory_event_residual_loss(
            prediction, current, following, correct_keep_fraction=0.5
        )
        self.assertEqual(int(sampled_parts["event_residual_hard_rows"]), 2)
        self.assertEqual(int(sampled_parts["event_residual_selected_correct_keep_rows"]), 2)
        target = categorical_crafter_inventory_event_target(
            current[:, 4:], following[:, 4:]
        )
        expected = torch.nn.functional.cross_entropy(
            base_scores[[0, 1, 2, 7]], target[[0, 1, 2, 7]]
        ) / np.log(base_scores.shape[1])
        self.assertTrue(torch.allclose(sampled_loss, expected))
        sampled_again, _ = crafter_inventory_event_residual_loss(
            prediction, current, following, correct_keep_fraction=0.5
        )
        self.assertTrue(torch.equal(sampled_loss, sampled_again))

    def test_event_residual_no_hard_rows_never_adds_correct_keep(self):
        current = torch.zeros(5, 16, dtype=torch.long)
        prediction = {
            "item_event_codebook": crafter_event_codebook(),
            "item_event_base_scores": torch.zeros(5, len(CRAFTER_CANONICAL_EVENT_CODEBOOK)),
            "item_event_residual_logits": torch.zeros(
                5, len(CRAFTER_CANONICAL_EVENT_CODEBOOK), requires_grad=True
            ),
            "item_event_change_bias": 0.0,
        }
        loss, parts = crafter_inventory_event_residual_loss(
            prediction, current, current, correct_keep_fraction=1.0
        )
        self.assertEqual(int(parts["event_residual_selected_correct_keep_rows"]), 0)
        self.assertEqual(float(loss.detach()), 0.0)
        with self.assertRaisesRegex(ValueError, "correct_keep_fraction"):
            crafter_inventory_event_residual_loss(
                prediction, current, current, correct_keep_fraction=1.1
            )

    def test_event_residual_isolated_from_slot_and_map_heads(self):
        module = AttentionModule(
            "discrete", (2, 5, 5), 5, 8, 1, env_type="crafter",
            crafter_output_mode="effect", crafter_inventory_output_mode="categorical_effect",
            crafter_inventory_event_residual_enabled=True,
        )
        _state, _attn, prediction = module(
            torch.zeros(2, 2, 5, 5), torch.tensor([0, 1]), None,
            inv=torch.zeros(2, 16),
        )
        prediction["item_event_residual_logits"].sum().backward()
        self.assertIsNone(module.inv_head[0].weight.grad)
        self.assertIsNone(module.action_embedding.weight.grad)
        self.assertIsNotNone(module.crafter_inventory_event_residual[-1].weight.grad)
    def test_sqrt_balanced_effect_loss_weights_groups_by_sqrt_count(self):
        # One KEEP cell and four changed cells should contribute in a 1:2
        # group-weight ratio, while each group is first reduced by its mean.
        target = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        logits = torch.zeros((5, 5), dtype=torch.float32)
        logits[0, 0] = 2.0
        logits[1:, 1:] += torch.eye(4) * 2.0
        per_cell = balanced_categorical_effect_loss(
            logits, target, reduction="none"
        )
        expected = (per_cell[0] + 2.0 * per_cell[1:].mean()) / 3.0
        actual = balanced_categorical_effect_loss(
            logits, target, reduction="sqrt_balanced"
        )
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_effect_loss_rejects_unknown_reduction(self):
        with self.assertRaisesRegex(ValueError, "Unsupported categorical-effect reduction"):
            balanced_categorical_effect_loss(
                torch.zeros((1, 5)), torch.zeros((1,), dtype=torch.long),
                reduction="not_a_reduction",
            )

    def test_unified_effect_target_and_decode_use_keep_plus_signed_deltas(self):
        current = torch.zeros((1, 16), dtype=torch.long)
        following = current.clone()
        # Item slots 4..8: KEEP, -4, -2, -1, +1.
        current[0, 5:9] = torch.tensor([4, 2, 1, 0])
        following[0, 5:9] = torch.tensor([0, 0, 0, 1])
        target = categorical_inventory_effect_target(current[:, 4:], following[:, 4:])
        self.assertEqual(target.tolist()[0][:5], [0, 1, 2, 3, 4])
        effect = torch.full((1, 5, 12), -8.0)
        effect.scatter_(1, target.unsqueeze(1), 8.0)
        prediction = {
            "survival_effect_logits": torch.zeros((1, 11, 4)),
            "item_effect_logits": effect,
        }
        prediction["survival_effect_logits"][:, 0] = 8.0
        decoded = decode_crafter_inventory(
            current, prediction, predict_survival=False,
            output_mode="categorical_effect",
        )
        self.assertTrue(torch.equal(decoded[:, 4:], following[:, 4:]))

    def test_unified_effect_loss_has_no_gate_or_value_head_requirement(self):
        current = torch.zeros((2, 16), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        prediction = {
            "survival_effect_logits": torch.zeros((2, 11, 4), requires_grad=True),
            "item_effect_logits": torch.zeros((2, 5, 12), requires_grad=True),
        }
        loss, components = crafter_inventory_effect_loss(
            prediction, current, following, predict_survival=False,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("item_effect", components)
        loss.backward()
        self.assertGreater(float(prediction["item_effect_logits"].grad.abs().sum()), 0.0)

    def test_inventory_effect_focal_is_zero_compatible_and_wired_from_model(self):
        current = torch.zeros((2, 16), dtype=torch.long)
        following = current.clone()
        following[1, 4] = 1
        item_logits = torch.zeros((2, 5, 12))
        item_logits[:, 0] = 4.0  # Easy KEEP predictions; the change stays difficult.
        prediction = {
            "survival_effect_logits": torch.zeros((2, 11, 4)),
            "item_effect_logits": item_logits,
        }
        target = categorical_inventory_effect_target(
            current[:, 4:], following[:, 4:]
        )
        expected_zero = balanced_categorical_effect_loss(
            item_logits, target, reduction="sqrt_balanced", keep_weight=0.95,
            focal_gamma=0.0,
        )
        expected_one = balanced_categorical_effect_loss(
            item_logits, target, reduction="sqrt_balanced", keep_weight=0.95,
            focal_gamma=1.0,
        )
        direct_zero, _ = crafter_inventory_effect_loss(
            prediction, current, following, predict_survival=False,
            effect_reduction="sqrt_balanced", keep_weight=0.95, focal_gamma=0.0,
        )
        direct_one, _ = crafter_inventory_effect_loss(
            prediction, current, following, predict_survival=False,
            effect_reduction="sqrt_balanced", keep_weight=0.95, focal_gamma=1.0,
        )
        torch.testing.assert_close(direct_zero, expected_zero)
        torch.testing.assert_close(direct_one, expected_one)
        self.assertNotAlmostEqual(float(direct_zero), float(direct_one), places=6)

        model = AttentionWorldModel.__new__(AttentionWorldModel)
        torch.nn.Module.__init__(model)
        model.env_type = "crafter"
        model.observation_schema = [{
            "name": "inventory",
            "distribution": "categorical_inventory_effect",
            "prediction_source": "auxiliary",
            "target_source": "inventory",
            "target_mode": "categorical_effect",
        }]
        model.focal_gamma = 1.0
        model.predict_survival = False
        model.crafter_inventory_output_mode = "categorical_effect"
        model.crafter_inventory_value_mode = "categorical_delta"
        model.crafter_inventory_effect_reduction = "sqrt_balanced"
        model.crafter_inventory_effect_keep_weight = 0.95
        model.crafter_inventory_event_residual_enabled = False
        actual, fields = model.observation_loss(
            torch.zeros((2, 1)), torch.zeros((2, 1)),
            aux_pred=prediction, inv=current, inv_next=following,
        )
        torch.testing.assert_close(actual, expected_one)
        torch.testing.assert_close(fields["inventory"], expected_one)

    def test_effect_attention_module_has_104_inventory_outputs(self):
        module = AttentionModule(
            data_type="discrete", grid_shape=(2, 8, 8), mask_size=5,
            embed_dim=8, num_heads=1, env_type="crafter",
            crafter_inventory_output_mode="categorical_effect",
            crafter_inventory_value_mode="categorical_delta",
        )
        self.assertEqual(module.inv_head[-1].out_features, 104)

    def test_crafter_pcgrad_projects_only_conflicting_shared_gradients(self):
        model = AttentionWorldModel.__new__(AttentionWorldModel)
        torch.nn.Module.__init__(model)
        shared = torch.nn.Parameter(torch.zeros(2))
        layout_head = torch.nn.Parameter(torch.zeros(2))
        inventory_head = torch.nn.Parameter(torch.zeros(2))
        layout_grad = torch.tensor([1.0, 0.0])
        inventory_grad = torch.tensor([-1.0, 1.0])
        other_grad = torch.tensor([0.3, 0.2])
        shared.grad = layout_grad + inventory_grad + other_grad
        layout_head.grad = torch.tensor([2.0, 3.0])
        inventory_head.grad = torch.tensor([4.0, 5.0])
        model._pcgrad_task_grads = (
            ((shared, layout_grad, inventory_grad),),
            (layout_grad * inventory_grad).sum(),
            layout_grad.square().sum(),
            inventory_grad.square().sum(),
        )
        model.on_after_backward()
        torch.testing.assert_close(shared.grad, torch.tensor([0.8, 1.7]))
        torch.testing.assert_close(layout_head.grad, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(inventory_head.grad, torch.tensor([4.0, 5.0]))
        self.assertIsNone(model._pcgrad_task_grads)

        aligned = torch.tensor([1.0, 1.0])
        shared.grad = layout_grad + aligned + other_grad
        model._pcgrad_task_grads = (
            ((shared, layout_grad, aligned),),
            (layout_grad * aligned).sum(),
            layout_grad.square().sum(),
            aligned.square().sum(),
        )
        model.on_after_backward()
        torch.testing.assert_close(shared.grad, layout_grad + aligned + other_grad)

    def test_shared_pool_detached_keeps_predictions_and_isolates_inventory_gradients(self):
        kwargs = dict(
            data_type="discrete", grid_shape=(2, 5, 5), mask_size=5,
            embed_dim=8, num_heads=1, env_type="crafter",
            crafter_output_mode="effect", crafter_inventory_output_mode="categorical_effect",
            crafter_inventory_event_residual_enabled=True,
        )
        shared = AttentionModule(**kwargs, crafter_inventory_architecture="shared_pool_v1").eval()
        detached = AttentionModule(**kwargs, crafter_inventory_architecture="shared_pool_detached_v1").eval()
        detached.load_state_dict(shared.state_dict())
        state = torch.zeros((2, 2, 5, 5), dtype=torch.long)
        action = torch.tensor([0, 1])
        inventory = torch.zeros((2, 16))
        shared_map, _, shared_inv = shared(state, action, None, inv=inventory)
        detached_map, _, detached_inv = detached(state, action, None, inv=inventory)
        torch.testing.assert_close(detached_map, shared_map)
        for name in ("item_effect_logits", "item_event_scores"):
            torch.testing.assert_close(detached_inv[name], shared_inv[name])

        (detached_inv["item_effect_logits"].sum()
         + detached_inv["item_event_residual_logits"].sum()).backward()
        self.assertIsNone(detached.conv1.weight.grad)
        self.assertIsNone(detached.action_embedding.weight.grad)
        self.assertIsNone(detached.fc.weight.grad)
        self.assertIsNotNone(detached.inv_head[-1].weight.grad)
        self.assertIsNotNone(detached.crafter_inventory_event_residual[-1].weight.grad)

        detached.zero_grad(set_to_none=True)
        map_logits, _, _ = detached(state, action, None, inv=inventory)
        map_logits.sum().backward()
        self.assertIsNotNone(detached.conv1.weight.grad)
        self.assertIsNone(detached.inv_head[-1].weight.grad)
        self.assertIsNone(detached.crafter_inventory_event_residual[-1].weight.grad)

    def test_isolated_inventory_decoder_shapes_and_gradient_isolation(self):
        module = AttentionModule(
            data_type="discrete", grid_shape=(2, 5, 5), mask_size=5,
            embed_dim=8, num_heads=1, env_type="crafter",
            crafter_inventory_output_mode="categorical_effect",
            crafter_inventory_value_mode="categorical_delta",
            crafter_inventory_architecture="isolated_global_v1",
        )
        state = torch.zeros((2, 2, 5, 5), dtype=torch.long)
        prediction = module(state, torch.tensor([0, 1]), None,
                            inv=torch.zeros((2, 16)))[2]
        self.assertEqual(tuple(prediction["survival_effect_logits"].shape), (2, 11, 4))
        self.assertEqual(tuple(prediction["item_effect_logits"].shape), (2, 5, 12))
        prediction["item_effect_logits"].sum().backward()
        self.assertIsNone(module.action_embedding.weight.grad)
        self.assertIsNone(module.conv1.weight.grad)
        self.assertGreater(float(module.crafter_inventory_head[-1].weight.grad.abs().sum()), 0.0)

    def test_slot_attention_inventory_decoder_shapes_and_is_invalid_with_unknown_name(self):
        module = AttentionModule(
            data_type="discrete", grid_shape=(2, 5, 5), mask_size=5,
            embed_dim=8, num_heads=1, env_type="crafter",
            crafter_inventory_output_mode="categorical_effect",
            crafter_inventory_architecture="slot_attention_v1",
        )
        prediction = module(torch.zeros((1, 2, 5, 5), dtype=torch.long), torch.tensor([0]),
                            None, inv=torch.zeros((1, 16)))[2]
        self.assertEqual(tuple(prediction["item_effect_logits"].shape), (1, 5, 12))
        with self.assertRaisesRegex(ValueError, "crafter_inventory_architecture"):
            AttentionModule("discrete", (2, 5, 5), 5, 8, 1, env_type="crafter",
                            crafter_inventory_architecture="invalid")
        state = torch.zeros((1, 2, 5, 5), dtype=torch.long)
        state[:, 0, 2, 2] = PLAYER_ID
        _, _, inventory = module(state, torch.tensor([0]), None, inv=torch.zeros(1, 16))
        self.assertEqual(tuple(inventory["item_effect_logits"].shape), (1, 5, 12))
        self.assertNotIn("item_gate_logits", inventory)

    def test_effect_schema_builds_full_world_model_with_104_outputs(self):
        hparams = OmegaConf.create({
            "attention_mask_size": 5,
            "grid_shape": [2, 8, 8],
            "lr": 5e-4,
            "wd": 1e-5,
            "visualization": False,
            "visualize_every": 1,
            "data_type": "discrete",
            "env_type": "crafter",
            "predict_survival": False,
            "crafter_pose": {"enabled": False, "mode": "learned_detached"},
            "crafter_inventory_output_mode": "categorical_effect",
            "crafter_inventory_value_mode": "categorical_delta",
            "crafter_inventory_effect_reduction": "balanced_mean",
            "crafter_inventory_gate_reduction": "global_balanced",
            "frame_stack": 1,
            "observation_schema": [
                {"name": "layout_object_effect", "distribution": "categorical_effect",
                 "prediction_slice": [0, 21], "target_index": 0, "classes": 20},
                {"name": "layout_direction_effect", "distribution": "categorical_effect",
                 "prediction_slice": [21, 27], "target_index": 1, "classes": 5},
                {"name": "inventory", "distribution": "categorical_inventory_effect",
                 "prediction_source": "auxiliary", "target_source": "inventory",
                 "target_mode": "categorical_effect", "classes": 10,
                 "effect_classes": 5},
            ],
            "use_bipedal_attention": False,
            "validation_metric": "mse",
            "model_type": "attention",
            "embed_dim": 8,
            "num_heads": 1,
            "obs_norm_values": [20, 5, 9],
            "keep_cell_loss": True,
            "freeze_weight": False,
            "model_save_path": "",
        })
        model = AttentionWorldModel(hparams)
        self.assertEqual(model.model.inv_head[-1].out_features, 104)
        self.assertEqual(
            model.model.checkpoint_contract["inventory"]["effect_values"],
            [-4, -2, -1, 1],
        )
        obs = torch.zeros((1, 2, 8, 8), dtype=torch.long)
        obs[:, 0, 4, 4] = PLAYER_ID
        result = model.validation_step(
            {
                "obs": obs,
                "act": torch.zeros(1, dtype=torch.long),
                "obs_next": obs.clone(),
                "inv": torch.zeros((1, 16), dtype=torch.float32),
                "inv_next": torch.zeros((1, 16), dtype=torch.float32),
            },
            0,
        )
        self.assertTrue(torch.isfinite(result["loss_wm_val"]))
        captured = {}
        model.log = lambda name, value, *args, **kwargs: captured.setdefault(name, value)
        model.on_validation_epoch_end()
        for name in (
            "val/inventory_effect_change_precision",
            "val/inventory_effect_change_recall",
            "val/inventory_effect_false_positive_rate",
            "val/inventory_effect_row_exact",
            "val/item_effect_accuracy_on_changed",
        ):
            self.assertIn(name, captured)
            self.assertTrue(torch.isfinite(captured[name]))
    @staticmethod
    def _attention_module(
        *, value_mode: str = "categorical_absolute"
    ) -> AttentionModule:
        return AttentionModule(
            data_type="discrete",
            grid_shape=(2, 8, 8),
            mask_size=5,
            embed_dim=8,
            num_heads=1,
            env_type="crafter",
            crafter_inventory_value_mode=value_mode,
        )

    def test_pose_flag_off_preserves_three_value_forward_contract(self):
        module = self._attention_module()
        state = torch.zeros((1, 2, 5, 5), dtype=torch.long)
        state[:, 0, 2, 2] = PLAYER_ID
        output = module(state, torch.tensor([0]), None, inv=torch.zeros(1, 16))
        self.assertEqual(len(output), 3)
        self.assertFalse(hasattr(module, "crafter_pose_head"))

    def test_pose_target_round_trip_20_classes(self):
        current = torch.zeros((1, 2, 5, 5), dtype=torch.long)
        current[:, 0, 2, 2] = PLAYER_ID
        for delta_index, (dy, dx) in enumerate(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
            for direction in range(1, 5):
                following = current.clone()
                following[:, 0, 2, 2] = 0
                following[:, 0, 2 + dy, 2 + dx] = PLAYER_ID
                following[:, 1, 2 + dy, 2 + dx] = direction
                target = crafter_pose_target(current, following)
                self.assertEqual(int(target), delta_index * 4 + direction - 1)
                logits = torch.full((1, 20), -10.0)
                logits[0, target] = 10.0
                decoded_delta, decoded_direction = decode_crafter_pose_logits(logits)
                self.assertEqual(tuple(decoded_delta[0].tolist()), (dy, dx))
                self.assertEqual(int(decoded_direction[0]), direction)

    def test_pose_head_gradient_does_not_reach_shared_transformer(self):
        module = AttentionModule(
            data_type="discrete", grid_shape=(2, 8, 8), mask_size=5,
            embed_dim=8, num_heads=1, env_type="crafter",
            crafter_pose_enabled=True,
        )
        state = torch.zeros((2, 2, 5, 5), dtype=torch.long)
        state[:, 0, 2, 2] = PLAYER_ID
        _, _, _, pose_logits = module(
            state, torch.tensor([0, 1]), None, inv=torch.zeros(2, 16), return_pose=True
        )
        gradient = torch.autograd.grad(
            pose_logits.square().mean(), module.fuse_fc.weight, allow_unused=True,
        )[0]
        self.assertTrue(gradient is None or torch.equal(gradient, torch.zeros_like(gradient)))
        pose_logits.square().mean().backward()
        self.assertGreater(float(module.crafter_pose_head[-1].weight.grad.abs().sum()), 0.0)

    def test_pose_projection_ties_exactly_one_player_and_direction(self):
        current = torch.zeros((1, 2, 5, 5), dtype=torch.long)
        current[:, 0, 2, 2] = PLAYER_ID
        decoded = current.clone().float()
        decoded[:, 0, 1, 1] = PLAYER_ID  # deliberately duplicate raw decoder
        decoded[:, 1, 1, 1] = 3
        logits = torch.full((1, 20), -10.0)
        # move right and face direction 4
        logits[0, 4 * 4 + 3] = 10.0
        projected = apply_crafter_pose_projection(decoded, current, logits)
        self.assertEqual(int(projected[:, 0].eq(PLAYER_ID).sum()), 1)
        self.assertEqual(int(projected[0, 0, 2, 3]), PLAYER_ID)
        self.assertEqual(int(projected[0, 1, 2, 3]), 4)
        self.assertEqual(int(projected[0, 1].ne(0).sum()), 1)

    def test_pose_projection_preserves_non_agent_direction(self):
        current = torch.zeros((1, 2, 5, 5), dtype=torch.long)
        current[:, 0, 2, 2] = PLAYER_ID
        decoded = current.clone().float()
        decoded[:, 0, 1, 1] = 15  # zombie
        decoded[:, 1, 1, 1] = 2
        logits = torch.full((1, 20), -10.0)
        logits[0, 0] = 10.0  # stay, direction 1
        projected = apply_crafter_pose_projection(decoded, current, logits)
        self.assertEqual(int(projected[0, 1, 1, 1]), 2)

    def test_pose_suppressed_decode_restores_best_non_player_class(self):
        current = torch.zeros((1, 2, 3, 3), dtype=torch.long)
        current[:, 0, 1, 1] = PLAYER_ID
        # Effect logits: agent is most likely at centre, but grass is the
        # strongest non-agent alternative and must survive pose projection.
        logits = torch.full((1, 27, 3, 3), -10.0)
        logits[:, 0] = 1.0  # KEEP baseline
        logits[:, 1 + PLAYER_ID, 1, 1] = 9.0
        logits[:, 1 + 2, 1, 1] = 8.0  # grass class
        logits[:, 21] = 5.0
        decoded = crafter_reconstruct_from_logits(
            logits, current=current, suppress_agent=True
        )
        self.assertEqual(int(decoded[0, 0, 1, 1]), 2)

    def test_crafter_value_only_detach_keeps_gate_gradient_and_isolates_value(self):
        state = torch.zeros((2, 2, 5, 5), dtype=torch.long)
        state[:, 0, 2, 2] = 1
        state[:, 1, 3, 3] = 2
        action = torch.tensor([0, 8], dtype=torch.long)
        inventory = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.float32)

        module = self._attention_module(value_mode="categorical_delta")
        module.train()
        _, _, prediction = module(state, action, None, inv=inventory)
        shared = module.fuse_fc.weight
        gate_grad = torch.autograd.grad(
            prediction["item_gate_logits"].square().mean(),
            shared,
            retain_graph=True,
            allow_unused=True,
        )[0]
        value_grad = torch.autograd.grad(
            prediction["item_value_logits"].square().mean(),
            shared,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNotNone(gate_grad)
        self.assertGreater(float(gate_grad.abs().sum()), 0.0)
        self.assertTrue(value_grad is None or torch.equal(value_grad, torch.zeros_like(value_grad)))

        total = (
            prediction["item_gate_logits"].square().mean()
            + prediction["item_value_logits"].square().mean()
        )
        total.backward()
        self.assertGreater(float(module.fuse_fc.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.inv_head[2].weight.grad.abs().sum()), 0.0)

    def test_absolute_value_mode_keeps_value_gradient_connected(self):
        state = torch.zeros((2, 2, 5, 5), dtype=torch.long)
        state[:, 0, 2, 2] = 1
        state[:, 1, 3, 3] = 2
        action = torch.tensor([0, 8], dtype=torch.long)
        inventory = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.float32)

        module = self._attention_module(value_mode="categorical_absolute")
        _, _, prediction = module(state, action, None, inv=inventory)
        value_grad = torch.autograd.grad(
            prediction["item_value_logits"].square().mean(),
            module.fuse_fc.weight,
            allow_unused=True,
        )[0]
        self.assertIsNotNone(value_grad)
        self.assertGreater(float(value_grad.abs().sum()), 0.0)

    def test_all_keep_gate_is_penalized_on_rare_changed_slots(self):
        current = torch.zeros((20, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        _, components = crafter_inventory_gate_loss(_prediction(20), current, following)
        # A plain mean would be about 1/240 of the normalized CE here.  The
        # balanced objective gives the one changed group equal influence.
        self.assertGreater(float(components["item_gate"]), 5.0)

    def test_gate_groups_have_equal_weight_despite_keep_count(self):
        losses = []
        for batch_size in (2, 64):
            current = torch.zeros((batch_size, INVENTORY_SLOTS), dtype=torch.long)
            following = current.clone()
            following[0, 4] = 1
            _, components = crafter_inventory_gate_loss(
                _prediction(batch_size), current, following
            )
            losses.append(float(components["item_gate"]))
        self.assertAlmostEqual(losses[0], losses[1], places=5)

    def test_all_keep_batch_is_finite(self):
        current = torch.zeros((4, INVENTORY_SLOTS), dtype=torch.long)
        loss, components = crafter_inventory_gate_loss(
            _prediction(4), current, current.clone()
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(components["item_gate"]))
        self.assertEqual(float(components["item_value"]), 0.0)

    def test_slot_macro_changed_uses_equal_slot_change_means(self):
        current = torch.zeros((4, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        # Slot 4 has three changes; slot 8 has one.  Give their gate logits
        # deliberately different CE values so flat and macro are distinguishable.
        following[:3, 4] = 1
        following[3, 8] = 1
        pred = _prediction(4)
        with torch.no_grad():
            pred["item_gate_logits"][:, 1, 4] = -2.0
            pred["item_gate_logits"][:, 0, 4] = 2.0
            pred["item_gate_logits"][:, 1, 8] = -6.0
            pred["item_gate_logits"][:, 0, 8] = 6.0
        _, global_parts = crafter_inventory_gate_loss(
            pred, current, following, gate_reduction="global_balanced"
        )
        _, macro_parts = crafter_inventory_gate_loss(
            pred, current, following, gate_reduction="slot_macro_changed"
        )
        # The two reductions intentionally differ when changed-slot counts
        # and per-slot CE differ.
        self.assertNotAlmostEqual(float(macro_parts["item_gate"]), float(global_parts["item_gate"]))
        _, all_keep = crafter_inventory_gate_loss(
            pred, current, current.clone(), gate_reduction="slot_macro_changed"
        )
        self.assertTrue(torch.isfinite(all_keep["item_gate"]))

    def test_selection_loss_is_exact_four_term_mean(self):
        self.assertEqual(float(crafter_selection_loss(
            torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), torch.tensor(6.)
        )), 3.0)

    def test_value_loss_matches_legacy_changed_entry_mean(self):
        current = torch.zeros((5, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        following[:, 5] = 1
        prediction = _prediction(5)
        _, components = crafter_inventory_gate_loss(
            prediction, current, following
        )
        expected = torch.nn.functional.cross_entropy(
            prediction["item_value_logits"], following[:, 4:], reduction="none"
        ) / torch.log(torch.tensor(float(INVENTORY_VALUES)))
        changed = following[:, 4:].ne(current[:, 4:])
        self.assertAlmostEqual(
            float(components["item_value"]), float(expected[changed].mean()), places=6
        )

    def test_delta_loss_and_decoder_use_four_delta_classes(self):
        current = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.long)
        current[:, 4] = 4
        following = current.clone()
        following[0, 4] = 3  # -1, delta class 2
        following[1, 4] = 5  # +1, delta class 3
        prediction = _prediction(2, gate_keep=False, value_classes=ITEM_DELTA_CLASSES)
        with torch.no_grad():
            prediction["item_value_logits"][:, :, 0] = -8.0
            prediction["item_value_logits"][0, 2, 0] = 8.0
            prediction["item_value_logits"][1, 3, 0] = 8.0
        _, components = crafter_inventory_gate_loss(
            prediction, current, following, value_mode="categorical_delta"
        )
        self.assertLess(float(components["item_value"]), 1e-4)
        decoded = decode_crafter_inventory(
            current, prediction, value_mode="categorical_delta"
        )
        self.assertTrue(torch.equal(decoded[:, 4], following[:, 4]))

    def test_delta_loss_rejects_unknown_nonzero_transition(self):
        current = torch.zeros((1, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 2
        with self.assertRaisesRegex(ValueError, "unsupported non-zero inventory delta"):
            crafter_inventory_gate_loss(
                _prediction(1, value_classes=ITEM_DELTA_CLASSES),
                current, following, value_mode="categorical_delta",
            )

    def test_delta_value_head_only_receives_changed_slot_gradients(self):
        current = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        prediction = _prediction(
            2, requires_grad=True, value_classes=ITEM_DELTA_CLASSES
        )
        loss, _ = crafter_inventory_gate_loss(
            prediction, current, following, value_mode="categorical_delta"
        )
        loss.backward()
        value_grad = prediction["item_value_logits"].grad
        self.assertGreater(float(value_grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(value_grad[:, :, 1:].abs().sum()), 0.0)

    def test_survival_uses_balanced_effect_reduction(self):
        current = torch.zeros((32, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 0] = 1
        _, components = crafter_inventory_gate_loss(_prediction(32), current, following)
        # An all-KEEP survival head must retain a substantial penalty for the
        # single changed survival slot instead of diluting it across the batch.
        self.assertGreater(float(components["survival"]), 1.0)

    def test_disabled_survival_has_no_head_gradient_and_does_not_weight_loss(self):
        current = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 0] = 1
        following[0, 4] = 3
        prediction = _prediction(2, requires_grad=True)
        loss, components = crafter_inventory_gate_loss(
            prediction, current, following, predict_survival=False
        )
        loss.backward()
        self.assertEqual(float(components["survival"].detach()), 0.0)
        survival_grad = prediction["survival_effect_logits"].grad
        self.assertTrue(
            survival_grad is None or float(survival_grad.abs().sum()) == 0.0
        )
        self.assertGreater(float(prediction["item_gate_logits"].grad.abs().sum()), 0.0)

    def test_value_head_only_receives_changed_item_supervision(self):
        current = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 3
        prediction = _prediction(2, requires_grad=True)
        loss, _ = crafter_inventory_gate_loss(prediction, current, following)
        loss.backward()
        value_grad = prediction["item_value_logits"].grad
        self.assertGreater(float(value_grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(value_grad[:, :, 1:].abs().sum()), 0.0)

    def test_decoder_keeps_or_applies_item_gate_without_change(self):
        current = torch.zeros((1, INVENTORY_SLOTS), dtype=torch.long)
        current[0, 4] = 2
        prediction = _prediction(1, gate_keep=True)
        decoded = decode_crafter_inventory(current, prediction)
        self.assertEqual(int(decoded[0, 4]), 2)
        prediction = _prediction(1, gate_keep=False)
        with torch.no_grad():
            prediction["item_value_logits"][:, :, 0] = -8.0
            prediction["item_value_logits"][:, 7, 0] = 8.0
        decoded = decode_crafter_inventory(current, prediction)
        self.assertEqual(int(decoded[0, 4]), 7)

    def test_disabled_survival_decoder_copies_current_slots(self):
        current = torch.zeros((1, INVENTORY_SLOTS), dtype=torch.long)
        current[0, :4] = torch.tensor([2, 3, 4, 5])
        prediction = _prediction(1)
        with torch.no_grad():
            # Make SET_TO 9 the head's preferred result; disabled decoding
            # must still preserve the current tracker-owned values.
            prediction["survival_effect_logits"][:, 0] = -8.0
            prediction["survival_effect_logits"][:, 10] = 8.0
        decoded = decode_crafter_inventory(current, prediction, predict_survival=False)
        self.assertTrue(torch.equal(decoded[0, :4], current[0, :4]))

    def test_default_decoder_retains_legacy_survival_prediction(self):
        current = torch.zeros((1, INVENTORY_SLOTS), dtype=torch.long)
        prediction = _prediction(1)
        with torch.no_grad():
            prediction["survival_effect_logits"][:, 0] = -8.0
            prediction["survival_effect_logits"][:, 6] = 8.0  # SET_TO 5
        decoded = decode_crafter_inventory(current, prediction)
        self.assertEqual(int(decoded[0, 0]), 5)

    def test_natural_inventory_diagnostics_expose_gate_failure(self):
        current = torch.zeros((2, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 4] = 1
        model = object.__new__(AttentionWorldModel)
        model.env_type = "crafter"
        model.predict_survival = True
        diagnostics = model._crafter_inventory_diagnostics(
            current, following, _prediction(2, gate_keep=True)
        )
        self.assertEqual(float(diagnostics["inventory_gate_change_count"]), 1.0)
        self.assertEqual(float(diagnostics["inventory_gate_true_positive"]), 0.0)
        self.assertEqual(float(diagnostics["inventory_gate_false_positive"]), 0.0)
        self.assertEqual(float(diagnostics["inventory_changed_correct"]), 0.0)

    def test_disabled_survival_diagnostics_only_score_wm_owned_slots(self):
        current = torch.zeros((1, INVENTORY_SLOTS), dtype=torch.long)
        following = current.clone()
        following[0, 0] = 1
        following[0, 4] = 1
        model = object.__new__(AttentionWorldModel)
        model.env_type = "crafter"
        model.predict_survival = False
        diagnostics = model._crafter_inventory_diagnostics(
            current, following, _prediction(1, gate_keep=True)
        )
        self.assertEqual(float(diagnostics["inventory_changed_count"]), 1.0)
        self.assertEqual(float(diagnostics["survival_changed_count"]), 0.0)
        self.assertEqual(float(diagnostics["survival_prediction_enabled"]), 0.0)


class CrafterAgentProjectionTest(unittest.TestCase):
    def _state(self, height=3, width=3):
        state = torch.zeros((1, 2, height, width), dtype=torch.float32)
        state[:, 0, height // 2, width // 2] = PLAYER_ID
        return state

    def test_effect_projection_maps_keep_and_removes_duplicates(self):
        current = self._state()
        logits = torch.full((1, 27, 3, 3), -8.0)
        # KEEP at the current player location must contribute to the absolute
        # next-player probability, and beat a competing SET_TO player cell.
        logits[:, 0, 1, 1] = 9.0
        logits[:, 14, 1, 2] = 8.0  # effect 14 means SET_TO object 13
        logits[:, 14, 0, 0] = 8.0

        raw = crafter_reconstruct_from_logits(logits, current=current)
        constrained = crafter_reconstruct_from_logits(
            logits, current=current, constrain_agent=True
        )
        self.assertGreaterEqual(
            int((raw[:, 0] == PLAYER_ID).sum()), 2,
            "raw effect decoding should expose duplicate-player predictions",
        )
        self.assertEqual(int((constrained[:, 0] == PLAYER_ID).sum()), 1)
        self.assertEqual(int(constrained[0, 0, 1, 1]), PLAYER_ID)

    def test_legacy_absolute_projection_and_raw_decode(self):
        current = self._state()
        logits = torch.full((1, 25, 3, 3), -8.0)
        logits[:, 13, 0, 0] = 8.0
        logits[:, 13, 0, 1] = 8.0
        raw = crafter_reconstruct_from_logits(logits)
        constrained = crafter_reconstruct_from_logits(
            logits, current=current, constrain_agent=True
        )
        self.assertEqual(int((raw[:, 0] == PLAYER_ID).sum()), 2)
        self.assertEqual(int((constrained[:, 0] == PLAYER_ID).sum()), 1)

    def test_imagined_step_returns_exactly_one_player(self):
        states = self._state(height=5, width=5)
        inventory = torch.zeros((1, INVENTORY_SLOTS), dtype=torch.float32)
        actions = torch.zeros((1,), dtype=torch.long)

        class DuplicatePlayerModel:
            def __call__(self, observation, action, reward, *, inv):
                batch, _, height, width = observation.shape
                output = torch.full(
                    (batch, 27, height, width), -8.0, dtype=observation.dtype
                )
                output[:, 14] = 8.0
                return output, None, None

        next_states, _ = imagined_crafter_step_batch(
            DuplicatePlayerModel(), states, actions, inventory,
            attention_mask_size=5,
            inventory_target_mode="categorical_gate",
        )
        self.assertTrue(torch.all(
            (next_states[:, 0] == PLAYER_ID).flatten(1).sum(dim=1) == 1
        ))

    def test_pose_imagined_step_blocks_global_boundary_moves_without_duplication(self):
        """Corner crops use neutral padding and out-of-map moves stay put."""
        states = torch.zeros((5, 2, 7, 7), dtype=torch.long)
        corners = torch.tensor(((0, 0), (0, 6), (6, 0), (6, 6), (0, 3)))
        states[torch.arange(5), 0, corners[:, 0], corners[:, 1]] = PLAYER_ID
        inventory = torch.zeros((5, INVENTORY_SLOTS), dtype=torch.float32)
        # Each action maps to an outward move for its corresponding corner.
        actions = torch.tensor((1, 2, 3, 3, 1), dtype=torch.long)

        class BoundaryPoseModel:
            crafter_pose_enabled = True

            def forward_pose(self, observation, action, reward, *, inv):
                batch, _, height, width = observation.shape
                output = torch.full((batch, 27, height, width), -8.0)
                output[:, 0] = 8.0  # KEEP every object/direction cell.
                pose = torch.full((batch, 20), -8.0)
                # actions 1..4 select up/right/down/left, all direction 1.
                pose_classes = torch.tensor((0, 4, 16, 8, 12))
                pose[torch.arange(batch), pose_classes[action.long()]] = 8.0
                return output, None, None, pose

        next_states, _ = imagined_crafter_step_batch(
            BoundaryPoseModel(), states, actions, inventory,
            attention_mask_size=5,
            inventory_target_mode="categorical_effect",
            predict_survival=False,
        )
        self.assertTrue(torch.equal(crafter_player_counts(next_states), torch.ones(5, dtype=torch.long)))
        self.assertTrue(torch.equal(
            torch.stack([
                torch.nonzero(next_states[index, 0].eq(PLAYER_ID), as_tuple=False)[0]
                for index in range(5)
            ]),
            corners,
        ))



class CrafterFocalDiagnosticsTest(unittest.TestCase):
    def test_effect_head_reports_separate_layout_and_inventory_counts(self):
        # Call the validation-only helper without constructing a Lightning trainer.
        helper = type("CrafterDiagnostics", (), {
            "env_type": "crafter",
            "crafter_inventory_output_mode": "categorical_effect",
            "observation_schema": (
                {"distribution": "categorical_effect", "target_index": 0, "prediction_slice": (0, 21)},
                {"distribution": "categorical_effect", "target_index": 1, "prediction_slice": (21, 27)},
            ),
        })()
        current = torch.zeros((1, 2, 2, 2), dtype=torch.long)
        following = current.clone()
        following[0, 0, 0, 0] = 2
        following[0, 1, 1, 1] = 3
        logits = torch.full((1, 27, 2, 2), -8.0)
        logits[:, 0] = 8.0
        logits[0, 3, 0, 0] = 12.0  # SET_TO object 2
        logits[0, 25, 1, 1] = 12.0  # SET_TO direction 3
        inv = torch.zeros((1, 16), dtype=torch.long)
        inv_next = inv.clone()
        inv_next[0, 4] = 1
        item_logits = torch.full((1, 5, 12), -8.0)
        item_logits[:, 0] = 8.0
        item_logits[0, 4, 0] = 12.0
        diagnostics = AttentionWorldModel._crafter_focal_diagnostics(
            helper, logits, following, current, inv, inv_next,
            {"item_effect_logits": item_logits},
        )
        self.assertEqual(float(diagnostics["layout_object_changed_count"]), 1.0)
        self.assertEqual(float(diagnostics["layout_direction_changed_count"]), 1.0)
        self.assertEqual(float(diagnostics["inventory_changed_count"]), 1.0)
        self.assertEqual(float(diagnostics["inventory_focal_available"]), 1.0)
        self.assertEqual(float(diagnostics["layout_object_false_set_sum"]), 0.0)
        self.assertEqual(float(diagnostics["inventory_false_set_sum"]), 0.0)

    def test_legacy_gate_omits_inventory_focal_metric(self):
        helper = type("CrafterDiagnostics", (), {
            "env_type": "crafter", "crafter_inventory_output_mode": "categorical_gate",
            "observation_schema": (),
        })()
        diagnostics = AttentionWorldModel._crafter_focal_diagnostics(
            helper, torch.zeros((1, 1, 1, 1)), torch.zeros((1, 2, 1, 1), dtype=torch.long),
            torch.zeros((1, 2, 1, 1), dtype=torch.long), torch.zeros((1, 16), dtype=torch.long),
            torch.zeros((1, 16), dtype=torch.long), {},
        )
        self.assertEqual(float(diagnostics["inventory_focal_available"]), 0.0)

class CrafterLearningProgressAggregationTest(unittest.TestCase):
    def test_aggregates_only_finite_identical_probe_pairs(self):
        from generator.generator_interface import GeneratorInterface
        pre, post, progress, paired = GeneratorInterface._aggregate_crafter_learning_progress(
            [1.0, float("nan"), 3.0], [0.5, 0.0, float("nan")]
        )
        self.assertAlmostEqual(pre, 1.0)
        self.assertAlmostEqual(post, 0.5)
        self.assertAlmostEqual(progress, 0.5)
        self.assertEqual(paired.tolist(), [True, False, False])

    def test_no_finite_pair_is_unavailable(self):
        import numpy as np
        from generator.generator_interface import GeneratorInterface
        pre, post, progress, paired = GeneratorInterface._aggregate_crafter_learning_progress(
            [float("nan")], [1.0]
        )
        self.assertTrue(np.isnan(pre))
        self.assertTrue(np.isnan(post))
        self.assertTrue(np.isnan(progress))
        self.assertEqual(paired.tolist(), [False])

class CrafterLearningProgressRewardTest(unittest.TestCase):
    def test_finalization_replaces_provisional_difficulty_rewards(self):
        import numpy as np
        from types import SimpleNamespace
        from generator.generator_interface import GeneratorInterface
        interface = SimpleNamespace(
            is_crafter=True,
            batch_size=3,
            crafter_reward_cfg=SimpleNamespace(learning_progress=2.0, clip=10.0),
            ppo=SimpleNamespace(buffer={"reward": [99.0, 88.0, 77.0]}),
            _pending_crafter_round={
                "valid": [True, True, False],
                "pre_changed_focal_losses": np.array([3.0, np.nan, 1.0]),
                "probe_trajectories": [{}, {}, {}],
                "rewards": [99.0, 88.0, 77.0],
                "auxiliary_rewards": [1.0, 2.0, 3.0],
                "stage_tokens": np.array([0, 0, 0]),
            },
            last_crafter_metrics={},
        )
        interface._evaluate_crafter_changed_focal_losses = lambda trajectories, valid, phase: np.array([2.5, 1.0, np.nan])
        interface._aggregate_crafter_learning_progress = GeneratorInterface._aggregate_crafter_learning_progress
        GeneratorInterface.finalize_crafter_learning_progress(interface, apply_rewards=True)
        # paired: 1 + 2 * (3 - 2.5); valid-unpaired: auxiliary only; invalid: -5.
        self.assertEqual(interface.ppo.buffer["reward"], [2.0, 2.0, -5.0])
        self.assertAlmostEqual(interface.last_crafter_metrics["Pre_Changed_Focal_Loss"], 3.0)
        self.assertAlmostEqual(interface.last_crafter_metrics["Post_Changed_Focal_Loss"], 2.5)
        self.assertAlmostEqual(interface.last_crafter_metrics["Learning_Progress"], 0.5)
        self.assertIsNone(interface._pending_crafter_round)

    def test_dr_finalization_groups_lp_by_random_stage_token(self):
        import numpy as np
        from types import SimpleNamespace
        from generator.generator_interface import GeneratorInterface
        pre = np.arange(1.0, 7.0, dtype=np.float32)
        interface = SimpleNamespace(
            is_crafter=True,
            batch_size=6,
            _pending_crafter_round={
                "valid": [True] * 6,
                "pre_changed_focal_losses": pre,
                "probe_trajectories": [{}] * 6,
                "rewards": [0.0] * 6,
                "auxiliary_rewards": [0.0] * 6,
                "stage_tokens": np.arange(6),
            },
            last_crafter_metrics={},
        )
        interface._evaluate_crafter_changed_focal_losses = (
            lambda trajectories, valid, phase: pre - 0.5
        )
        interface._aggregate_crafter_learning_progress = (
            GeneratorInterface._aggregate_crafter_learning_progress
        )

        GeneratorInterface.finalize_crafter_learning_progress(
            interface, apply_rewards=False
        )

        self.assertAlmostEqual(interface.last_crafter_metrics["Learning_Progress"], 0.5)
        self.assertAlmostEqual(interface.last_crafter_metrics["Inventory_KEEP_Ratio"], 1.0 / 6.0)
        for stage in range(5):
            self.assertEqual(interface.last_crafter_metrics[f"Inventory_Stage_{stage}_Count"], 1)
            self.assertAlmostEqual(interface.last_crafter_metrics[f"Inventory_Stage_{stage}_Mean_LP"], 0.5)


class CrafterGeneratorStageTokenTest(unittest.TestCase):
    def test_stage_token_policy_starts_with_uniform_bias_prior(self):
        from generator.generator_network import MapEditorActorCritic
        model = MapEditorActorCritic(env_type="crafter", initial_inventory_edit_ratio=0.25)
        bias = model.stats_actor[-1].bias.detach()
        torch.testing.assert_close(bias, torch.zeros_like(bias))
        probabilities = torch.softmax(bias, dim=-1)
        torch.testing.assert_close(probabilities, torch.full_like(probabilities, 1.0 / 6.0))

    def test_stage_token_policy_act_and_evaluate_agree(self):
        from generator.generator_network import MapEditorActorCritic
        torch.manual_seed(7)
        model = MapEditorActorCritic(env_type="crafter", initial_inventory_edit_ratio=0.25)
        maps = torch.zeros((3, 3, 4, 4))
        context = torch.zeros((3, 64))
        action, token, map_logp, location_logp, token_logp, _, topk, order, selected = model.act(
            maps, context, torch.zeros((3, 1, 4, 4)), stats_heat=torch.zeros((3, 16))
        )
        self.assertEqual(tuple(token.shape), (3, 1))
        self.assertTrue(torch.all((0 <= token) & (token <= 5)))
        evaluated = model.evaluate(
            maps, context, (action, token), torch.zeros((3, 1, 4, 4)),
            target_topk_mask=topk, target_location_order=order,
            target_stats_topk_mask=selected, stats_heat=torch.zeros((3, 16)),
        )
        self.assertTrue(torch.allclose(token_logp, evaluated[2]))

    def test_stage_inventory_ranges_and_unlock_constraints(self):
        from generator.generator_interface import GeneratorInterface
        from types import SimpleNamespace

        instance = object.__new__(GeneratorInterface)
        instance.cfg = SimpleNamespace(seed=3)
        GeneratorInterface._initialize_crafter_stage_inventory_rng(instance)
        for stage in range(5):
            items = np.stack([
                GeneratorInterface._sample_crafter_stage_inventory(instance, stage)
                for _ in range(32)
            ])
            self.assertEqual(items.shape, (32, 12))
            self.assertTrue(np.all(items[:, :6] >= 0))
            self.assertTrue(np.all(items[:, :6] <= 9))
            self.assertTrue(np.all(items[:, 6:] >= 0))
            self.assertTrue(np.all(items[:, 6:] <= 9))
            if stage == 0:
                np.testing.assert_array_equal(items, 0)
            else:
                unlocked_resources = {
                    1: (0, 5), 2: (0, 1, 2, 5),
                    3: (0, 1, 2, 3, 5), 4: (0, 1, 2, 3, 5),
                }[stage]
                locked_resources = sorted(set(range(6)) - set(unlocked_resources))
                if locked_resources:
                    np.testing.assert_array_equal(items[:, locked_resources], 0)
                if stage == 1:
                    self.assertTrue(np.all(items[:, 0] >= 1))
            np.testing.assert_array_equal(items[:, 4], 0)
            pickaxes = {2: (6,), 3: (6, 7), 4: (6, 7, 8)}.get(stage, ())
            for index in pickaxes:
                self.assertTrue(np.all((items[:, index] >= 1) & (items[:, index] <= 9)))
            locked_pickaxes = sorted(set(range(6, 9)) - set(pickaxes))
            np.testing.assert_array_equal(items[:, locked_pickaxes], 0)
            swords = {1: (9,), 2: (9, 10), 3: (9, 10, 11), 4: (9, 10, 11)}.get(stage, ())
            locked_swords = sorted(set(range(9, 12)) - set(swords))
            np.testing.assert_array_equal(items[:, locked_swords], 0)
            for index in swords:
                self.assertTrue(np.all((items[:, index] >= 0) & (items[:, index] <= 9)))

    def test_stage_inventory_sampling_is_seed_reproducible(self):
        from generator.generator_interface import GeneratorInterface
        from types import SimpleNamespace

        def sample_sequence(seed):
            instance = object.__new__(GeneratorInterface)
            instance.cfg = SimpleNamespace(seed=seed)
            GeneratorInterface._initialize_crafter_stage_inventory_rng(instance)
            return np.stack([
                GeneratorInterface._sample_crafter_stage_inventory(instance, stage)
                for stage in (0, 1, 2, 3, 4, 4, 2)
            ])

        np.testing.assert_array_equal(sample_sequence(17), sample_sequence(17))
        self.assertFalse(np.array_equal(sample_sequence(17), sample_sequence(18)))

    def test_stage_materialization_preserves_survival(self):
        from generator.generator_interface import GeneratorInterface
        from types import SimpleNamespace

        instance = object.__new__(GeneratorInterface)
        instance.cfg = SimpleNamespace(seed=3)
        GeneratorInterface._initialize_crafter_stage_inventory_rng(instance)
        fake = SimpleNamespace(
            is_crafter=True, is_minigrid=False,
            _apply_action=lambda obj, action, mask: (obj, obj, obj),
            _sample_crafter_stage_inventory=lambda stage: GeneratorInterface._sample_crafter_stage_inventory(instance, stage),
        )
        base = np.zeros((1, 16), dtype=np.float32)
        base[0, :4] = [9, 8, 7, 6]
        common = (np.zeros((1, 2, 2), dtype=np.int64), np.zeros((1, 2, 2), dtype=np.int64))
        _, _, _, keep = GeneratorInterface._materialize_candidates(
            fake, *common, np.array([[0]]), base, torch.zeros((1, 1, 2, 2))
        )
        _, _, _, changed = GeneratorInterface._materialize_candidates(
            fake, *common, np.array([[4]]), base, torch.zeros((1, 1, 2, 2))
        )
        np.testing.assert_array_equal(keep[0], base[0])
        np.testing.assert_array_equal(changed[0][:4], base[0][:4])
        self.assertEqual(changed[0][4:16].shape, (12,))

    def test_random_crafter_tokens_are_uniform_and_ignore_ratio(self):
        from generator.random_generator_agent import RandomGeneratorAgent
        random_agent = RandomGeneratorAgent(12, device="cpu", env_type="crafter")
        tokens_by_ratio = []
        for ratio in (0.0, 1.0):
            torch.manual_seed(13)
            result = random_agent.select_action(
                torch.zeros((4096, 3, 3, 3)), None, torch.zeros((4096, 1, 3, 3)),
                max_edits_layout=0.0, max_stats_edit_ratio=ratio,
            )
            token, selected = result[1], result[6]
            self.assertEqual(tuple(token.shape), (4096, 1))
            self.assertTrue(torch.all((0 <= token) & (token <= 5)))
            counts = torch.bincount(token[:, 0], minlength=6)
            self.assertTrue(torch.all(counts > 0))
            self.assertTrue(torch.all(selected == token.ne(0)))
            tokens_by_ratio.append(token)
        torch.testing.assert_close(tokens_by_ratio[0], tokens_by_ratio[1])

    def test_random_bipedal_stats_remain_32_zeroes(self):
        from generator.random_generator_agent import RandomGeneratorAgent
        result = RandomGeneratorAgent(10, device="cpu", env_type="bipedalwalker").select_action(
            torch.zeros((4, 3, 3, 3)), None, torch.zeros((4, 1, 3, 3)), 0.0, 1.0
        )
        self.assertEqual(tuple(result[1].shape), (4, 32))
        self.assertEqual(tuple(result[6].shape), (4, 32))
        self.assertFalse(result[1].any())
        self.assertFalse(result[6].any())

    def test_old_32_key_crafter_action_is_rejected(self):
        from types import SimpleNamespace
        from generator.generator_interface import GeneratorInterface
        fake = SimpleNamespace(is_crafter=True, is_minigrid=False, _apply_action=lambda obj, action, mask: (obj, obj, obj))
        with self.assertRaisesRegex(ValueError, r"must have shape \(1, 1\)"):
            GeneratorInterface._materialize_candidates(
                fake, np.zeros((1, 2, 2), dtype=np.int64), np.zeros((1, 2, 2), dtype=np.int64),
                np.zeros((1, 32), dtype=np.int64), np.zeros((1, 16), dtype=np.float32), torch.zeros((1, 1, 2, 2))
            )

    def test_crafter_fixed_budget_locations_replay_exactly(self):
        from generator.generator_network import MapEditorActorCritic
        torch.manual_seed(11)
        model = MapEditorActorCritic(env_type="crafter")
        maps, context = torch.zeros((1, 3, 4, 4)), torch.zeros((1, 64))
        immutable = torch.zeros((1, 1, 4, 4)); immutable[:, :, 0, 0] = 1
        action, token, map_logp, location_logp, token_logp, _, selected, order, stats_selected = model.act(maps, context, immutable, max_edits=0.25, stats_heat=torch.zeros((1, 16)))
        self.assertEqual(int(selected.sum()), 4)
        self.assertFalse(selected[0, 0, 0])
        self.assertEqual(int(order.max()), 4)
        self.assertEqual(len(torch.unique(order[selected])), 4)
        self.assertTrue(torch.all(action[selected] != 0))
        self.assertTrue(torch.all(action[torch.logical_not(selected)] == 0))
        evaluated = model.evaluate(maps, context, (action, token), immutable, selected, order, stats_selected, torch.zeros((1, 16)))
        self.assertTrue(torch.allclose(location_logp, evaluated[1]))
        self.assertTrue(torch.allclose(map_logp.sum((1, 2)) + location_logp + token_logp, evaluated[0].sum((1, 2)) + evaluated[1] + evaluated[2]))

    def test_random_crafter_uses_fixed_location_budget(self):
        from generator.random_generator_agent import RandomGeneratorAgent
        mask = torch.zeros((2, 1, 4, 4)); mask[:, :, 0, 0] = 1
        result = RandomGeneratorAgent(12, device="cpu", env_type="crafter").select_action(torch.zeros((2, 3, 4, 4)), None, mask, max_edits_layout=0.25)
        self.assertTrue(torch.equal(result[4].sum((1, 2)), torch.tensor([4, 4])))

    def test_path_is_not_a_generator_action_and_multiple_objects_are_applied(self):
        from types import SimpleNamespace
        from generator.generator_interface import GeneratorInterface
        from generator.crafter_env_designer import CRAFTER_OBJ_MAP
        fake = SimpleNamespace(
            is_crafter=True, is_bipedal=False, is_minigrid=False,
            OBJ_START=CRAFTER_OBJ_MAP["agent"], OBJ_EMPTY=CRAFTER_OBJ_MAP["grass"],
            ACTION_TABLE={1: CRAFTER_OBJ_MAP["diamond"], 2: CRAFTER_OBJ_MAP["table"],
                          3: CRAFTER_OBJ_MAP["furnace"]},
        )
        self.assertNotIn(CRAFTER_OBJ_MAP["path"], fake.ACTION_TABLE.values())
        action = np.array([[0, 1, 1, 2], [2, 3, 3, 4], [4, 0, 0, 0]], dtype=np.int64)
        base = np.full((3, 4), CRAFTER_OBJ_MAP["grass"])
        base[0, 0] = CRAFTER_OBJ_MAP["agent"]
        obj, _, _ = GeneratorInterface._apply_action(fake, base, action)
        self.assertEqual(int((obj == CRAFTER_OBJ_MAP["diamond"]).sum()), 2)
        self.assertEqual(int((obj == CRAFTER_OBJ_MAP["table"]).sum()), 2)
        self.assertEqual(int((obj == CRAFTER_OBJ_MAP["furnace"]).sum()), 2)
        self.assertEqual(int((obj == CRAFTER_OBJ_MAP["path"]).sum()), 0)

class CrafterTargetValidationIdentityTest(unittest.TestCase):
    def test_rmax_uniform_metadata_aligns_isolated_validation_config(self):
        import hashlib
        import json
        from unittest.mock import patch

        from omegaconf import OmegaConf
        from modelBased.common.artifacts import dataset_matches
        from trainer.common.utils import validate_on_target_task

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout_path = root / "target.txt"
            layout_path.write_text("AAAAAAAA\n", encoding="utf-8")
            metadata = {
                "domain": "crafter",
                "task_name": "target",
                "env_path": str(layout_path.resolve()),
                "layout_path": str(layout_path.resolve()),
                "layout_hash": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
                "reward_schema": "crafter_native_v1",
                "inventory_encoding": "crafter_inventory_v1",
                "episode_max_steps": 4096,
                "collection_policy": "crafter_rmax_count_local5_inv12_v1",
                "rmax_like": {
                    "count_key_version": "local5_inv12_sa_v1",
                    "mask_size": 5,
                    "death_penalty": 5.0,
                    "train_steps": 360000,
                    "frozen_steps": 40000,
                    "seed": 1,
                },
            }
            dataset_path = root / "target_uniform.npz"
            np.savez(dataset_path, metadata=np.array(json.dumps(metadata), dtype=object))
            cfg = OmegaConf.create({
                "domain": "crafter",
                "attention_model": {
                    "env_type": "crafter",
                    "freeze_weight": False,
                    "keep_cell_loss": False,
                    "use_wandb": False,
                    "target_validation_max_samples": 500,
                    "target_validation_batch_size": 256,
                    "target_validation_seed": 0,
                },
                "domains": {
                    "crafter": {
                        "task_name": "active",
                        "layout_path": str(layout_path.resolve()),
                    },
                },
                "env": {
                    "env_path": str(layout_path.resolve()),
                    "collect": {
                        "data_type": "random",
                        "replace_start_with_empty": False,
                    },
                },
            })

            def fake_train_api(validation_cfg, *args):
                self.assertTrue(dataset_matches(dataset_path, validation_cfg))
                self.assertEqual(validation_cfg.domains.crafter.data_collection.max_steps, 4096)
                self.assertTrue(validation_cfg.rmax_like.enabled)
                return {"avg_val_loss": {"avg_val_loss_wm": 1.0}}, None, object()

            with patch(
                "trainer.common.utils.AttentionWM_training.train_api",
                side_effect=fake_train_api,
            ):
                result = validate_on_target_task(
                    cfg, object(), None, str(root), dataset_path.name
                )
            self.assertEqual(result["avg_val_loss_wm"], 1.0)


class TargetBaselineCsvSchemaTest(unittest.TestCase):
    def test_headered_schema_mismatch_is_backed_up_not_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "summary.csv"
            csv_path.write_text("Seed,Iter\n0,1\n", encoding="utf-8")
            self.assertFalse(_ensure_baseline_csv(csv_path, ["Seed", "Iter", "new_metric"]))
            self.assertFalse(csv_path.exists())
            backups = list(Path(directory).glob("summary_schema_backup*.csv"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "Seed,Iter\n0,1\n")
if __name__ == "__main__":
    unittest.main()
