"""Focused compatibility tests for checkpoint-authoritative Crafter planning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from domain.crafter.crafter_support import (
    inspect_crafter_planning_checkpoint,
    load_crafter_planning_model,
)
from modelBased.world_model.AttentionWM_support import AttentionModule


class CrafterPlanningCheckpointTest(unittest.TestCase):
    def _checkpoint(self, directory: Path, *, value_mode: str, pose: bool = False,
                    contract: dict | None = None) -> Path:
        model = AttentionModule(
            "discrete", (2, 8, 8), 5, 8, 1, env_type="crafter",
            crafter_output_mode="effect", crafter_inventory_output_mode="categorical_gate",
            crafter_inventory_classes=10, crafter_inventory_value_mode=value_mode,
            crafter_pose_enabled=pose,
        )
        path = directory / f"{value_mode}_{pose}.ckpt"
        torch.save({
            "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
            "hyper_parameters": {
                "env_type": "crafter", "model_type": "Attention", "data_type": "discrete",
                "grid_shape": [2, 8, 8], "attention_mask_size": 5, "embed_dim": 8,
                "num_heads": 1, "frame_stack": 1, "action_norm_values": 17,
                "obs_norm_values": [20, 5, 0], "predict_survival": False,
                "crafter_inventory_value_mode": value_mode,
                "crafter_pose": {"enabled": pose, "mode": "learned_detached"},
            },
            **({"world_model_contract": contract} if contract is not None else {}),
        }, path)
        return path

    @staticmethod
    def _valid_contract(*, value_mode: str = "categorical_delta", pose: bool = False) -> dict:
        return {
            "version": "crafter_planning_v1", "domain": "crafter", "data_type": "discrete",
            "grid_shape": [2, 8, 8], "frame_stack": 1, "attention_mask_size": 5,
            "embed_dim": 8, "num_heads": 1, "output_mode": "effect", "action_count": 17,
            "inventory": {"output_mode": "categorical_gate", "classes": 10,
                          "value_mode": value_mode},
            "pose": {"enabled": pose, "mode": "learned_detached"},
            "predict_survival": False, "obs_norm_values": [20, 5, 0],
        }

    def test_legacy_delta_spec_and_strict_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self._checkpoint(Path(temp), value_mode="categorical_delta", pose=True)
            spec = inspect_crafter_planning_checkpoint(path)
            self.assertEqual(spec.inventory_value_mode, "categorical_delta")
            self.assertEqual(spec.attention_mask_size, 5)
            self.assertTrue(spec.pose_enabled)
            loaded, loaded_spec = load_crafter_planning_model(path)
            self.assertEqual(len(loaded.state_dict()), 54)
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            self.assertEqual(loaded_spec, spec)

    def test_absolute_width_is_recognised(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self._checkpoint(Path(temp), value_mode="categorical_absolute")
            self.assertEqual(
                inspect_crafter_planning_checkpoint(path).inventory_value_mode,
                "categorical_absolute",
            )

    def test_slot_attention_contract_is_strict_and_legacy_defaults_shared_pool(self):
        with tempfile.TemporaryDirectory() as temp:
            model = AttentionModule(
                "discrete", (2, 8, 8), 5, 8, 1, env_type="crafter",
                crafter_output_mode="effect", crafter_inventory_output_mode="categorical_effect",
                crafter_inventory_value_mode="categorical_delta",
                crafter_inventory_architecture="slot_attention_v1",
            )
            contract = dict(model.checkpoint_contract)
            contract.update({"predict_survival": False, "obs_norm_values": [20, 5, 0]})
            path = Path(temp) / "slot_attention.ckpt"
            torch.save({"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
                        "world_model_contract": contract}, path)
            spec = inspect_crafter_planning_checkpoint(path)
            self.assertEqual(spec.inventory_architecture, "slot_attention_v1")
            loaded, _ = load_crafter_planning_model(path)
            self.assertEqual(len(loaded.state_dict()), len(model.state_dict()))

    def test_event_residual_contract_strictly_reconstructs_decoder(self):
        with tempfile.TemporaryDirectory() as temp:
            model = AttentionModule(
                "discrete", (2, 8, 8), 5, 8, 1, env_type="crafter",
                crafter_output_mode="effect", crafter_inventory_output_mode="categorical_effect",
                crafter_inventory_value_mode="categorical_delta",
                crafter_inventory_event_residual_enabled=True,
                crafter_inventory_event_residual_hidden_dims=(64,),
                crafter_inventory_event_residual_action_embed_dim=8,
                crafter_inventory_event_residual_change_bias=1.875,
            )
            contract = dict(model.checkpoint_contract)
            contract.update({"predict_survival": False, "obs_norm_values": [20, 5, 0]})
            path = Path(temp) / "event_residual.ckpt"
            torch.save({"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
                        "world_model_contract": contract}, path)
            spec = inspect_crafter_planning_checkpoint(path)
            self.assertTrue(spec.inventory_event_residual_enabled)
            self.assertEqual(spec.inventory_event_residual_hidden_dims, (64,))
            loaded, _ = load_crafter_planning_model(path)
            self.assertIn("crafter_inventory_event_codebook", loaded.state_dict())

    def test_valid_world_model_contract_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self._checkpoint(
                Path(temp), value_mode="categorical_delta", pose=True,
                contract=self._valid_contract(pose=True),
            )
            raw = torch.load(path, weights_only=False)
            raw["hyper_parameters"]["crafter_inventory_value_mode"] = "categorical_absolute"
            torch.save(raw, path)
            spec = inspect_crafter_planning_checkpoint(path)
            self.assertEqual(spec.inventory_value_mode, "categorical_delta")
            self.assertTrue(spec.pose_enabled)

    def test_legacy_metadata_rejects_non_crafter_missing_survival_and_bad_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for field, value, message in (
                ("env_type", "minigrid", "not for env_type='crafter'"),
                ("predict_survival", None, "lacks required predict_survival"),
                ("attention_mask_size", 7, "attention_mask_size"),
            ):
                path = self._checkpoint(directory, value_mode="categorical_delta")
                raw = torch.load(path, weights_only=False)
                if value is None:
                    del raw["hyper_parameters"][field]
                else:
                    raw["hyper_parameters"][field] = value
                torch.save(raw, path)
                with self.assertRaisesRegex(ValueError, message):
                    inspect_crafter_planning_checkpoint(path)

    def test_legacy_metadata_rejects_non17_action_and_pose_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = self._checkpoint(directory, value_mode="categorical_delta")
            raw = torch.load(path, weights_only=False)
            raw["hyper_parameters"]["action_norm_values"] = 16
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, "17 actions"):
                inspect_crafter_planning_checkpoint(path)

            path = self._checkpoint(directory, value_mode="categorical_delta", pose=True)
            raw = torch.load(path, weights_only=False)
            raw["hyper_parameters"]["crafter_pose"]["enabled"] = False
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, "pose metadata"):
                inspect_crafter_planning_checkpoint(path)

    def test_pure_state_dict_and_contract_conflict_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = self._checkpoint(directory, value_mode="categorical_delta")
            raw = torch.load(path, weights_only=False)
            pure = directory / "pure.ckpt"
            torch.save(raw["state_dict"], pure)
            with self.assertRaisesRegex(ValueError, "pure state_dict"):
                inspect_crafter_planning_checkpoint(pure)
            raw["world_model_contract"] = self._valid_contract(
                value_mode="categorical_absolute"
            )
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, "inventory mode"):
                inspect_crafter_planning_checkpoint(path)

    def test_isolated_architecture_rejects_legacy_gate_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            model = AttentionModule(
                "discrete", (2, 8, 8), 5, 8, 1, env_type="crafter",
                crafter_output_mode="effect", crafter_inventory_output_mode="categorical_effect",
                crafter_inventory_value_mode="categorical_delta",
                crafter_inventory_architecture="isolated_global_v1",
            )
            contract = dict(model.checkpoint_contract)
            contract["predict_survival"] = False
            contract["obs_norm_values"] = [20, 5, 0]
            contract["inventory"] = dict(contract["inventory"])
            contract["inventory"]["output_mode"] = "categorical_gate"
            path = Path(temp) / "isolated_gate.ckpt"
            torch.save({
                "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
                "world_model_contract": contract,
            }, path)
            with self.assertRaisesRegex(ValueError, "Isolated Crafter inventory architectures"):
                inspect_crafter_planning_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
