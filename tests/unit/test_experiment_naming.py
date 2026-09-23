import unittest
from pathlib import Path
from types import SimpleNamespace

from modelBased.policy_training.common.experiment_naming import (
    _bounded_wandb_name,
    policy_checkpoint_path,
    policy_selection_paths,
    policy_validation_stem,
)


class WandbNameLimitTest(unittest.TestCase):
    def test_short_name_is_unchanged(self):
        self.assertEqual(_bounded_wandb_name("crafter_policy"), "crafter_policy")

    def test_long_name_is_stable_and_bounded(self):
        name = "crafter_" + "long_variant_" * 20
        bounded = _bounded_wandb_name(name)

        self.assertEqual(len(bounded), 128)
        self.assertEqual(bounded, _bounded_wandb_name(name))
        self.assertTrue(bounded.startswith(name[:100]))

    def test_long_names_with_same_prefix_do_not_collide(self):
        prefix = "crafter_" + "same_prefix_" * 20
        self.assertNotEqual(
            _bounded_wandb_name(prefix + "a"),
            _bounded_wandb_name(prefix + "b"),
        )


class CrafterPolicyCheckpointNameTest(unittest.TestCase):
    @staticmethod
    def _cfg(*, wm_path, explicit_path=None, train_in_real_env=False):
        crafter = SimpleNamespace(
            task_name="crafter_target_task_5",
            planning_world_model_checkpoint=wm_path,
            continual_learning=SimpleNamespace(enabled=False),
            observation_schema=[],
        )
        return SimpleNamespace(
            domain="crafter",
            domains=SimpleNamespace(crafter=crafter),
            PPO=SimpleNamespace(
                train_in_real_env=train_in_real_env,
                checkpoint_path=explicit_path,
                checkpoint_path_wm=wm_path,
                checkpoint_dir="/tmp/policies",
                checkpoint_path_planning="/tmp/policies/legacy_seed1.ckpt",
                checkpoint_path_real_env="/tmp/policies/legacy_realenv_seed1.ckpt",
                seed=1,
                experiment_label="many_suffixes_are_not_used",
                use_main_dense_reward=True,
            ),
            p2e=SimpleNamespace(enabled=False),
            rmax_like=SimpleNamespace(enabled=False),
        )

    def test_planning_name_uses_domain_task_and_wm_seed_only(self):
        cfg = self._cfg(
            wm_path="/tmp/dr_world_model_crafter_none_seed0_best.ckpt"
        )

        self.assertEqual(
            policy_checkpoint_path(cfg, domain="crafter"),
            Path(
                "/tmp/policies/policy_crafter_crafter_target_task_5_seed0.ckpt"
            ),
        )

    def test_explicit_policy_path_still_has_priority(self):
        cfg = self._cfg(
            wm_path="/tmp/dr_world_model_crafter_none_seed0_best.ckpt",
            explicit_path="/tmp/manual_policy.ckpt",
        )

        self.assertEqual(
            policy_checkpoint_path(cfg, domain="crafter"),
            Path("/tmp/manual_policy.ckpt"),
        )

    def test_missing_wm_seed_fails_instead_of_using_ppo_seed(self):
        cfg = self._cfg(wm_path="/tmp/world_model_best.ckpt")

        with self.assertRaisesRegex(ValueError, "include '_seed<N>'"):
            policy_checkpoint_path(cfg, domain="crafter")

    def test_validation_name_matches_loaded_policy(self):
        cfg = self._cfg(
            wm_path="/tmp/dr_attention_world_model_crafter_none_seed0_best.ckpt"
        )

        self.assertEqual(
            policy_validation_stem(cfg, domain="crafter"),
            "policy_crafter_crafter_target_task_5_seed0_test",
        )

    def test_selection_artifacts_keep_canonical_short_name(self):
        paths = policy_selection_paths("/tmp/policies/policy_seed0.ckpt")
        self.assertEqual(paths["canonical"], Path("/tmp/policies/policy_seed0.ckpt"))
        self.assertEqual(paths["best"], Path("/tmp/policies/policy_seed0_best.ckpt"))
        self.assertEqual(paths["last"], Path("/tmp/policies/policy_seed0_last.ckpt"))
        self.assertEqual(paths["manifest"], Path("/tmp/policies/policy_seed0_selection.json"))


if __name__ == "__main__":
    unittest.main()
