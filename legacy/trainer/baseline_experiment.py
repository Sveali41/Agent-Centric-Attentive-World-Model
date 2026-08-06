import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modelBased.policy_training import PPO_world_training, PPO_world_test
from modelBased.common.utils import TRAINER_PATH
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_baseline")
def train_policy_on_env(cfg):
    """
    Train the optimal policy on a specific environment file.
    """
    print("++++++++++++++++++++++++++++++++++++ training PPO policy on a single environment... ++++++++++++++++++++++++++++++++++++++++++++++")
    PPO_world_training.run_training_real_env(cfg)

@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_baseline")
def validate_policy_on_env(cfg):
    """
    Validate the trained policy on a specific environment file.
    """
    print("++++++++++++++++++++++++++++++++++++ validating PPO policy on a single environment... ++++++++++++++++++++++++++++++++++++++++++++++")
    PPO_world_test.validate_policy(cfg)




if __name__ == "__main__":
    # train_policy_on_env()
    validate_policy_on_env()
