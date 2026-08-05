# Agent-Centric Attentive World Model

This repository contains the Agent-Centric Attentive World Model and its domain adapters for MiniGrid, Crafter, and BipedalWalker.

## Environment Setup

1. **Clone this repository on your machine**:
    ```bash
    git clone https://github.com/Sveali41/Agent-Centric-Attentive-World-Model.git
    ```

2. **Install the requirements inside the cloned folder**:
    ```bash
    pip3 install -r requirements.txt
    ```

3. **Data Collection for Transition Function**:
   - Source the repository-local paths before running experiments:
     ```bash
     source .env
     ```

   - Select the domain in `modelBased/config/config.yaml` (`crafter`, `minigrid`, or `bipedalwalker`).
   - Collect trajectory data for a selected domain:
     ```bash
     python modelBased/data/data_collect.py domain=crafter
     python modelBased/data/data_collect.py domain=bipedalwalker
     ```
   - Save the data into `modelBased/data/train_world_model/`.
     - *(Note: Some collected data may already exist here, which you can use and proceed to the next step.)*

4. **Run the World Model**:
   - Configure the world model in `modelBased/config/config.yaml`.
   - train the model
     ```bash
     python modelBased/world_model/AttentionWM_training.py domain=crafter
     ```
5. **Train the policy based on World Model**:
   - Configure PPO in `modelBased/config/config.yaml`.
   - train the PPO model
   ```bash
   python modelBased/policy_training/PPO_world_training.py domain=bipedalwalker
   ```

6. **Run the trained policy model in the real world**:
   ```bash
   python modelBased/policy_training/PPO_world_test.py domain=minigrid
   ```
   
## Q&A

1. If you encounter issues when importing packages, check the absolute path set in every script (you need to adjust this to your path). 
This is on the TODO list and will be fixed in the future—though I'm unsure how soon :)
   ```bash
   import sys
   sys.path.append('/home/siyao/phd_file/Research/rlPractice/Agent-Centric-Attentive-World-Model')

2. If you encounter issues related to can't find the path, change the path in .env file to your own device.
   ```bash
   export PROJECT_ROOT="/home/siyao/phd_file/Research/rlPractice/Agent-Centric-Attentive-World-Model"
   export TRAIN_DATASET_PATH="${PROJECT_ROOT}/data/train_world_model"
   export PTH_FOLDER="${PROJECT_ROOT}/modelBased/models/ckpt"
   export LOG_FOLDER="${PROJECT_ROOT}/modelBased/models/log"
   PYDEVD_WARN_EVALUATION_TIMEOUT=100.00
