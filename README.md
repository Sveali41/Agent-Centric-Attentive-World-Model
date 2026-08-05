# Agent-Centric Attentive World Model

## How to Reproduce the Results

1. **Clone this repository on your machine**:
    ```bash
    git clone https://github.com/Sveali41/Agent-Centric-Attentive-World-Model.git
    ```

2. **Install the requirements inside the cloned folder**:
    ```bash
    pip3 install -r requirements.txt
    ```

3. **Data Collection for Transition Function**:
   - You can personalize the parameters you prefer in the config file located at `/MiniGrid/modelBased/conf/env/config.yaml`.
   - Randomly run the environment and collect the trajectory data (observation, action, observation_next, reward, done):
     ```bash
     python /modelBased/modelBased/data_collect.py
     ```
   - Save the data into `MiniGrid/modelBased/data/train_world_model/gridworld_full.npz`.
     - *(Note: Some collected data may already exist here, which you can use and proceed to the next step.)*

4. **Run the World Model**:
   - You can personalize the parameters you prefer in the config file located at `/MiniGrid/modelBased/conf/model/config.yaml`.
   - train the model
     ```bash
     python /modelBased/modelBased/world_model_training.py
     ```
5. **Train the policy based on World Model**:
   - Still, You can personalize the parameters you prefer in the config file located at `/MiniGrid/modelBased/conf/model/config.yaml`.
   - train the PPO model
   ```bash
   python /modelBased/modelBased/PPO_model_training.py
   ```

6. **Run the trained policy model in the real world**:
   ```bash
   python /modelBased/modelBased/PPO_model_test.py
   ```
   
## Q&A

1. If you encounter issues when importing packages, check the absolute path set in every script (you need to adjust this to your path). 
This is on the TODO list and will be fixed in the future—though I'm unsure how soon :)
   ```bash
   import sys
   sys.path.append('/home/siyao/project/rlPractice/MiniGrid')

2. If you encounter issues related to can't find the path, change the path in .env file to your own device.
   ```bash
   export PROJECT_ROOT="/home/siyao/project/rlPractice/MiniGrid/modelBased"
   export TRAIN_DATASET_PATH="${PROJECT_ROOT}/data/train_world_model"
   export PTH_FOLDER="${PROJECT_ROOT}/modelBased/models/ckpt"
   export LOG_FOLDER="${PROJECT_ROOT}/modelBased/models/log"
   PYDEVD_WARN_EVALUATION_TIMEOUT=100.00
