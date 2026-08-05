# Agent-Centric Attentive World Model

This repository contains the Agent-Centric Attentive World Model and its domain adapters for three types of environments: MiniGrid, Crafter, and BipedalWalker.

The `modelBased/` and shared `domain/` packages are the canonical world-model source. Other projects, including Curriculum/MAC, should install this repository in editable mode instead of copying these directories:

```bash
pip install -e .
```

## Supported Domains

The same world-model pipeline is shared by all three domains, while the domain adapters convert each environment into the representation expected by the model.

| Domain | Environment characteristics | Observation representation | Action space | WM data | Typical tasks |
| --- | --- | --- | --- | --- | --- |
| **MiniGrid** | 2D discrete grid with walls, doors, keys, objects, and lava | Discrete grid with 3 channels | Discrete | `discrete` | Empty grids, mazes, key-door navigation, obstacle avoidance |
| **Crafter** | 2D open-world survival/crafting environment with resources and entities | Discrete symbolic grid with 2 channels, plus inventory state | Discrete | `discrete` | Collecting resources, crafting, exploration, target-task layouts |
| **BipedalWalker** | Continuous physics-based locomotion over custom terrain | Normalized continuous state vector | Continuous | `norm` | Walking over target terrain, mini-task terrain, stump/obstacle layouts |

### Domain-specific configuration

Select the active domain at the top of `modelBased/config/config.yaml`:

```yaml
domain: minigrid  # minigrid | crafter | bipedalwalker
```

The main differences are already defined under `domains:` in the same file:

- **MiniGrid** uses a `3`-channel discrete grid, an attention mask size of `3`, and level files under `trainer/level/`.
- **Crafter** uses a `2`-channel discrete symbolic grid, an attention mask size of `5`, and custom layouts under `trainer/level/crafter/`. Its custom environment creates the world from the layout, so `env_path` is kept only for compatibility.
- **BipedalWalker** uses a normalized continuous state with shape `[1, 1, 24]`, an attention mask size of `5`, and terrain files under `trainer/level/bipedal_walker/`. Data collection can use either the heuristic behavior policy or a pretrained SB3 policy.

Available task groups include:

- MiniGrid: `Grid_11_11_KD_level1.txt`, `Grid_11_11_KD_level2.txt`, `Grid_11_11_KD_level3.txt`, and custom maze/corridor layouts.
- Crafter: `crafter_minitask_*.txt` and `target_tasks/crafter_target_task_*.txt`.
- BipedalWalker: `minitasks/minitask_*.txt` and `target_tasks/bipedal_target_task_*.txt`.

## Environment Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/Sveali41/Agent-Centric-Attentive-World-Model.git
   cd Agent-Centric-Attentive-World-Model
   ```

2. **Install the requirements:**

   ```bash
   pip3 install -r requirements.txt
   pip install -e .
   ```

3. **Load the repository-local paths before running experiments:**

   ```bash
   source .env
   ```

## Training Pipeline

### 1. Collect transition data

Select a domain in `modelBased/config/config.yaml`, then collect trajectories:

```bash
python modelBased/data/data_collect.py domain=minigrid
python modelBased/data/data_collect.py domain=crafter
python modelBased/data/data_collect.py domain=bipedalwalker
```

The output is selected automatically from the active domain configuration and saved under `modelBased/data/train_world_model/`:

- MiniGrid: `minigrid_train_random.npz`
- Crafter: `crafter_*.npz`
- BipedalWalker: `bipedalwalker_<task_group>_<task_name>_<data_type>.npz`

Some datasets may already exist in this directory and can be reused.

### 2. Train the world model

```bash
python modelBased/world_model/AttentionWM_training.py domain=minigrid
python modelBased/world_model/AttentionWM_training.py domain=crafter
python modelBased/world_model/AttentionWM_training.py domain=bipedalwalker
```

The model automatically uses discrete losses for MiniGrid and Crafter, and normalized continuous targets for BipedalWalker.

### 3. Train a policy using the world model

Configure PPO in `modelBased/config/config.yaml`, then run:

```bash
python modelBased/policy_training/PPO_world_training.py domain=minigrid
python modelBased/policy_training/PPO_world_training.py domain=crafter
python modelBased/policy_training/PPO_world_training.py domain=bipedalwalker
```

### 4. Test a trained policy

```bash
python modelBased/policy_training/PPO_world_test.py domain=minigrid
python modelBased/policy_training/PPO_world_test.py domain=crafter
python modelBased/policy_training/PPO_world_test.py domain=bipedalwalker
```

## Recommended Workflow by Domain

For **MiniGrid**, choose a text layout in `trainer/level/`, set `domain: minigrid`, and use the discrete world-model pipeline. This is the simplest domain for checking grid transitions and key-door or obstacle behavior.

For **Crafter**, choose a layout from `trainer/level/crafter/`, set `domain: crafter`, and keep `env.crafter.stochastic: false` for deterministic initial experiments. Set it to `true` when evaluating robustness to moving entities and stochastic behavior.

For **BipedalWalker**, choose a terrain from `trainer/level/bipedal_walker/`, set `domain: bipedalwalker`, and use normalized data. The default `behavior_policy: heuristic` is suitable for initial data collection; set `behavior_policy: pretrained_sb3` and provide `sb3_model_path` when using a trained continuous-control policy.

## Q&A

1. If imports fail, check that the repository is installed in editable mode and that `.env` has been sourced. If a script still requires an absolute path, replace it with the path to your local clone.

2. If a dataset, checkpoint, or level cannot be found, verify the paths in `.env`, especially `PROJECT_ROOT`, `TRAINER_PATH`, `TRAIN_DATASET_PATH`, and `MODEL_FPATH`.

3. For a different task, update the corresponding values under `domains.<domain>` in `modelBased/config/config.yaml`, such as `env_path`, `task_name`, `task_folder`, or `validation_task_name`.
