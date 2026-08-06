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

All documented entry points add the repository root automatically, so they can be run from the repository root with either direct script syntax or Python module syntax. Module syntax is also supported:

```bash
python -m modelBased.world_model.AttentionWM_training domain=minigrid
```

## Training Pipeline

### One-command pipeline

The repository root contains the Hydra-based `run_pipeline.py`, which checks expected dataset and checkpoint files before each stage. Existing stages are skipped automatically:

```bash
source .env

python run_pipeline.py pipeline.label=minigrid
python run_pipeline.py pipeline.label=crafter
python run_pipeline.py pipeline.label=bipedalwalker
python run_pipeline.py pipeline.label=full                         # all domains
python run_pipeline.py pipeline.label=full pipeline.skip_policy=true # data + world model only
python run_pipeline.py pipeline.label=minigrid pipeline.force=true  # force rerun
```

The default values are stored in `modelBased/config/config.yaml` under `pipeline:`. Dataset and checkpoint paths can also be changed there or overridden with Hydra arguments.

The current PPO world-model policy implementation is MiniGrid-specific. Therefore, `run_pipeline.py` runs the policy stage for MiniGrid and reports a clear skip for Crafter and BipedalWalker until their corresponding policy adapters are added.

For a longer policy run after validating the setup, override the PPO budget explicitly:

```bash
python run_pipeline.py pipeline.label=minigrid \
  PPO.max_training_timesteps=300000 \
  PPO.max_ep_len=512
```

If `pipeline.label` is omitted, it follows the top-level `domain` value in `modelBased/config/config.yaml`. For example, with `domain: bipedalwalker`, simply run:

```bash
python run_pipeline.py
```

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

Configure PPO in `modelBased/config/config.yaml` with real-environment training
disabled:

```yaml
PPO:
  train_in_real_env: false
```

Then run:

```bash
python -m modelBased.policy_training.PPO_world_training domain=minigrid
```

The current world-model PPO implementation is MiniGrid-specific.

### 3a. Train model-free PPO directly in the real environment

To bypass data collection and the world model, enable real-environment
training in `modelBased/config/config.yaml`:

```yaml
domain: minigrid

domains:
  minigrid:
    task_name: simple_test
    layout_path: ${oc.env:PROJECT_ROOT}/level/simple_test.txt

PPO:
  train_in_real_env: true
```

`domains.minigrid.layout_path` selects the real MiniGrid environment used by
both policy training and testing. Change `task_name` together with the layout
so that a new environment does not reuse a checkpoint name from an older task.

Run model-free PPO directly with:

```bash
python -m modelBased.policy_training.PPO_world_training domain=minigrid
```

The same settings can be supplied without editing the YAML file:

```bash
python -m modelBased.policy_training.PPO_world_training \
  domain=minigrid \
  PPO.train_in_real_env=true \
  domains.minigrid.task_name=simple_test \
  domains.minigrid.layout_path=${PROJECT_ROOT}/level/simple_test.txt
```

When using the complete pipeline, `PPO.train_in_real_env=true` also tells
`run_pipeline.py` to skip transition-data collection and world-model training:

```bash
python run_pipeline.py pipeline.label=minigrid PPO.train_in_real_env=true
```

### 4. Test a trained policy

```bash
# Run the learned compact-action policy in the real MiniGrid environment and
# open a live render window.
python -m modelBased.policy_training.PPO_world_test domain=minigrid \
  PPO.total_test_episodes=1 PPO.render=true PPO.save_gif=false PPO.save_csv=false

# Headless alternative: save the first real-environment episode as a GIF.
python -m modelBased.policy_training.PPO_world_test domain=minigrid \
  PPO.total_test_episodes=1 PPO.render=false PPO.save_gif=true PPO.save_csv=false
```

The evaluator loads `PPO.checkpoint_path`, reads the same configured layout as
policy training, and maps compact actions `[0,1,2,3,4]` back to MiniGrid native
actions `[0,1,2,3,5]` before calling the real environment's `step()` method.
Set `PPO.test_deterministic=false` to sample from the learned categorical
policy, or `PPO.test_deterministic=true` to evaluate its argmax behavior. The
default configuration evaluates the learned stochastic policy. Argmax remains
available as a stricter diagnostic, but it can turn a valid stochastic policy
into a repeated-action loop in states where its highest-probability action does
not change the observation.

## Recommended Workflow by Domain

For **MiniGrid**, choose a text layout in `trainer/level/`, set `domain: minigrid`, and use the discrete world-model pipeline. This is the simplest domain for checking grid transitions and key-door or obstacle behavior.

For **Crafter**, choose a layout from `trainer/level/crafter/`, set `domain: crafter`, and keep `env.crafter.stochastic: false` for deterministic initial experiments. Set it to `true` when evaluating robustness to moving entities and stochastic behavior.

For **BipedalWalker**, choose a terrain from `trainer/level/bipedal_walker/`, set `domain: bipedalwalker`, and use normalized data. The default `behavior_policy: heuristic` is suitable for initial data collection; set `behavior_policy: pretrained_sb3` and provide `sb3_model_path` when using a trained continuous-control policy.

## Configuring Environment Layouts

Each domain uses a different layout format. After creating or selecting a layout file, update the matching values under `domains.<domain>` in `modelBased/config/config.yaml`.

### MiniGrid layout

MiniGrid files contain two equally sized blocks separated by one empty line:

1. An object layout.
2. A color layout with the same dimensions.

For example, `level/Grid_11_11_KD_level1.txt` uses symbols such as:

| Symbol | Meaning |
| --- | --- |
| `W` | Wall |
| `E` | Empty/floor cell |
| `S` | Agent start position |
| `K` | Key |
| `D` | Locked door |
| `O` | Unlocked door |
| `B` | Ball |
| `X` | Box |
| `G` | Goal |
| `L` | Lava |

The second block specifies colors character-by-character. Supported color symbols include `R` (red), `G` (green), `B` (blue), `Y` (yellow), `M` (purple), and `W`/`E`/`S` (grey). Both blocks must have the same number of rows and columns.

```text
WWWWW
WSEKW
WEEGW
WWWWW

WWWWW
WGYGW
WEEGW
WWWWW
```

Point `domains.minigrid.env_path` to the new file. The file path can also be changed temporarily from the command line:

```bash
python modelBased/data/data_collect.py \
  domain=minigrid \
  domains.minigrid.env_path=/path/to/my_minigrid_layout.txt
```

### Crafter layout

Crafter files contain a character grid, optionally followed by an initial-inventory block separated by an empty line. The player must be represented by `A`.

Common map symbols are:

| Symbol | Meaning |
| --- | --- |
| `G` or `.` | Grass |
| `W` | Water |
| `T` | Tree |
| `R` | Stone |
| `C` | Coal |
| `I` | Iron |
| `O` | Diamond |
| `L` | Lava |
| `P` | Path |
| `S` | Sand |
| `X` | Table |
| `U` | Furnace |
| `A` | Player |
| `M` | Cow |
| `Z` | Zombie |
| `K` | Skeleton |
| `t` | Plant |
| `F` | Fence |

Example:

```text
TTTTTT
GGAWOG
GGGGGG

# --- Initial Stats ---
wood: 0
food: 9
drink: 9
energy: 9
```

Set `domains.crafter.env_path` to the new file. Crafter reads the map directly from this file; the `env_path` comment in the configuration only means that the path is not used by the generic MiniGrid loader.

### BipedalWalker layout

BipedalWalker uses a one-line sequence of terrain tokens. Each token consists of a terrain type followed by a numeric parameter:

| Token | Meaning |
| --- | --- |
| `G<n>` | Grass segment with length/width parameter `n` |
| `S<n>` | Stump with height `n` |
| `P<n>` | Pit with width `n` |
| `T<n>` | Stairs; positive/negative `n` controls direction |
| `R<n>` | Terrain roughness parameter |

For example:

```text
G35 S3.61 P4 G5 G5 T3 G5 T-2 G5 R2.86 G5
```

Tokens may be separated by spaces or split across multiple lines. Lines beginning with `#` are treated as comments. Set `domains.bipedalwalker.task_name` and, if needed, `task_folder` to use a new terrain file:

```yaml
domains:
  bipedalwalker:
    task_group: target
    task_folder: target_tasks
    task_name: bipedal_target_task_16
```

For a custom file outside the default task folders, update `domains.bipedalwalker.env_path` directly. The generated dataset filename changes automatically with the selected task name and data type.

## Q&A

1. If imports fail, check that the repository is installed in editable mode and that `.env` has been sourced. If a script still requires an absolute path, replace it with the path to your local clone.

2. If a dataset, checkpoint, or level cannot be found, verify the paths in `.env`, especially `PROJECT_ROOT`, `TRAINER_PATH`, `TRAIN_DATASET_PATH`, and `MODEL_FPATH`.

3. For a different task, update the corresponding values under `domains.<domain>` in `modelBased/config/config.yaml`, such as `env_path`, `task_name`, `task_folder`, or `validation_task_name`.
