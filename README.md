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
| **MiniGrid** | 2D discrete grid with walls, doors, keys, objects, and lava | Discrete grid with 3 channels, plus colour-aware carried inventory | Discrete | `discrete` | Empty grids, mazes, key-door navigation, obstacle avoidance |
| **Crafter** | 2D open-world survival/crafting environment with resources and entities | Discrete symbolic grid with 2 channels, plus inventory state | Discrete | `discrete` | Collecting resources, crafting, exploration, target-task layouts |
| **BipedalWalker** | Continuous physics-based locomotion over custom terrain | Normalized continuous state vector | Continuous | `norm` | Walking over target terrain, mini-task terrain, stump/obstacle layouts |

### Domain-specific configuration

Select the active domain at the top of `modelBased/config/config.yaml`:

```yaml
domain: minigrid  # minigrid | crafter | bipedalwalker
```

The main differences are already defined under `domains:` in the same file:

- **MiniGrid** uses a `3`-channel discrete grid, an attention mask size of `3`, and level files under `level/minigrid/`.
- **Crafter** uses a `2`-channel discrete symbolic grid, an attention mask size of `5`, and custom layouts under `level/crafter/`. Its custom environment creates the world from the layout, so `env_path` is kept only for compatibility.
- **BipedalWalker** uses a normalized continuous state with shape `[1, 1, 24]`, an attention mask size of `5`, and terrain files under `level/bipedalwalker/`. Data collection can use either the heuristic behavior policy or a pretrained SB3 policy.

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
   # For GPU training, install the PyTorch build matching your CUDA version
   # first: https://pytorch.org/get-started/locally/
   pip3 install -r requirements.txt
   pip install -e .
   ```

`requirements.txt` contains the direct dependencies for the supported
MiniGrid, Crafter, and BipedalWalker pipeline. It intentionally does not pin
transitive packages, Jupyter tooling, or CUDA runtime wheels from one specific
machine.

3. **Create the local path configuration:**

Create a file named `.env` in the repository root. It is ignored by Git and
must be configured separately on each machine:

```bash
export PROJECT_ROOT="/absolute/path/to/Agent-Centric-Attentive-World-Model"

export ENV_PATH="${PROJECT_ROOT}/level"
export WORLD_MODEL_PATH="${PROJECT_ROOT}/modelBased"
export TRAIN_DATASET_PATH="${PROJECT_ROOT}/modelBased/data/train_world_model"
export MODEL_FPATH="${WORLD_MODEL_PATH}/models"

export GENERATOR_PATH="${PROJECT_ROOT}/legacy/generator"
export GENERATOR_MODEL_PATH="${GENERATOR_PATH}/models/ckpt"
export TRAINER_PATH="${PROJECT_ROOT}/legacy/trainer"
export TRAINER_MODEL_PATH="${TRAINER_PATH}/models/ckpt"
```

Normally, only `PROJECT_ROOT` needs to be changed. All other paths are derived
from it. Load the variables in every new terminal before running the project:

```bash
source .env
```

The active pipeline lives under `modelBased/` and `domain/`. Older curriculum,
continual-learning, and environment-generator experiments are retained under
`legacy/trainer/` and `legacy/generator/`. `TRAINER_PATH` and `GENERATOR_PATH`
point there for compatibility with the archived code. All active environment
layouts are selected through `ENV_PATH`.

4. **Select the environment to run:**

Edit `modelBased/config/config.yaml`. Set the top-level `domain`, then change
the `task_name` under the matching domain to a filename stem from its level
folder:

```yaml
domain: minigrid  # minigrid | crafter | bipedalwalker

domains:
  minigrid:
    task_name: Grid_11_11_KD_level1
    layout_path: ${oc.env:ENV_PATH}/minigrid/${domains.minigrid.task_name}.txt
```

The domain layout folders are:

```text
minigrid       -> level/minigrid/
crafter        -> level/crafter/target_tasks/
bipedalwalker  -> level/bipedalwalker/target_tasks/
```

For MiniGrid policy training, use the same config file to select the transition
source:

```yaml
PPO:
  train_in_real_env: false  # false: plan/train in WM; true: model-free real env
```

Then run the configured environment with:

```bash
source .env
python run_pipeline.py
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
  PPO.max_training_timesteps=320000 \
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

- MiniGrid: `minigrid_<task_name>_inventory_v1_<data_type>.npz`
- Crafter: `crafter_*.npz`
- BipedalWalker: `bipedalwalker_<task_group>_<task_name>_<data_type>.npz`

Some datasets may already exist in this directory and can be reused.

### 2. Train the world model

```bash
python modelBased/world_model/AttentionWM_training.py domain=minigrid
python modelBased/world_model/AttentionWM_training.py domain=crafter
python modelBased/world_model/AttentionWM_training.py domain=bipedalwalker
```

All domains optimize the same schema-driven observation objective:

```text
observation_loss = mean(normalized_field_loss for field in observation_schema)
```

Categorical fields use cross-entropy normalized by `log(number_of_classes)`,
binary fields use BCE normalized by `log(2)`, normalized continuous fields use
MSE, and count-like Crafter inventory fields use symlog MSE. The field terms
are averaged without domain-specific or rarity-specific multipliers. Training
logs one public loss curve, `train/observation_loss`; validation uses the same
objective as `val/observation_loss`. EWC, when enabled for continual learning,
remains an optimization regularizer and is not reported as a second observation
loss. Each domain declares only its fields in `domains.<domain>.observation_schema`
inside `modelBased/config/config.yaml`; adding a domain does not require another
loss implementation.

MiniGrid treats carried inventory as part of the learned state: token `0`
means empty hands and tokens `1..6` represent the six key colours. The WM
receives `(layout_t, inventory_t, action_t)` and predicts both
`layout_(t+1)` and `inventory_(t+1)`. All six actions, including pickup,
toggle, and drop, use this learned transition; imagined planning does not call
hand-written interaction dynamics. Training batches automatically balance
generic `(action, state_changed)` buckets so rare successful interactions are
not hidden by invalid no-op attempts.

### 3. Train a policy using the world model

Configure PPO in `modelBased/config/config.yaml` with real-environment training
disabled:

```yaml
PPO:
  train_in_real_env: false
  # GPU-batched imagined trajectories. This setting is used only by WM PPO.
  num_imagined_envs: 64
  # Total transitions per PPO update, so 4096 / 64 = 64 temporal WM steps.
  rollout_steps: 4096
```

Then run:

```bash
python -m modelBased.policy_training.PPO_world_training domain=minigrid
```

The current world-model PPO implementation is MiniGrid-specific.
`rollout_steps` and `max_training_timesteps` must be divisible by
`num_imagined_envs`. Set `num_imagined_envs: 1` to reproduce serial imagined
rollouts. Parallel environments keep independent episode lengths, rewards,
terminal flags, learned colour-aware inventory state, returns, and bootstrap values; PPO flattens
the resulting `[time, environment]` batch only after computing per-environment
returns.

### 3a. Train model-free PPO directly in the real environment

To bypass data collection and the world model, enable real-environment
training in `modelBased/config/config.yaml`:

```yaml
domain: minigrid

domains:
  minigrid:
    # Switch the MiniGrid task here. Use the layout .txt filename without .txt.
    task_name: simple_test
    layout_path: ${oc.env:ENV_PATH}/minigrid/${domains.minigrid.task_name}.txt

PPO:
  train_in_real_env: true
```

Switch MiniGrid environments at `domains.minigrid.task_name` in
`modelBased/config/config.yaml`. Set it to the layout text filename **without
the `.txt` extension**:

```text
level/minigrid/simple_test.txt              -> task_name: simple_test
level/minigrid/Grid_11_11_KD_level1.txt     -> task_name: Grid_11_11_KD_level1
```

This is normally the only field that needs to change. `task_name` is the
canonical task identifier: it selects `level/minigrid/<task_name>.txt` and is reused in
dataset, world-model checkpoint, policy-checkpoint, and visualization
filenames. Policy training and testing both resolve the same `layout_path`.

Run model-free PPO directly with:

```bash
wandb login  # required once per machine
python -m modelBased.policy_training.PPO_world_training domain=minigrid
```

With `PPO.use_wandb=true`, every episode records `episode/reward`,
`episode/success`, and `episode/steps`. Smoothed training statistics are logged
as `rolling/average_reward`, `rolling/success_rate`, and
`rolling/average_episode_steps`; PPO update loss and gradient statistics are
stored under `ppo/`. The rolling curves use the latest
`PPO.rolling_window_episodes` episodes (50 by default), not a fixed timestep
bucket. A converged policy should show a stable reward curve, success rate close
to 100%, and episode steps decreasing to a stable range.
WandB prints the run URL in the terminal but does not necessarily open a web
browser automatically. The same `PPO.use_wandb` setting is respected when
training through `run_pipeline.py`.

For controlled multi-seed experiments, set `PPO.seed` in the YAML or override
it on the command line:

```bash
python -m modelBased.policy_training.PPO_world_training domain=minigrid PPO.seed=0
python -m modelBased.policy_training.PPO_world_training domain=minigrid PPO.seed=1
python -m modelBased.policy_training.PPO_world_training domain=minigrid PPO.seed=2
```

WandB grouping and artifact names are derived automatically from
`PPO.train_in_real_env`. Each layout has its own group, and different seeds are
runs inside that group. WM planning preserves the historical group name;
direct-environment training adds a `realenv` marker:

```text
planning: minigrid_Grid_11_11_KD_level2_policy
real env: minigrid_Grid_11_11_KD_level2_realenv_policy
```

Runs and policy files keep only the layout, optional `realenv` marker, and seed:

```text
planning run: minigrid_Grid_11_11_KD_level2_seed4
real run:     minigrid_Grid_11_11_KD_level2_realenv_seed4

planning: policy_minigrid_Grid_11_11_KD_level2_seed4.ckpt
real env: policy_minigrid_Grid_11_11_KD_level2_realenv_seed4.ckpt
```

Different seeds remain separate. If an old five-action checkpoint already has
the simple planning filename, the pipeline detects its incompatible network
shape and retrains it instead of silently skipping. Use the same
`PPO.train_in_real_env` and `PPO.seed=<N>` settings when evaluating a specific
checkpoint. Set `PPO.checkpoint_path=/custom/path.ckpt` only when an explicit
override is needed.

The same settings can be supplied without editing the YAML file:

```bash
python -m modelBased.policy_training.PPO_world_training \
  domain=minigrid \
  PPO.train_in_real_env=true \
  domains.minigrid.task_name=simple_test
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

The evaluator automatically selects the matching `real_env` or `planning`
checkpoint from `PPO.train_in_real_env`, reads the same configured layout as
policy training, and maps compact actions `[0,1,2,3,4,5]` back to MiniGrid
native actions `[0,1,2,3,5,4]` before calling the real environment's `step()`
method. Historical compact IDs remain unchanged (`4=toggle`); `5=drop` is
appended so an agent can free its single carrying slot when a task contains
multiple keys. Earlier five-action PPO checkpoints remain under their old
filenames and are not compatible with the new six-output, inventory-aware
actor. MiniGrid datasets and WM checkpoints created before the colour-aware
inventory representation must also be recollected/retrained; `run_pipeline.py`
detects the dataset metadata version and does this automatically.
Set `PPO.test_deterministic=false` to sample from the learned categorical
policy, or `PPO.test_deterministic=true` to evaluate its argmax behavior. The
default configuration evaluates the learned stochastic policy. Argmax remains
available as a stricter diagnostic, but it can turn a valid stochastic policy
into a repeated-action loop in states where its highest-probability action does
not change the observation.

## Recommended Workflow by Domain

For **MiniGrid**, choose a text layout in `level/minigrid/`, set `domain: minigrid`, and use the discrete world-model pipeline. This is the simplest domain for checking grid transitions and key-door or obstacle behavior.

For **Crafter**, choose a layout from `level/crafter/`, set `domain: crafter`, and keep `env.crafter.stochastic: false` for deterministic initial experiments. Set it to `true` when evaluating robustness to moving entities and stochastic behavior.

For **BipedalWalker**, choose a terrain from `level/bipedalwalker/`, set `domain: bipedalwalker`, and use normalized data. The default `behavior_policy: heuristic` is suitable for initial data collection; set `behavior_policy: pretrained_sb3` and provide `sb3_model_path` when using a trained continuous-control policy.

## Configuring Environment Layouts

Each domain uses a different layout format. After creating or selecting a layout file, update the matching values under `domains.<domain>` in `modelBased/config/config.yaml`.

```text
level/
├── minigrid/
├── crafter/
└── bipedalwalker/
```

### MiniGrid layout

MiniGrid files contain two equally sized blocks separated by one empty line:

1. An object layout.
2. A color layout with the same dimensions.

For example, `level/minigrid/Grid_11_11_KD_level1.txt` uses symbols such as:

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

Put the new file under `level/minigrid/`, then switch the task in
`modelBased/config/config.yaml`. The `task_name` must match the `.txt` filename
without its extension. For example, `level/minigrid/my_layout.txt` requires:

```yaml
domains:
  minigrid:
    # Switch the active layout here.
    task_name: my_layout
```

The same selection can be made temporarily from the command line:

```bash
python -m modelBased.data.data_collect \
  domain=minigrid \
  domains.minigrid.task_name=my_layout
```

The former trainer directory contained different files with the same
`Grid_11_11_KD_level1/2/3.txt` names. Those older variants are preserved under
`level/minigrid/legacy_variants/`; the files directly under `level/minigrid/`
remain the canonical layouts selected by `task_name`.

For a layout outside the default `level/minigrid/` directory, override
`domains.minigrid.layout_path` explicitly. In that exceptional case, also set a
matching `task_name` so generated artifacts remain identifiable.

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

Place the file under `level/crafter/target_tasks/` and set
`domains.crafter.task_name` to its filename stem. Crafter's `layout_path`,
dataset, checkpoints, and visualization filenames are derived from that name.
Override `domains.crafter.layout_path` only for a file outside the default
folder.

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

For a custom file outside the default task folders, update
`domains.bipedalwalker.layout_path` directly. The generated dataset filename
changes automatically with the selected task name and data type.

## Q&A

1. If imports fail, check that the repository is installed in editable mode and that `.env` has been sourced. If a script still requires an absolute path, replace it with the path to your local clone.

2. If a dataset, checkpoint, or level cannot be found, verify the paths in `.env`, especially `PROJECT_ROOT`, `TRAINER_PATH`, `TRAIN_DATASET_PATH`, and `MODEL_FPATH`.

3. For a different task in a default task folder, change
   `domains.<domain>.task_name`; the layout, dataset, visualization, and
   checkpoint paths are derived from it. Override `layout_path` only when the
   layout is outside its default folder.
