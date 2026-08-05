from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from minigrid_custom_env import CustomMiniGridEnv
from modelBased.common.utils import TRAINER_PATH, extract_unique_patches, generate_minitasks_until_covered
from modelBased.world_model import AttentionWM_training
from modelBased.policy_training import PPO_world_training
from modelBased.data.data_collect import visualize_agent_coverage, visualize_saved_dataset
from datetime import datetime
import hydra
import os
import torch
import gc
import csv
import matplotlib.pyplot as plt
import random
import numpy as np
from minigrid.wrappers import FullyObsWrapper


'''
Process
1. load the minitasks
2. collect data from the env
3. train(finetuning) the attention & WM
4. using the trained attention & WM to play in the final task 
5. return score in the final task as the feedback
'''

# ============================================================
# 1. Global seed fixing utility
# ============================================================
def set_seed(seed: int):
    """Fix all random sources to ensure full reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Random seed fixed to {seed}]")

def count_data_in_dataset(file_name):
    """
    输入: data 文件名（例如 'only_lava_minitask_test.npz'）
    输出: 样本数量（data['a'].shape[0]）
    """
    data_path = TRAINER_PATH / 'data' / file_name
    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return None

    try:
        data = np.load(data_path, allow_pickle=True)
        num_samples = data['a'].shape[0]
        print(f"{file_name}: {num_samples} samples")
        return num_samples
    except Exception as e:
        print(f"[Error] Failed to read {file_name}: {e}")
        return None

def split_targets_into_minitasks(
    target_task_input, patch_size=3, patches_per_minitask=4, trials=200
):
    """
    Extract patches from one or multiple target tasks, remove duplicates,
    and generate minitasks multiple times to return the smallest minitask set.

    Args:
        target_task_input: str or List[str]
        patch_size: int
        patches_per_minitask: int
        trials: how many times to run the generation (default=10)

    Returns:
        List[str]: the smallest minitask set found across trials
    """

    # --------- Normalize input to list ---------
    if isinstance(target_task_input, str):
        target_files = [target_task_input]
    elif isinstance(target_task_input, (list, tuple)):
        target_files = list(target_task_input)
    else:
        raise ValueError("target_task_input must be a str or list[str]")

    # print(f"[Patch Collect] Processing {len(target_files)} target tasks...")

    # --------- Collect patches from all targets ---------
    all_patches = set()

    for file in target_files:
        env = CustomMiniGridEnv(
            txt_file_path=TRAINER_PATH / "level" / file,
            custom_mission="Find the key and open the door.",
            max_steps=5000,
            render_mode=None
        )
        env.reset()
        layout_str = env.layout_str
        patches = extract_unique_patches(layout_str, patch_size)
        all_patches.update(patches)

    # print(f"[Patch Collect] Total unique patches across all targets = {len(all_patches)}")

    all_patches_list = list(all_patches)

    # ======================================================
    # Run multiple times and pick the smallest minitask set
    # ======================================================
    best_minitasks = None
    best_size = float("inf")

    # print(f"[Minitasks] Running {trials} trials to find minimal cover...")

    for t in range(trials):
        minitasks = generate_minitasks_until_covered(
            all_patches_list,
            patch_size,
            patches_per_minitask=patches_per_minitask
        )

        size = len(minitasks)
        # print(f"  Trial {t+1}/{trials}: {size} minitasks")

        if size < best_size:
            best_size = size
            best_minitasks = minitasks

    print(f"[Minitasks] Best result: {best_size} minitasks (out of {trials} trials)")
    return best_minitasks


def collect_data_general(
    cfg,
    env_source,
    save_name: str,
    max_steps: int = 10000,
    maximum_dataset_size: int = None,
):
    """
    General environment data-collection function.

    env_source can be:
        - str (ending with .txt): path to MiniGrid layout file
        - tuple(layout_str, color_str): minitask strings
    
    save_name: file prefix to save data, e.g. "lava_minitask"
    """
    cfg.env.collect.maximum_dataset_size = maximum_dataset_size
    support = Support.Support(cfg)

    # -----------------------------
    # 1. Build environment
    # -----------------------------
    if isinstance(env_source, (str, os.PathLike)) and str(env_source).endswith(".txt"):
        # From text file
        env = support.wrap_env_from_text(env_source, max_steps=max_steps)

    elif isinstance(env_source, tuple) and len(env_source) == 2:
        # From minitask strings
        layout_str, color_str = env_source

        env = FullyObsWrapper(CustomMiniGridEnv(
            layout_str=layout_str,
            color_str=color_str,
            custom_mission="Learn minitask",
            render_mode=None,
            max_steps=max_steps,
        ))
    else:
        raise ValueError("env_source must be a .txt filepath or (layout_str, color_str) tuple")

    # -----------------------------
    # 2. Set dataset save paths
    # -----------------------------
    data_save_dir = TRAINER_PATH / "data"
    explore_type = cfg.env.collect.data_type  # random / uniform
    save_path = data_save_dir / f"{save_name}_test_{explore_type}.npz"

    cfg.env.collect.data_save_path = str(save_path)
    cfg.env.collect.visualize_save_path = TRAINER_PATH / "logs" / "dataset_visualization"
    cfg.env.collect.visualize_filename = f"{save_name}_{explore_type}.png"

    # -----------------------------
    # 3. Delete old dataset file
    # -----------------------------
    support.del_env_data_file()

    # -----------------------------
    # 4. Run actual data collection
    # -----------------------------
    support.collect_data_trainer(
        env=env,
        wandb_run=None,
        validate=False,
        save_img=False,
        log_name=f"collect_{save_name}",
        max_steps=None,  # already set in env
    )

    print("Data collection complete!")
    return save_path

def create_data_subsets(dataset_npz, interval_size):
    """
    Shuffle and split dataset_npz into multiple subsets of size interval_size.
    Return a list of dict subsets: [{a,b,c,f}, ...]
    """

    obs_all = dataset_npz["a"]
    next_all = dataset_npz["b"]
    act_all = dataset_npz["c"]
    info_all = dataset_npz["f"] if "f" in dataset_npz else None

    total = len(obs_all)
    if interval_size is None:
        return [{
            "a": obs_all,
            "b": next_all,
            "c": act_all,
            "f": info_all,
        }]

    # ---- Shuffle ----
    indices = np.arange(total)
    np.random.shuffle(indices)

    obs_all = obs_all[indices]
    next_all = next_all[indices]
    act_all = act_all[indices]
    if info_all is not None:
        info_all = info_all[indices]

    # ---- Split into subsets ----
    subsets = []
    num_rounds = int(np.ceil(total / interval_size))

    for i in range(num_rounds):
        start = i * interval_size
        end = min((i + 1) * interval_size, total)

        subset = {
            "a": obs_all[start:end],
            "b": next_all[start:end],
            "c": act_all[start:end],
            "f": info_all[start:end] if info_all is not None else None,
        }

        subsets.append(subset)

    return subsets

def train_wm_with_subsets(
    cfg,
    subsets,
    fisher_buffer,
    temp_dir,
    num_iterations,
    old_params,
    fisher,
    current_sample_ratio,
    fisher_buffer_elements_ratio
):
    """
    Train WM on multiple subsets with Fisher-based replay.
    Keeps old_params/fisher across phases.
    """

    for it in range(num_iterations):

        # -------- pick subset based on iteration --------
        idx = it if it < len(subsets) else np.random.randint(len(subsets))
        subset = subsets[idx]

        # ---- write subset to temp npz ----
        temp_path = os.path.join(temp_dir, f"subset_{idx}.npz")
        np.savez_compressed(temp_path, **subset)
        cfg.attention_model.data_dir = temp_path

        # ---- Prepare replay data ----
        replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None

        # ---- Train WM ----
        cfg.attention_model.freeze_weight = False
        old_params, fisher = AttentionWM_training.train_api(
            cfg,
            old_params,
            fisher,
            replay_data=replay_data
        )

        # ---- Update fisher buffer ----
        samples = {
            'obs': subset['a'],
            'obs_next': subset['b'],
            'act': subset['c'],
            'info': subset['f']
        }

        fisher_buffer.update_combined(samples, current_sample_ratio, fisher_buffer_elements_ratio)

        print(f"[WM] Iter {it+1}/{num_iterations} using subset {idx}")

    return old_params, fisher

def validate_on_target_task(cfg, old_params, data_save_dir, target_file, phase_name, VALID_TIMES=1):
    """
    Run WM validation on the fixed target task, return avg loss.
    Save no heatmap here (can add if needed).
    """

    cfg.attention_model.freeze_weight = True
    cfg.attention_model.keep_cell_loss = True
    cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)

    losses = []

    for v in range(VALID_TIMES):
        val_result, model = AttentionWM_training.train_api(cfg, old_params, None)
        loss_val = float(val_result[0]['avg_val_loss_wm'])
        losses.append(loss_val)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    cfg.attention_model.keep_cell_loss = False

    avg_loss = float(np.mean(losses))
    return avg_loss

def save_validation_csv(csv_path, seed, mode, phase_name, transitions, loss):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['seed', 'mode', 'phase', 'transitions', 'avg_target_loss']
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'seed': seed,
            'mode': mode,
            'phase': phase_name,
            'transitions': transitions,
            'avg_target_loss': loss,
        })

@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def collect_data_for_txt(cfg: DictConfig):
    """
    Collects data from a MiniGrid environment defined by a text file.

    Args:
        cfg (DictConfig): Configuration object containing environment and collection settings
    """
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    env_text_file_name = '3obstacles_target_task.txt'
    file_name = os.path.splitext(env_text_file_name)[0]
    explore_type = cfg.env.collect.data_type  # 'random' or 'uniform'
    cfg.env.collect.data_save_path = TRAINER_PATH / 'data' / f'{file_name}_test_{explore_type}.npz'
    cfg.env.collect.visualize_save_path = TRAINER_PATH / 'logs' / 'dataset_visualization'
    cfg.env.collect.visualize_filename = f"{file_name}_{explore_type}.png"

    collect_data_general(
        cfg,
        env_source=TRAINER_PATH / 'level' / env_text_file_name,
        save_name=file_name,
        max_steps=1000
    )

@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def visualize_CL_dataset(cfg: DictConfig):
    """
    Visualizes the agent coverage from a saved dataset.

    Args:
        data_path (str): Path to the saved dataset (.npz file).
        save_path (str): Directory to save the visualization.
        fig_name (str): Filename for the saved figure.
    """
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    data_save_dir = TRAINER_PATH / 'data'
    env_text_file_name = '3obstacles_target_task.txt'
    file_name = os.path.splitext(env_text_file_name)[0]
    explore_type = cfg.env.collect.data_type  # 'random' or 'uniform'
    cfg.env.collect.data_save_path = os.path.join(data_save_dir, f'{file_name}_test_{explore_type}.npz')
    cfg.env.collect.visualize_save_path = TRAINER_PATH / 'logs' / 'dataset_visualization'
    cfg.env.collect.visualize_filename = f"{file_name}_{explore_type}.png"

    data_path = cfg.env.collect.data_save_path
    save_path = cfg.env.collect.visualize_save_path
    fig_name = cfg.env.collect.visualize_filename

    visualize_saved_dataset(
        data_path=data_path,
        save_path=os.path.join(save_path, fig_name),
        fig_name=fig_name
    )

@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def test_1(cfg: DictConfig):
    """
    Performs continual training of the Attention-based World Model (WM) on a sequence of environments.
    Optionally supports interval training and validation:
      - If interval_train_steps is None, behaves as before (train once per minitask, then validate VALID_TIMES times).
      - If interval_train_steps is an integer, trains for 'interval_train_steps' and validates 'validation_rounds' times.
    """
    import csv
    import numpy as np
    from fisher_buffer import FisherReplayBuffer
    from modelBased.world_model.AttentionWM import AttentionWorldModel
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    # ----------------------------------------------------------
    # New optional variables for interval control
    # ----------------------------------------------------------
    interval_train_steps = None     # e.g. 1000 to enable interval mode, None = original behavior
    validation_rounds = 10          # number of validation rounds if interval mode enabled
    VALID_TIMES = 50                # number of target validations per phase (original setting)

    csv_path = os.path.join(TRAINER_PATH, 'logs', 'target_eval_log.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    old_params, fisher = None, None

    env_text_file_name = [
        'only_wall_minitask.txt',
        'only_wall_minitask_2.txt',
        'only_lava_minitask.txt',
        'only_lava_minitask_2.txt',
        'only_key_minitask.txt',
        'only_key_minitask_2.txt',
        'only_door_minitask.txt',
        'only_door_minitask_2.txt',
    ]
    step_len = len(env_text_file_name)
    data_save_dir = TRAINER_PATH / 'data'

    print(f"Saving target validation results to CSV: {csv_path}")

    for step in range(step_len):
        print(f"\n===== Training Phase {step+1}/{step_len}: {env_text_file_name[step]} =====")
        cfg.attention_model.freeze_weight = False

        file_name = os.path.splitext(env_text_file_name[step])[0]
        phase_name = file_name
        cfg.attention_model.data_dir = os.path.join(data_save_dir, f'{file_name}_test_uniform.npz')

        if len(fisher_buffer) > 0:
            replay_data = fisher_buffer.export_dict()
        else:
            replay_data = None

        # ----------------------------------------------------------
        # (1) Training logic
        # ----------------------------------------------------------
        if interval_train_steps is None:
            # Original full-phase training
            cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher, replay_data=replay_data)
            old_params, fisher = cur_old_params, cur_fisher

        else:
            # Interval training mode: multiple partial trainings before validation
            for round_idx in range(validation_rounds):
                print(f"[{phase_name}] Interval training {round_idx+1}/{validation_rounds} "
                      f"({interval_train_steps} steps each)")
                cur_old_params, cur_fisher = AttentionWM_training.train_api(
                    cfg, old_params, fisher, replay_data=replay_data, train_steps=interval_train_steps
                )
                old_params, fisher = cur_old_params, cur_fisher

        # ----------------------------------------------------------
        # (2) Update Fisher replay buffer
        # ----------------------------------------------------------
        task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
        samples = {
            'obs': task_npz['a'],
            'obs_next': task_npz['b'],
            'act': task_npz['c'],
            'info': task_npz['f'] if 'f' in task_npz else None
        }
        model_eval = AttentionWorldModel(cfg.attention_model).to(device)
        fisher_buffer.update_combined(samples, 0.3, 0.5)

        # ----------------------------------------------------------
        # (3) Validation (same as before)
        # ----------------------------------------------------------
        print(f"\nStart validating target task {VALID_TIMES} times for phase: {phase_name}")
        results_to_save = []

        cfg.attention_model.freeze_weight = True
        target_file = '3obstacles_target_task_test_uniform.npz'
        cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)
        cfg.attention_model.keep_cell_loss = True
        
        sum_map = None
        all_loss_whole_map = []

        for v in range(VALID_TIMES):
            val_result, model = AttentionWM_training.train_api(cfg, None, None)
            loss_map = model.loss_map_result

            if sum_map is None:
                sum_map = np.array(loss_map, dtype=np.float32)
            else:
                sum_map += loss_map

            if isinstance(val_result, list) and isinstance(val_result[0], dict) and 'avg_val_loss_wm' in val_result[0]:
                loss_val = float(val_result[0]['avg_val_loss_wm'])
                all_loss_whole_map.append(loss_val)
            else:
                raise ValueError(f"Unexpected validation return format: {val_result}")

            results_to_save.append({
                'phase': phase_name,
                'validate_index': v + 1,
                'target_loss': loss_val
            })

            del model, loss_map
            torch.cuda.empty_cache()
            gc.collect()

        cfg.attention_model.keep_cell_loss = False

        # ----------------------------------------------------------
        # (4) Save validation results
        # ----------------------------------------------------------
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['phase', 'validate_index', 'target_loss'])
            if not file_exists:
                writer.writeheader()
            writer.writerows(results_to_save)

        print(f"Finished phase {phase_name}. Results saved to {csv_path}")
        print(f"Average Target Loss over {VALID_TIMES} validations: {np.mean(all_loss_whole_map)}")

        # ----------------------------------------------------------
        # (5) Plot average loss heatmap (unchanged)
        # ----------------------------------------------------------
        avg_loss_map = sum_map / VALID_TIMES
        plt.figure(figsize=(6, 5))
        plt.imshow(avg_loss_map, cmap='viridis_r', interpolation='nearest')
        plt.colorbar(label='Average Loss Value')
        plt.title(f"Average Loss Map Heatmap ({phase_name})")
        plt.xlabel("X Position (columns)")
        plt.ylabel("Y Position (rows)")

        output_path = TRAINER_PATH / 'logs' / f"loss_map_avg_{phase_name}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Average heatmap saved to {output_path}")


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def curriculum_learning_transitions(cfg: DictConfig):
    """
    Curriculum Learning (CL) with Fisher Replay Buffer.
    Using modular functions:
        - create_data_subsets()
        - train_wm_with_subsets()
        - validate_on_target_task()
        - save_validation_csv()
    """

    import numpy as np
    import torch
    import os
    import gc
    import matplotlib.pyplot as plt

    from fisher_buffer import FisherReplayBuffer
    from modelBased.world_model.AttentionWM import AttentionWorldModel

    # --------------------------------------
    # Setup
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    test = False # True: using random data directly collect from target task -- baseline / False: using minitask strings
    interval_size = None # number of transitions per training phase # not split when None
    training_data_intotal = 100000 # total number of transitions for training
    explore_type = cfg.env.collect.data_type # uniform / random
    data_save_dir = TRAINER_PATH / "data"

    log_dir = TRAINER_PATH / "logs"
    csv_path = log_dir / "target_eval_log_compare_patches_minitask.csv"
    os.makedirs(log_dir, exist_ok=True)

    target_task_name = ['3obstacles_target_task.txt', '3obstacles_target_task1.txt','3obstacles_target_task2.txt', 
                        '3obstacles_target_task3.txt', '3obstacles_target_task4.txt', 
                        # '3obstacles_target_task5.txt',
                        # '3obstacles_target_task6.txt', '3obstacles_target_task7.txt'
                       ]
   

    target_file = [os.path.splitext(i)[0] + f"_test_uniform.npz" for i in target_task_name]

    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    current_sample_ratio = cfg.attention_model.current_sample_ratio
    fisher_buffer_elements_ratio = cfg.attention_model.fisher_buffer_elements_ratio
    old_params, fisher = None, None

    # --------------------------------------
    # New Parameter & Setup for N-phase collection
    # --------------------------------------
    N_PHASES_TO_COLLECT = cfg.attention_model.n_phases_to_collect # how many phases to accumulate before training
    
    # Initialize data accumulation buffer
    # Standard keys for transitions: 'obs', 'act', 'obs_next', 'reward', 'terminal', 'info'
    combined_data = {k: [] for k in ['a', 'b', 'c', 'd', 'e', 'f']}
    phases_collected = 0
    if test:
        phase_files = target_task_name
        mode = "Baseline" # 'CL' / 'Baseline'
    else:
        phase_files = split_targets_into_minitasks(target_task_name, patch_size=3, patches_per_minitask=4, trials=1)
        mode = "CL" # 'CL' / 'Baseline'

    for idx, phase in enumerate(phase_files):
        print(f"\n===== Data Collection: Phase {idx+1}/{len(phase_files)} =====")
        # --------------------------------------
        # Select sources
        # --------------------------------------


        # --------------------------------------
        # Training Loop
        # --------------------------------------
        # 2) minitask string mode (generating)
        if test:
            maximum_dataset_size = training_data_intotal // len(phase_files) 
            print(f"maximum_dataset_size per phase: {maximum_dataset_size}")
            # 1) txt file mode (loading)
            phase_name = os.path.splitext(phase)[0]
            dataset_path = collect_data_general(
                cfg,
                env_source=TRAINER_PATH / "level" / phase,
                save_name=phase_name,
                max_steps=10000,
                maximum_dataset_size=maximum_dataset_size
            )
            task_npz = np.load(dataset_path, allow_pickle=True)

        else:
            layout_str, color_str = phase.split("\n\n")
            save_name = f"minitask_{idx}"
            maximum_dataset_size = training_data_intotal // len(phase_files) 
            print(f"maximum_dataset_size per phase: {maximum_dataset_size}")
            dataset_path = collect_data_general(
                cfg,
                env_source=(layout_str, color_str),
                save_name=save_name,
                max_steps=80,
                maximum_dataset_size=maximum_dataset_size
            )
            phase_name = save_name
            task_npz = np.load(dataset_path, allow_pickle=True)
    # ---------------------------------------------------------------------------------

    # 1. Accumulate collected data
        data_dict = dict(task_npz)
        for k in combined_data.keys():
            if k in data_dict:
                # Append the NumPy array from the current phase to the list for this key
                combined_data[k].append(data_dict[k])
        phases_collected += 1
        
        # 2. Check if training should be triggered
        is_accumulation_complete = (phases_collected >= N_PHASES_TO_COLLECT)
        is_last_phase = (idx == len(phase_files) - 1)
        
        # Only train if we have collected N phases, OR if we're at the very end
        if is_accumulation_complete or (is_last_phase and phases_collected > 0):
            
            # --- Combine collected data into one structure ---
            print(f"--- Training on combined data from last {phases_collected} phases ---")
            
            final_npz = {}
            # Concatenate all lists of arrays into single NumPy arrays for training
            for k, arrays in combined_data.items():
                if arrays: 
                    final_npz[k] = np.concatenate(arrays, axis=0)
            
            # Determine logging info (transitions and phase name)
            transitions = maximum_dataset_size 
            
            # Use a descriptive name for logging the combined phase
            log_phase_name = f"Combined_P{idx - phases_collected + 2}_to_P{idx+1}" 
            
            # --------------------------------------
            # Create subsets (using the combined data)
            # --------------------------------------
            subsets = create_data_subsets(final_npz, interval_size)

            # --------------------------------------
            # Train WM on subsets
            # --------------------------------------
            old_params, fisher = train_wm_with_subsets(
                cfg,
                subsets,
                fisher_buffer,
                temp_dir=TRAINER_PATH /'data'/"temp",
                num_iterations=1,
                old_params=old_params,    
                fisher=fisher,             
                current_sample_ratio=current_sample_ratio,
                fisher_buffer_elements_ratio=fisher_buffer_elements_ratio
            )

            

            # --------------------------------------
            # VALIDATION (unified)
            # --------------------------------------
            validate_loss_on_target_task = []
            for i in target_file:
                avg_loss = validate_on_target_task(
                    cfg,
                    old_params=old_params,
                    data_save_dir=data_save_dir,
                    target_file=i,
                    phase_name=log_phase_name, # Use combined name
                    VALID_TIMES=1
                )
                validate_loss_on_target_task.append(avg_loss)
            avg_loss = float(np.mean(validate_loss_on_target_task))
            print(f"[Unified Validation] {log_phase_name} → Overall Avg Target Loss = {avg_loss:.5f}")

            # --------------------------------------
            # Save CSV (unified)
            # --------------------------------------
            save_validation_csv(
                csv_path=csv_path,
                seed=seed,
                mode=mode,
                phase_name=log_phase_name, # Use combined name
                transitions=transitions,
                loss=avg_loss
            )
            
            # 3. Reset buffer and counter for the next batch
            combined_data = {k: [] for k in combined_data.keys()}
            phases_collected = 0
            
        else:
            # Continue collecting data if N phases haven't been reached
            print(f"Current accumulation: {phases_collected}/{N_PHASES_TO_COLLECT}. Continuing collection...")
    
    # validate on the historical minitasks checking for forgetting (optional)
    print("\n===== Final Forgetting Check on Historical Minitasks =====")
    for step in range(len(phase_files)):
        minitask_filename = f"minitask_{step}_test_{explore_type}.npz"
        cfg.attention_model.freeze_weight = True
        cfg.attention_model.data_dir = os.path.join(data_save_dir, minitask_filename)
        val_result, model = AttentionWM_training.train_api(cfg, old_params, None)
        loss_val = float(val_result[0]['avg_val_loss_wm'])
        print(f"[Forgetting Check] Minitask {step} Loss: {loss_val:.5f}")


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def train_on_target_task_only(cfg: DictConfig):
    """
    Train Attention WM only on the target task dataset.
    """

    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    data_save_dir = TRAINER_PATH / "data"
    target_file = "only_lava_minitask_test_random.npz"

    cfg.attention_model.freeze_weight = False
    cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)

    old_params, fisher = AttentionWM_training.train_api(cfg, None, None)

if __name__ == "__main__":
    # collect_data_for_txt()
    # count_data_in_dataset("3obstacles_target_task_test.npz")
    # test_1()
    # # visualize_CL_dataset()
    curriculum_learning_transitions()
    # train_on_target_task_only()
