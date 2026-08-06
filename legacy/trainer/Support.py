import hydra
import sys
from pathlib import Path

ROOTPATH = Path(__file__).resolve().parents[1]
if str(ROOTPATH) not in sys.path:
    sys.path.insert(0, str(ROOTPATH))
from modelBased.common.utils import GENERATOR_PATH, LEVEL_PATH, TRAINER_PATH
from omegaconf import DictConfig
from legacy.generator.common.utils import load_gen, generate_color_map, generate_obj_map, layout_to_string, combine_maps, clean_and_place_goal
from legacy.generator.gen import GAN
from domain.minigrid.minigrid_custom_env import *
import textwrap
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
import torch
from modelBased.data.data_collect import *
from modelBased.data.datamodule import *
# from legacy.generator.data.env_dataset_support import generate_valid_minigrid_with_key_door
from matplotlib import pyplot as plt
import pickle
import random
from legacy.generator.data.env_dataset_support import generate_envs_dataset
from legacy.generator.data.env_dataset_support import replace_vector_value, visualize_grid
from learning_buffer import EnvLearningBuffer
from legacy.generator.data.env_dataset_support import is_reachable
from modelBased.world_model import AttentionWM_training
from modelBased.policy_training import PPO_world_training


class Support:
    def __init__(self, cfg):
        self.cfg = cfg
        pass
    def _plot_subplot(self, row, col, position, data, cmap, colorbar_label, title, shrink):
        plt.subplot(row, col, position)
        im = plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, shrink=shrink, label=colorbar_label)
        plt.title(title)
    
    def _plot(self, data, cmap, title, shrink):
        plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(shrink=shrink, label=title)
        plt.title(title)
        plt.show()

    def visualize_dataset(self, file_name, customize_name='',total_count=20, saveImage=False, size=(10, 4), shrink=0.5):
        data = np.load(file_name)
        obs = data['a']
        next_obs = data['b']
        act = data['c']
        
        for idx in range(total_count):
            plt.close()
            direction = self.cfg.attention_model.direction_map[round(obs[idx, 2, :, :].max())]
            state_image = obs[idx, 0, :, :]
            state_image_next = next_obs[idx, 0, :, :]
            direction_next = self.cfg.attention_model.direction_map[round(next_obs[idx, 2, :, :].max())]
            if act is None:
                action = "None"
            else:
                action = self.cfg.attention_model.action_map[round(act[idx])]
        
            num_colors = 11
            custom_cmap = plt.cm.get_cmap('jet', num_colors)
            self._plot_subplot(1, 2, 1, state_image, custom_cmap, 'State', f"Dir: {direction}  Action: {action}", shrink)
            self._plot_subplot(1, 2, 2, state_image_next, custom_cmap, 'State Pre', f"Dir: {direction_next}", shrink)
            plt.tight_layout()
            if saveImage:
                save_file = os.path.join(self.cfg.attention_model.save_path, f"Compare_{customize_name}_{idx}.png")
                plt.savefig(save_file)
                plt.close()
            else:
                plt.show()

    def load_sample_MAP(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            all_maps = list(data_dict.values())
            random.shuffle(all_maps)
            return all_maps
        
    # def complete_map_from_MAP_sample(self, env):

    def loading_tasks(self, cfg):
        # add the ablation for MAP if we're not using MAP. we just load the ramdom sample envs
        if cfg.training_generator.elites_path is not None:
            # load the elites from the MAP sample
            env_database = self.load_sample_MAP(cfg.training_generator.elites_path)
            file_dir = None
        else:
            # for testing
            env_database = ['env1_move.txt','env2_move.txt','env3_move.txt', 'env3_move.txt']
            file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'level'))
        return env_database, file_dir


    def generate_env_from_generator(self, cfg, env_database=None,file_dir=None):
        print("++++++++++++++++++++++++++++++++++++ generating environment... ++++++++++++++++++++++++++++++++++++++++++++++")
        if cfg.training_generator.elites_path is not None:
            env_layout = env_database
            env = self.wrap_env(torch.tensor(env_layout).unsqueeze(0))
        else:
            file_path = os.path.join(file_dir, env_database)
            env_layout = self.load_text_map(file_path)
            env = self.wrap_env_from_text(file_path, max_steps=10000)
        return env, env_layout

    def collect_data_from_env(self, env, wandb_run, validate, save_img, log_name, max_steps):
        print("++++++++++++++++++++++++++++++++++++ collecting data from the environment... ++++++++++++++++++++++++++++++++++++++++++++++")
        self.del_env_data_file()  # clear the data_save_path
        self.collect_data_trainer(env, wandb_run, validate=validate, save_img=save_img, log_name=log_name, max_steps=max_steps)  # collect data from the env for training the WM
        if validate:
            print("Data collected for validation.")
        else:
            print("Data collected for training.")

    def train_world_model(self, cfg, old_params=None, fisher=None, env_layout=None, replay_data=None):
        print("++++++++++++++++++++++++++++++++++++ training the world model... ++++++++++++++++++++++++++++++++++++++++++++++")
        cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher, env_layout, replay_data)
        old_params, fisher = cur_old_params, cur_fisher
        return old_params, fisher


    def validate_world_model(self, cfg, old_params=None, fisher=None, env_layout=None):
        print("++++++++++++++++++++++++++++++++++++ training the world model... ++++++++++++++++++++++++++++++++++++++++++++++")
        cfg.attention_model.freeze_weight = True
        validation_error, _ = AttentionWM_training.train_api(cfg, old_params, fisher, env_layout, replay_data=None)
        return validation_error

    def generate_final_task(self, rows, cols, num_maps, save=True, wall_p_range=(0.1, 0.5),door_p_range=(0.075, 0.1), 
                            key_p_range=(0.1, 0.15), max_len=1e7,random_gen_max=3e4):
        final_task_dict = generate_envs_dataset(
            rows, cols, num_maps,
            wall_p_range=wall_p_range,
            door_p_range=door_p_range,
            key_p_range=key_p_range,
            max_len=max_len,
            random_gen_max=random_gen_max
        )
        file_names = []
        if save:
            for idx, key in enumerate(final_task_dict):
                map = final_task_dict[key]
            # 控制是否保存图片
                # visualize_grid(
                #         map,
                #         save_flag=True,
                #         save_path=LEVEL_PATH / 'minigrid' / 'final_task_set',
                #         idx=f'map_{idx}'
                #     )
                map_tensor = torch.tensor(map).unsqueeze(0)
                layout_string = generate_obj_map(map_tensor, self.cfg.training_generator.map_element)
                color_string = generate_color_map(layout_string)
                save_path = os.path.join(LEVEL_PATH, 'minigrid', 'final_task', f'gen_final_task_{idx}.txt')
                combine_maps(layout_string, color_string, save_path)
                file_names.append(save_path)

        return final_task_dict

    def load_gen_func(self):
        model = load_gen(self.cfg)
        return model

    def generate_env(self, model):
        z = torch.randn(1, self.cfg.training_generator.z_shape)
        env_layout = torch.argmax(model(z), dim=1)
        return env_layout
    
    def wrap_env(self, env):
        if self.cfg.env.visualize:
            render_mode = "human"
        else:
            render_mode = None
        layout_string = generate_obj_map(env, self.cfg.training_generator.map_element)
        # layout_string = clean_and_place_goal(layout_string)
        color_string = generate_color_map(layout_string)
        print("layout_string: ", layout_string)
        env = FullyObsWrapper(CustomMiniGridEnv(
            layout_str=layout_string,
            color_str=color_string,
            custom_mission="Navigate to the start position.",
            render_mode = render_mode
        ))
        # env.reset()
        # manual_control = ManualControl(env)  # Allows manual control for testing and visualization
        # manual_control.start()  # Start the manual control interface
        return env
    
    def wrap_env_from_text(self, file_path, max_steps):
        if self.cfg.env.visualize:
            render_mode = "human"
        else:
            render_mode = None
        env = FullyObsWrapper(CustomMiniGridEnv(
            txt_file_path=file_path,
            custom_mission="Navigate to the start position.",
            max_steps=max_steps,
            render_mode=render_mode
        ))
        return env
    
    def collect_data_trainer(self, env, wandb_run, validate, save_img, log_name, max_steps):
        cfg_local = copy.deepcopy(self.cfg)
        if validate:
            cfg_local.env.collect.episodes = 20
            save_img = (log_name != 'mini_task')
        if not os.path.exists(cfg_local.env.collect.data_save_path):
            data_collect_api(cfg_local, env, wandb_run, save_img, log_name, max_steps=max_steps)
        
    def decision_model(self):
        return random.choices([0, 1], weights=[0.35, 0.65])[0]
    # def save_data_to_buffer(self, data):
    #     pass

    def load_env_from_buffer(self, learning_buffer):
        """
        Load an environment from the learning buffer.
        If the buffer is empty, generate a new environment.
        """
        env_map = learning_buffer.sample()
        print("Loaded environment from learning buffer.")
        env = self.wrap_env(torch.tensor(env_map['map']).unsqueeze(0))
        # remove the entity from the learning buffer
        self.remove_from_learning_buffer(env_map['map'], learning_buffer)
        return env, env_map['map']
    

    def remove_from_learning_buffer(self, env_map, learning_buffer):
        # remove the entity from the learning buffer
        learning_buffer.remove(env_map)
            
            

    def add_into_learning_buffer(self, env_map, wm_loss, samples, learning_buffer):
        """
        Add the environment to the learning buffer.

        - Normal mode: Add only when wm_loss is higher than the threshold (indicating a challenging environment)
        - Ablation mode (disable_buffer_filter=True): Add all environments without filtering
        """
        disable_filter = getattr(self.cfg.training_generator.ablation, "disable_buffer_filter", False)
        threshold = getattr(self.cfg.training_generator, "learning_buffer_threshold", 0.5)

        entity = {
            "map": env_map,                      # Environment layout array
            "score": wm_loss[0]['val/observation_loss'],  # Unified world-model observation objective.
            "data": samples
        }

        # --- Ablation mode: no filtering, add all environments ---
        if disable_filter:
            print(f"[Ablation] Environment added to buffer unconditionally (wm_loss={entity['score']:.4f})")
            learning_buffer.add(entity)
            return

        # --- Normal mode: add only when wm_loss exceeds threshold ---
        if entity['score'] > threshold:
            print(f"Added to learning buffer (wm_loss={entity['score']:.4f} > {threshold})")
            learning_buffer.add(entity)
        else:
            print(f"Not added to learning buffer (wm_loss={entity['score']:.4f} ≤ {threshold})")




    def env_editor(self, env, dynamic_object, flip_ratio=0.15, max_attempts=20000):
        """
        Mutate the environment by:
        - Swapping movable elements (e.g., 8, 4, 5) to new valid positions
        - Flipping wall/floor tiles (1 ↔ 2) inside inner area only
        """

        if 4 or 5 in dynamic_object:
            key_door = True
        else:
            key_door = False

        if flip_ratio <= 0.03:
            env_original_layout = env.copy()
            env = self.wrap_env(torch.tensor(env_original_layout).unsqueeze(0))
            return env, env_original_layout
        for _ in range(max_attempts):
            env = env.copy()
            h, w = env.shape

            # Step 1: Flip 1s and 2s in the inner region
            inner_coords = [(i, j) for i in range(1, h-1) for j in range(1, w-1) if env[i, j] in (1, 2)]
            num_flips = int(len(inner_coords) * flip_ratio)
            flip_coords = random.sample(inner_coords, num_flips)

            for i, j in flip_coords:
                env[i, j] = 2 if env[i, j] == 1 else 1

            # Step 2: Move movable elements
            movable_coords = [(i, j) for i in range(1, h-1) for j in range(1, w-1) if env[i, j] in dynamic_object]
            empty_inner_coords = [(i, j) for i in range(1, h-1) for j in range(1, w-1)
                                if env[i, j] not in dynamic_object and (i, j) not in flip_coords]

            for (i, j) in movable_coords:
                val = env[i, j]
                env[i, j] = 1  # Clear old pos
                new_i, new_j = random.choice(empty_inner_coords)
                env[new_i, new_j] = val 
                empty_inner_coords.remove((new_i, new_j))
            

            if is_reachable(env, key_door=key_door):
                env_layout = env
                env = self.wrap_env(torch.tensor(env).unsqueeze(0))
                return env, env_layout
        # raise ValueError(f"After {max_attempts} attempts, no reachable environment could be generated.")
        return self.env_editor(env, dynamic_object, flip_ratio-0.02, max_attempts)


        

    # def generate_final_task(self):
    #     # Generate the final task
    #     layout_list = generate_valid_minigrid_with_key_door(15, 15, start=None, goal=None, wall_prob=0.5, max_attempts=1000)
    #     layout_string = layout_to_string(layout_list)
    #     color_string = generate_color_map(layout_string)
    #     combine_maps(layout_string, color_string, self.cfg.PPO.env_path)

    def del_env_data_file(self):
        """
        Remove the previous collected .npz file.

        Purpose:
            - Each data collection creates a new dataset at data_save_path.
            - If an old file exists, new data may mix with or overwrite it incorrectly.
            - Deleting ensures every round starts with a clean, fresh dataset.
        """
        if os.path.exists(self.cfg.env.collect.data_save_path):
            os.remove(self.cfg.env.collect.data_save_path)


    def generate_final_task_set(self, rows, cols, num_maps, wall_p_range,door_p_range, 
                            key_p_range, max_len,random_gen_max=1e5):
        """
        Generate a set of final tasks for wm performance evaluation.
        """
        print("++++++++++++++++++++++++++++++++++++ generating final task set... ++++++++++++++++++++++++++++++++++++++++++++++")
  
        final_task_set = self.generate_final_task(rows, cols, num_maps, save=True, wall_p_range=wall_p_range,
                            door_p_range=door_p_range, key_p_range=key_p_range, max_len=max_len, random_gen_max=random_gen_max)

        print(f"Final task set generated with {num_maps} maps.")
        return final_task_set

    def editing_process(self, env_string, main_run, old_params, fisher, learning_buffer, use_wandb):
            print("+++++++++++ Editing env +++++++++++")
            env_edited, env_layout = self.env_editor(env_string, self.cfg.training_generator.dynamic_objects)
            print("+++++++++++ Checking if add to buffer +++++++++++")
            self.cfg.attention_model.freeze_weight = True
            save_img = False
            self.collect_data_from_env(env_edited, wandb_run=main_run if use_wandb else None, validate=self.cfg.attention_model.freeze_weight, save_img = save_img, log_name='mini_task', max_steps=3e4)
            task_npz = np.load(self.cfg.attention_model.data_dir, allow_pickle=True)
            samples = {
                'obs': task_npz['a'],
                'obs_next': task_npz['b'],
                'act': task_npz['c'],
                'info': task_npz.get('f', None)  
            }
            wm_loss = self.validate_world_model(self.cfg, old_params, fisher, env_edited)
            self.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)
            
    # def assessing_performance_on_final_task(self, cfg, final_task_set, save_data=False, save_root=None):
    #     """
    #     Assess the performance of the trained model on the final task set.
    #     Optionally save collected data for each environment.
    #     """
    #     print("++++++++++++++ Assessing performance on final task set... +++++++++++++++")
    #     loss_set = []
    #     data_paths = []

    #     # 自动生成保存目录
    #     save_root = os.path.join(cfg.env.collect.data_folder, "final_task_envs")
    #     for i, final_task in enumerate(final_task_set):
    #         env = self.wrap_env(torch.tensor(final_task_set[final_task]).unsqueeze(0))
    #         save_path = os.path.join(save_root, f"env_{i}.npz")
    #         if not os.path.exists(save_path):
    #             cfg.env.collect.data_save_path = save_path
    #             self.collect_data_from_env(env, validate=True)
            
    #         cfg.attention_model.data_dir = save_path
    #         loss = self.validate_world_model(cfg, old_params=None, fisher=None, env_layout=final_task)
    #         loss_set.append(loss[0]['val/observation_loss'])
    #     avg_loss = sum(loss_set) / len(loss_set)
    #     print(f"Average loss on final task set: {avg_loss}")

    #     return avg_loss

    def assessing_performance_on_final_task(self, cfg, final_task_set, wandb_run, save_data=False, save_root=None):
        """
        Assess the performance of the trained model on the final task set.
        """
        print("++++++++++++++++++++++++++++++++++++ assessing performance on final task set... ++++++++++++++++++++++++++++++++++++++++++++++")
        # Load the trained model
        loss_set = []
        for i, final_task in enumerate(final_task_set):
            if i == 0:
                save_img = True
            else:
                save_img = False
            env = self.wrap_env(torch.tensor(final_task_set[final_task]).unsqueeze(0))
            self.collect_data_from_env(env, wandb_run, validate=True, save_img=save_img, log_name="final_task", max_steps=1e4)
            loss = self.validate_world_model(cfg, old_params=None, fisher=None, env_layout=final_task)
            loss_set.append(loss[0]['val/observation_loss'])
        avg_loss = sum(loss_set) / len(loss_set)
        print(f"Average loss on final task set: {avg_loss}")
        return avg_loss

    def train_policy_on_final_task(self, cfg, final_task_set):
        """
        Train the policy on the final task set.
        """
        print("++++++++++++++++++++++++++++++++++++ training policy on final task set... ++++++++++++++++++++++++++++++++++++++++++++++")
        # Load the trained model
        for final_task in final_task_set:
            cfg.PPO.wandb_run_name = f"final_task_{final_task}"
            cfg.PPO.env_path = os.path.join(LEVEL_PATH, 'minigrid', 'final_task', f'gen_final_task_{final_task}.txt')
            PPO_world_training.run_ppo_wm(cfg)

    # === Resume save/load ===
    def model_save_checkpoint(self, save_dir, step, old_params, fisher, learning_buffer, fisher_buffer):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({'step': step, 'old_params': old_params, 'fisher': fisher}, os.path.join(save_dir, 'wm.pt'))
        with open(os.path.join(save_dir, 'learning_buffer.pkl'), 'wb') as f:
            pickle.dump(learning_buffer.buffer, f)
        with open(os.path.join(save_dir, 'fisher_buffer.pkl'), 'wb') as f:
            pickle.dump(fisher_buffer.buffer, f)

    def model_load_checkpoint(save_dir, learning_buffer, fisher_buffer):
        ckpt = torch.load(os.path.join(save_dir, 'wm.pt'))
        with open(os.path.join(save_dir, 'learning_buffer.pkl'), 'rb') as f:
            learning_buffer.buffer = pickle.load(f)
        with open(os.path.join(save_dir, 'fisher_buffer.pkl'), 'rb') as f:
            fisher_buffer.buffer = pickle.load(f)
        return ckpt['step'], ckpt['old_params'], ckpt['fisher']
    
    def resume_training(self, cfg, learning_buffer, fisher_buffer):
        resume = cfg.attention_model.resume
        save_dir = cfg.attention_model.checkpoint_dir

        if resume and os.path.exists(os.path.join(save_dir, 'wm.pt')):
            start_step, old_params, fisher = self.model_load_checkpoint(save_dir, learning_buffer, fisher_buffer)
            print(f"Resuming training from step {start_step}")
        else:
            start_step = 0
            old_params, fisher = None, None
        return start_step, old_params, fisher

        
