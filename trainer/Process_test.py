import os
from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from modelBased.common.utils import TRAINER_PATH
import hydra
from learning_buffer import EnvLearningBuffer
import numpy as np
import wandb
from fisher_buffer import FisherReplayBuffer
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def run(cfg: DictConfig):

    old_params, fisher = None, None
    fisher_buffer = FisherReplayBuffer(max_size=150000)
    learning_steps = cfg.training_generator.learning_steps
    support = Support.Support(cfg)
    # load the map from MAP sample
    env_database, file_dir = support.loading_tasks(cfg)
    learning_buffer = EnvLearningBuffer(max_size=cfg.training_generator.learning_buffer_size)

    for step in range(learning_steps):
        # within certain training steps
        print(f"Step {step+1} of {learning_steps}...")
        decision =  support.decision_model()
        decision = 0
        if len(fisher_buffer) > 0:
            replay_data = fisher_buffer.export_dict()
        else:
            replay_data = None

        if step % 5 == 0:
            save_img = True
        else:
            save_img = False
            
        # === Step 1: 决定 env 来源 ===
        if step == 0 or len(learning_buffer) == 0:

            env, env_layout = support.generate_env_from_generator(
                cfg, env_database[step], file_dir
            )
            cfg.attention_model.freeze_weight = True
            main_run = None
            if not os.path.exists(cfg.env.collect.data_save_path):
                support.collect_data_from_env(env, validate=cfg.attention_model.freeze_weight, wandb_run=main_run, save_img = save_img, log_name= 'mini_task', max_steps=3e4) 
            cur_old_params, cur_fisher = support.train_world_model(cfg, old_params, fisher, env_layout=None, replay_data=replay_data)
            cur_old_params, cur_fisher = None, None
            old_params, fisher = cur_old_params, cur_fisher
            task_npz_train = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples_train = {
                'obs': task_npz_train['a'],
                'obs_next': task_npz_train['b'],
                'act': task_npz_train['c'],
                'info': task_npz_train.get('f', None)
            }
            # fisher_buffer.update_with_random_by_ratio(samples_train, 0.3)
            print("+++++++++++ Editing env +++++++++++")
            env_edited, env_layout = support.env_editor(env_layout, cfg.training_generator.dynamic_objects)
            print("+++++++++++ Checking if add to buffer +++++++++++")
            cfg.attention_model.freeze_weight = True
            support.collect_data_from_env(env_edited, wandb_run=main_run, validate=cfg.attention_model.freeze_weight, save_img = save_img, max_steps=3e4)
            task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples = {
                'obs': task_npz['a'],
                'obs_next': task_npz['b'],
                'act': task_npz['c'],
                'info': task_npz.get('f', None)
            }
            wm_loss = support.validate_world_model(cfg, old_params, fisher, env_edited)
            support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)

#         elif decision == 0:
#             cfg.attention_model.freeze_weight = False
#             env, env_layout = support.generate_env_from_generator(
#                             cfg, env_database[step], file_dir
#                         )
#             cfg.attention_model.freeze_weight = True
#             support.collect_data_from_env(env, wandb_run=main_run, validate=cfg.attention_model.freeze_weight, save_img = save_img, max_steps=3e4)
#             task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
#             samples = {
#                 'obs': task_npz['a'],
#                 'obs_next': task_npz['b'],
#                 'act': task_npz['c'],
#                 'info': task_npz.get('f', None)  
#             }
#             wm_loss = support.validate_world_model(cfg, old_params, fisher, env_layout)
#             support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)

#         elif decision == 1:
#             cfg.attention_model.freeze_weight = False
#             # load the env from the learning buffer
#             env, env_string = support.load_env_from_buffer(learning_buffer)
#             support.collect_data_from_env(env, wandb_run=main_run, validate=cfg.attention_model.freeze_weight, save_img = save_img, max_steps=3e4)
#             cur_old_params, cur_fisher = support.train_world_model(cfg, old_params, fisher, env_layout=None, replay_data=replay_data)
#             task_npz_train = np.load(cfg.attention_model.data_dir, allow_pickle=True)
#             samples_train = {
#                 'obs': task_npz_train['a'],
#                 'obs_next': task_npz_train['b'],
#                 'act': task_npz_train['c'],
#                 'info': task_npz_train.get('f', None)  
#             }
#             fisher_buffer.update_with_random_by_ratio(samples_train, 0.4)
#             old_params, fisher = cur_old_params, cur_fisher
#             print("+++++++++++ Editing env +++++++++++")
#             env_edited, env_layout = support.env_editor(env_string, cfg.training_generator.dynamic_objects)
#             print("+++++++++++ Checking if add to buffer +++++++++++")
#             cfg.attention_model.freeze_weight = True
#             support.collect_data_from_env(env_edited, wandb_run=main_run, validate=cfg.attention_model.freeze_weight, save_img = save_img, max_steps=3e4)
#             task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
#             samples = {
#                 'obs': task_npz['a'],
#                 'obs_next': task_npz['b'],
#                 'act': task_npz['c'],
#                 'info': task_npz.get('f', None)  
#             }
#             wm_loss = support.validate_world_model(cfg, old_params, fisher, env_edited)
#             support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)
            

#         if step % 5 == 0:
#             rows = 20
#             cols = 20
#             num_maps = 3
#             final_task_set = support.generate_final_task_set(rows, cols, num_maps, 
#                                 wall_p_range=(0.1, 0.5),door_p_range=(0.075, 0.1), 
#                                 key_p_range=(0.1, 0.15), max_len=1e7,random_gen_max=1e5)
#             # === Step 2: Assessing performance on final task set ===
#             avg_loss = support.assessing_performance_on_final_task(cfg, final_task_set, wandb_run=main_run)

#         if step % 200 == 0 and step != 0:
#         # if step % 30 == 0:
#             support.train_policy_on_final_task(cfg, final_task_set)
#             main_run = wandb.init(
#             project='World_Model_Curriculum_Learning', 
#             entity='18920011663-king-s-college-london',
#             id=main_run.id,
#             resume="must"
#              )



# @hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
# def check_data(cfg: DictConfig):
#     import numpy as np
#     import pandas as pd

#     data = np.load(cfg.env.collect.data_save_path, allow_pickle=True)
#     obs      = data['a']  # shape: (N, H, W, C) or (N, F)
#     obs_next = data['b']
#     act      = data['c']
#     rew      = data['d']
#     done     = data['e']
#     info     = data.get('f', None)

#     # 若里面是图像或多维数组，先压平
#     N = obs.shape[0]
#     obs_flat      = obs.reshape(N, -1)
#     obs_next_flat = obs_next.reshape(N, -1)

#     # 创建 DataFrame
#     df = pd.DataFrame({
#         'obs': list(obs_flat),
#         'obs_next': list(obs_next_flat),
#         'action': act.flatten(),
#         'reward': rew.flatten(),
#         'done': done.flatten(),
#     })

#     # 加上 info 列（如果存在）
#     if info is not None:
#         df['info'] = list(info)

#     # 显示表格前几行和结构
#     print("Number of samples:", len(df))
#     print(df.head())
#     print(df.info())
#     print(df.describe(include='all'))

    
if __name__ == "__main__":
    run()
    # check_data()  # Uncomment to check data


