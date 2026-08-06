from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from modelBased.common.utils import TRAINER_PATH
import hydra
from learning_buffer import EnvLearningBuffer
import numpy as np
import wandb
from fisher_buffer import FisherReplayBuffer



'''
Process
1. load the generator (MAP selected env)
2. use the generator to generator env 
3. main function 
    if first iteration:
        train WM on the first env
        edit this env
        evaluation the WM on the edited env
        whether or not put it in the learning buffer
    else:
        decition model D(d): 1/0
        if d == 0:
            generator new env & collect data
            validation the WM performance on the new env
            whether or not put it in the learning buffer
        if d == 1:
            load the env from the learning buffer 
            train WM on the env
            edit this env
            evaluate the WM on the edited env
            whether or not put it in the learning buffer

'''


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_ablation")
def run(cfg: DictConfig):
    use_wandb = cfg.training_generator.use_wandb
    if use_wandb:
        run_name = "CL_WM_without_RB_2" 
        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        main_run = wandb.init(project='World_Model_Curriculum_Learning', entity='18920011663-king-s-college-london', name=run_name, reinit=True)
        wandb.define_metric("curriculum_step")  
        wandb.define_metric("final_task_performance", step_metric="curriculum_step")
    old_params, fisher = None, None
    support = Support.Support(cfg)
    fisher_buffer = FisherReplayBuffer(cfg.attention_model.fisher_buffer_size)
    learning_steps = cfg.training_generator.learning_steps
    # load the map from MAP sample
    print("++++++++++++++++++++++++++++++++++++  loading tasks... ++++++++++++++++++++++++++++++++++++++++++++++")
    env_database, file_dir = support.loading_tasks(cfg)

    learning_buffer = EnvLearningBuffer(max_size=cfg.training_generator.learning_buffer_size)

    # old_params, fisher = support.resume_training(cfg, learning_buffer, fisher_buffer)

    for step in range(learning_steps):
        # within certain training steps
        print(f"Step {step+1} of {learning_steps}...")
        decision =  support.decision_model()
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

            cfg.attention_model.freeze_weight = False
            support.collect_data_from_env(env, validate=cfg.attention_model.freeze_weight, wandb_run=main_run if use_wandb else None, save_img = save_img, log_name= 'mini_task', max_steps=3e4) 
            cur_old_params, cur_fisher = support.train_world_model(cfg, old_params, fisher, env_layout=None, replay_data=replay_data)
            old_params, fisher = cur_old_params, cur_fisher
            task_npz_train = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples_train = {
                'obs': task_npz_train['a'],
                'obs_next': task_npz_train['b'],
                'act': task_npz_train['c'],
                'info': task_npz_train.get('f', None)
            }

            fisher_buffer.update_combined(samples_train, 0.3, 0.5)
            
            if not cfg.training_generator.ablation.disable_editor:
                print("+++++++++++ Editing env +++++++++++")
                support.editing_process(env_database[step], main_run, old_params, fisher, learning_buffer, use_wandb=use_wandb)

        elif decision == 0:
            print("+++++++++++ Generating new env +++++++++++")
            cfg.attention_model.freeze_weight = False
            env, env_layout = support.generate_env_from_generator(
                            cfg, env_database[step], file_dir
                        )
            cfg.attention_model.freeze_weight = True
            support.collect_data_from_env(env, wandb_run=main_run if use_wandb else None, validate=cfg.attention_model.freeze_weight, save_img = save_img, log_name='mini_task', max_steps=3e4)
            task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples = {
                'obs': task_npz['a'],
                'obs_next': task_npz['b'],
                'act': task_npz['c'],
                'info': task_npz.get('f', None)  
            }
            wm_loss = support.validate_world_model(cfg, old_params, fisher, env_layout)
            support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)

        elif decision == 1:
            print("+++++++++++ Using env from learning buffer +++++++++++")
            cfg.attention_model.freeze_weight = False
            # load the env from the learning buffer
            env, env_string = support.load_env_from_buffer(learning_buffer)
            support.collect_data_from_env(env, wandb_run=main_run if use_wandb else None, validate=cfg.attention_model.freeze_weight, save_img = save_img, log_name='mini_task', max_steps=3e4)
            cur_old_params, cur_fisher = support.train_world_model(cfg, old_params, fisher, env_layout=None, replay_data=replay_data)
            task_npz_train = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples_train = {
                'obs': task_npz_train['a'],
                'obs_next': task_npz_train['b'],
                'act': task_npz_train['c'],
                'info': task_npz_train.get('f', None)  
            }
            fisher_buffer.update_combined(samples_train, 0.3, 0.5)
            old_params, fisher = cur_old_params, cur_fisher
            if not cfg.training_generator.ablation.disable_editor:
                print("+++++++++++ Editing env +++++++++++")
                support.editing_process(env_string, main_run, old_params, fisher, learning_buffer, use_wandb)

        if step % 3 == 0:
            rows = 15
            cols = 15
            num_maps = 3
            if cfg.attention_model.env_type == 'key_door':
                final_task_set = support.generate_final_task_set(rows, cols, num_maps, 
                                    wall_p_range=(0.1, 0.5),door_p_range=(0.075, 0.1), 
                                    key_p_range=(0.1, 0.15), max_len=1e7,random_gen_max=1e5)
            elif cfg.attention_model.env_type == 'empty':
                # empty env
                final_task_set = support.generate_final_task_set(rows, cols, num_maps, 
                                    wall_p_range=(0.1, 0.4),door_p_range=(0.0, 0.0), 
                                    key_p_range=(0.0, 0.0), max_len=1e7,random_gen_max=1e5)
            # === Step 2: Assessing performance on final task set ===
            avg_loss = support.assessing_performance_on_final_task(cfg, final_task_set, wandb_run=main_run if use_wandb else None)
            if use_wandb:
                
                main_run.log({
                    "curriculum_step": step,             
                    "final_task_performance": avg_loss
                })


        # # === Step 3: policy performance on the final task set ===
        # if step % 200 == 0 and step != 0:
        # # if step % 30 == 0:
        #     support.train_policy_on_final_task(cfg, final_task_set)
        #     main_run = wandb.init(
        #     project='World_Model_Curriculum_Learning', 
        #     entity='18920011663-king-s-college-london',
        #     id=main_run.id,
        #     resume="must"
        #     )

    if use_wandb:
        main_run.finish()

@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def check_data(cfg: DictConfig):
    import numpy as np
    import pandas as pd

    data = np.load(cfg.env.collect.data_save_path, allow_pickle=True)
    obs      = data['a']  # shape: (N, H, W, C) or (N, F)
    obs_next = data['b']
    act      = data['c']
    rew      = data['d']
    done     = data['e']
    info     = data.get('f', None)

    # 若里面是图像或多维数组，先压平
    N = obs.shape[0]
    obs_flat      = obs.reshape(N, -1)
    obs_next_flat = obs_next.reshape(N, -1)

    # 创建 DataFrame
    df = pd.DataFrame({
        'obs': list(obs_flat),
        'obs_next': list(obs_next_flat),
        'action': act.flatten(),
        'reward': rew.flatten(),
        'done': done.flatten(),
    })

    # 加上 info 列（如果存在）
    if info is not None:
        df['info'] = list(info)

    # 显示表格前几行和结构
    print("Number of samples:", len(df))
    print(df.head())
    print(df.info())
    print(df.describe(include='all'))

    
if __name__ == "__main__":
    run()
    # check_data()  # Uncomment to check data


