import sys
from modelBased.common.utils import PROJECT_ROOT
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
from modelBased.policy_training.PPO import PPO
import hydra
from datetime import datetime
from modelBased.common import utils

from omegaconf import DictConfig, OmegaConf 
from modelBased.world_model import AttentionWM_support
from modelBased.world_model import Embedding_support
from modelBased.world_model import MLP_support
import wandb
from modelBased.policy_training.PPO import preprocess_observation 
import time



# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

def get_destination(obs, episode, maxstep, destination):
    """
    from the obs state, check if the agent has reached the destination
    and return done and reward

    1.object:("unseen": 0,  "empty": 1, "wall": 2, "door": 4, "key": 5, "goal": 8, "agent": 10)
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10

    2. color:
    "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5

    3. status
    State, 0: open, 1: closed, 2: locked

    check from wrappers.py full_obs-->encode
    """
    if obs[0, destination[0], destination[1]] == 10:
        # agent has reached the destination
        if episode >= maxstep:
            done = True
            reward = 0
        else:
            reward = 1 - 0.9 * (episode / maxstep)
            done = True
    else:
        done = False
        reward = 0
    return done, reward


def find_position(array, target):
    """
    Find the position of a target value in a 3D numpy array.
    
    Args:
        array (np.ndarray): The 3D array to search.
        target (tuple): The target value to locate (e.g., (8, 1, 0)).

    Returns:
        tuple: The position (x, y) of the target in the array if found, otherwise None.
    """
    # Find all indices where the value matches the target
    target = np.array(target).reshape(-1, 1, 1)
    result = np.argwhere((array == target).all(axis=0))

    # Check if any matches were found
    if result.size > 0:
        return tuple(result[0])  # Return the first match as a tuple (x, y)
    else:
        return None

def process_data(state, maks_size):
    agent_postion_yx = utils.get_agent_position(state)
    state_masked = utils.extract_masked_state(state, maks_size, agent_postion_yx) 
    return state_masked

def evaluate_policy(policy, env, episodes, obs_norm_values):
    """
    Evaluate the policy by running it in the environment for a number of episodes.
    """
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()[0]['image']
        state = utils.ColRowCanl_to_CanlRowCol(obs)
        done = False
        ep_reward = 0
        for _ in range(env.max_steps):
            state_tensor = torch.tensor(utils.normalize_obs(state, obs_norm_values)).to(device)
            action, _, _, _, _ = policy.select_action(state_tensor.flatten())
            obs, reward, done, _, _ = env.step(action)
            state = utils.ColRowCanl_to_CanlRowCol(obs['image'])
            ep_reward += reward
            if done:
                break
        total_reward += ep_reward
    return total_reward / episodes

# add the function add objects into the inventory
def add_object_to_inventory(delta_state, info):
    """
    Add an object to the agent's inventory in the environment.
    It should depend on the changes between next state and current state.
    the delta: whether the delta includes a minus key value 
    assuming for key valus == 5
    
    Args:
        for a keydoor environment
        info['carraying_key'] (bool): Whether the agent is carrying a key.
    """

    if (delta_state[0,:,:] == -5).any():
        info['carrying_key'] = True
    return info




@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def training_agent_wm(cfg: DictConfig):
    regret = run_ppo_wm(cfg)
    return regret

def run_ppo_wm(cfg):
    hparams = cfg
    
    # 1. World Model
    hparams_world_model = hparams.attention_model

    MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
    # Initialize the world model.
    module_class = MODEL_MAPPING.get(hparams_world_model.model_type.lower())
    if module_class is not None:
        model = module_class(
            hparams_world_model.data_type,  
            hparams_world_model.grid_shape, 
            hparams_world_model.attention_mask_size, 
            hparams_world_model.embed_dim, 
            hparams_world_model.num_heads
        )
    else:
        print(f"Model type: {hparams_world_model.model_type} not supported")
        exit()
    utils.load_model_weight(model, hparams_world_model.model_save_path)
    model.eval() 
    


    # 2. PPO
    # hyperparameters
    # compute regret
    hparams_PPO = hparams.PPO
    compute_regret = hparams_PPO.compute_regret
    if compute_regret:
        regret_eval_freq = hparams_PPO.get("regret_eval_freq", 5000)
        regret_eval_episodes = hparams_PPO.get("regret_eval_episodes", 5)
        real_policy_path = hparams_PPO.get("real_policy_path")


    start_time = datetime.now().replace(microsecond=0)
    lr_actor = hparams_PPO.lr_actor
    lr_critic = hparams_PPO.lr_critic
    gamma = hparams_PPO.gamma
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    action_std = hparams_PPO.action_std
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    max_training_timesteps = hparams_PPO.max_training_timesteps
    save_model_freq = hparams_PPO.save_model_freq
    max_ep_len = hparams_PPO.max_ep_len
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    checkpoint_path = hparams_PPO.checkpoint_path
    env_path = hparams_PPO.env_path
    visualize_flag = hparams_PPO.visualize
    env_type =  hparams_PPO.env_type
    use_wandb = hparams_PPO.use_wandb
    wandb_run_name = hparams_PPO.wandb_run_name
    

    if use_wandb:
        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        sub_run = wandb.init(project='Trainer_policy', entity='18920011663-king-s-college-london',name=wandb_run_name, reinit=True)
    else:
        sub_run = None

    # training_agent()

    if visualize_flag:
        visualize = utils.Visualization(hparams_world_model)
    # 3. Real environment
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=4000, render_mode=None))
    # 4. Initialize training
    i_episode = 0
    update_timestep = max_ep_len   # update policy every n timesteps
    print_freq = max_ep_len * 2
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    step_penalty = -0.9 / max_ep_len
    final_norm_regret = None
    
    # action space dimension
    if has_continuous_action_space:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 5
    else:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 5
    state_dim = np.prod(env.observation_space['image'].shape)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    if compute_regret:
        real_policy_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)
        real_policy_agent.load(real_policy_path)

    

    # training loop
    while time_step <= max_training_timesteps:

        print(f'time step: {time_step}')
        state_init = env.reset()[0]['image']
        if time_step == 0 and sub_run is not None:
            img = env.get_frame()
            sub_run.log({"final_tasks": wandb.Image(img)})
        state_0 = utils.ColRowCanl_to_CanlRowCol(state_init)
        goal_position_yx = find_position(state_0, (8, 1, 0)) # find the goal position
        current_ep_reward = 0
        info = {'carrying_key': False}
        for t in range(1, int(max_ep_len + 1)):
            need_update = False
            # self.buffer.states = [state.squeeze(0) if state.dim() > 1 else state for state in self.buffer.states]
            if t==1:
                # state = utils.normalize_obs(state_0, hparams_world_model.obs_norm_values)
                state_0 = torch.tensor(state_0).to(device)
            
            state_norm = utils.normalize_obs(state_0, hparams_world_model.obs_norm_values)
            action, state_buffer, action_buffer, action_logprob, state_val = ppo_agent.select_action(state_norm.flatten()) # state is the dimension of flatten

 
            state_masked = process_data(state_0.clone(), hparams_world_model.attention_mask_size)
            with torch.no_grad():
                delta_masked, _ = model(state_masked, action, info)
            
            state_pre_masked = state_masked + delta_masked
            if visualize_flag:
                visualize.compare_states(state_masked, state_pre_masked, action, t, True)
            # delta_state_pre = delta_state_pre.to(dtype=torch.float32)
            # denorm the state
            # state_pre_denorm = utils.denormalize_obj(state_pre, hparams_world_model.obs_norm_values)

            state_pre_masked = utils.map_obs_to_nearest_value(state_pre_masked, 
                                                              hparams_world_model.valid_values_obj,
                                                              hparams_world_model.valid_values_color,
                                                              hparams_world_model.valid_values_state)

            info = add_object_to_inventory((state_pre_masked - state_masked), info)
            agent_postion_yx = utils.get_agent_position(state_0)
            state_pre = utils.put_back_masked_state(state_pre_masked, state_0, hparams_world_model.attention_mask_size, agent_postion_yx)
            

                
            state_0 = state_pre
            # obtain reward from the state representation & done
            done, reward = get_destination(state_0, t, max_ep_len, goal_position_yx)
            reward += step_penalty
            # saving reward and is_terminals
            ppo_agent.save_buffer(state_buffer, action_buffer, action_logprob, state_val, reward, done)
            

            time_step += 1
            current_ep_reward += reward
    
            # # update PPO agent
            # if time_step % update_timestep == 0:
            #     if len(ppo_agent.buffer.rewards) == len(ppo_agent.buffer.state_values) and len(ppo_agent.buffer.rewards) > 1:
            #         ppo_agent.update()
            #     else:
            #         print(f"[WARNING] Buffer mismatch, skipping update. Rewards={len(ppo_agent.buffer.rewards)}, StateValues={len(ppo_agent.buffer.state_values)}")
            #         ppo_agent.buffer.clear()  # Force-clear the buffer to avoid stale accumulation.



            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
    
            if time_step % print_freq == 0 and print_running_episodes > 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                if use_wandb:
                    sub_run.log({"average_reward": print_avg_reward})

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            #region
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
            #endregion

            # compute the regret
            #region
            if compute_regret and time_step % regret_eval_freq == 0:
                
                print(f"Evaluating regret at timestep {time_step}...")
                R_wm = evaluate_policy(ppo_agent, env, regret_eval_episodes, hparams_world_model.obs_norm_values)
                R_real = evaluate_policy(real_policy_agent, env, regret_eval_episodes, hparams_world_model.obs_norm_values)
                regret = R_real - R_wm
                norm_regret = regret / max(R_real, 1e-8)
                final_regret = regret
                final_norm_regret = norm_regret
                print(f"[Regret @ timestep {time_step}] Real: {R_real:.2f}, WM: {R_wm:.2f}, Regret: {regret:.2f}, Norm: {norm_regret:.2%}")

                if use_wandb:
                    sub_run.log({
                        "regret": regret,
                        "normalized_regret": norm_regret,
                        "real_policy_reward_in_eva": R_real,
                        "wm_policy_reward_in_eva": R_wm,
                        "timestep": time_step
                    })
            #endregion

            # break; if the episode is over
            # Update is triggered in either of these two cases:
            if time_step % update_timestep == 0 or done or t >= max_ep_len:
                need_update = True

            if need_update:
                if len(ppo_agent.buffer.rewards) == len(ppo_agent.buffer.state_values) and len(ppo_agent.buffer.rewards) >= 2:
                    ppo_agent.update()
                else:
                    print(f"[WARNING] Buffer mismatch, skipping update. Rewards={len(ppo_agent.buffer.rewards)}, StateValues={len(ppo_agent.buffer.state_values)}")
                    ppo_agent.buffer.clear()
                break  # Break immediately after update to avoid buffer contamination.

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1
    env.close()
    if use_wandb:
        sub_run.finish()
    if compute_regret: 
        return final_norm_regret

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def training_agent_real_env(cfg: DictConfig):
    run_training_real_env(cfg)

def run_training_real_env(cfg):
    # parameters
    hparams = cfg
    hparams_PPO = hparams.PPO
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    max_ep_len =  hparams_PPO.max_ep_len
    max_training_timesteps = hparams_PPO.max_training_timesteps
    print_freq = max_ep_len * 10
    save_model_freq = hparams_PPO.save_model_freq
    update_timestep = max_ep_len // 2  # update policy every n timesteps
    print_running_reward = 0
    print_running_episodes = 0
    start_time = datetime.now().replace(microsecond=0)
    env_type =  hparams_PPO.env_type
    wandb_run_name = hparams_PPO.wandb_run_name

    time_step = 0
    i_episode = 0
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    checkpoint_path = hparams_PPO.checkpoint_path

    # param for agent
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    gamma = hparams_PPO.gamma
    lr_actor = hparams_PPO.lr_actor  # learning rate for actor network
    lr_critic = hparams_PPO.lr_critic  # learning rate for critic network
    action_std = hparams_PPO.action_std  # default std for action distribution (can be overwritten by action_std_decay_rate)
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    env_path = hparams_PPO.env_path
    use_wandb = hparams_PPO.use_wandb
    step_penalty = 0

    if use_wandb:
        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        subrun = wandb.init(project='final_task_policy', entity='18920011663-king-s-college-london', name=wandb_run_name, reinit=True)


    # state space dimension
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=4000, render_mode=None))
    
    state_dim = np.prod(env.observation_space['image'].shape)

    # action space dimension
    # action space dimension
    if has_continuous_action_space:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 6
    else:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 6

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)


    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        state = preprocess_observation(state[0]['image']).to(device)

        for t in range(1, int(max_ep_len + 1)):

            # select action with policy
            action, state_buffer, action_buffer, action_logprob, state_val = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            reward += step_penalty
            done = terminated or truncated
            state = preprocess_observation(state['image']).to(device)
            # saving reward and is_terminals
            ppo_agent.save_buffer(state_buffer, action_buffer, action_logprob, state_val, reward, done)


            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                if len(ppo_agent.buffer.rewards) > 1:
                    ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0 and print_running_episodes > 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                if use_wandb:
                    subrun.log({"average_reward": print_avg_reward})

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done or t == max_ep_len:
                if len(ppo_agent.buffer.rewards) >= 2:
                    ppo_agent.update()

                break
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()
    if use_wandb:
        subrun.finish()


def run_policy_evaluation(cfg: DictConfig):
    hparams = cfg
    # 1. World Model
    hparams_world_model = hparams.attention_model
    # 2. PPO
    # hyperparameters
    # compute regret
    hparams_PPO = hparams.PPO

    lr_actor = hparams_PPO.lr_actor
    lr_critic = hparams_PPO.lr_critic
    gamma = hparams_PPO.gamma
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    action_std = hparams_PPO.action_std
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    checkpoint_path = hparams_PPO.checkpoint_path_wm
    env_path = hparams_PPO.env_path
    env_type =  hparams_PPO.env_type
    episodes = hparams_PPO.get("episodes_eval")

    # 3. Real environment
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=4000, render_mode=None))
 
    # action space dimension
    if has_continuous_action_space:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 6
    else:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 6
    state_dim = np.prod(env.observation_space['image'].shape)

    policy_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std)
    policy_agent.load(checkpoint_path)
    Reward = evaluate_policy(policy_agent, env, episodes, hparams_world_model.obs_norm_values)
    return Reward

if __name__ == "__main__":
    use_wandb = True
    if use_wandb:
        import wandb

        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        wandb.init(project='WM Attention PPO', entity='svea41')

    training_agent_wm()

    if use_wandb:
        wandb.finish()
