import torch
import numpy as np
import pandas as pd
import gym
import hydra
import imageio  # Import imageio to save GIFs
from modelBased.common.utils import PROJECT_ROOT, normalize_obs
from modelBased.policy_training.PPO import PPO
from omegaconf import DictConfig, OmegaConf
from minigrid.wrappers import FullyObsWrapper
from domain.minigrid.minigrid_custom_env import *
from domain.minigrid import minigrid_support as minigrid_utils
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv

# Set device to CPU or CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

#################################### Testing ###################################
@hydra.main(version_base=None, config_path= str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def test(cfg: DictConfig):
    validate_policy(cfg)

def validate_policy(cfg):
    print("============================================================================================")

    ################## Hyperparameters ##################
    has_continuous_action_space = cfg.PPO.has_continuous_action_space
    action_std = cfg.PPO.action_std
    lr_actor = cfg.PPO.lr_actor
    lr_critic = cfg.PPO.lr_critic
    gamma = cfg.PPO.gamma
    K_epochs = cfg.PPO.K_epochs
    eps_clip = cfg.PPO.eps_clip
    total_test_episodes = cfg.PPO.total_test_episodes
    max_ep_len = cfg.PPO.max_ep_len
    render = cfg.PPO.render
    checkpoint_path = cfg.PPO.checkpoint_path
    env_type = cfg.PPO.env_type
    obs_norm_values = cfg.attention_model.obs_norm_values
    visualize_obs = minigrid_utils.Visualization(cfg.attention_model)
    visualize_flag = cfg.PPO.visualize
    env_path = cfg.PPO.env_path
    # save experiment results
    # gif
    save_path_gif = cfg.PPO.save_path_gif
    save_gif = cfg.PPO.save_gif  # Set to True to save the GIF
    gif_filename = os.path.join(save_path_gif, "ppo_test.gif")
    # csv
    save_path_csv = cfg.PPO.save_path_csv
    csv_output = cfg.PPO.save_csv  # Set to True to save rewards
    data = []  # Store episode rewards
    csv_filename = os.path.join(save_path_csv, "test_PPO_rewards.csv")  # CSV file path
    
    #####################################################
    # Load environment with RGB rendering enabled
    env = FullyObsWrapper(CustomMiniGridEnv(txt_file_path=env_path, 
                                            custom_mission="Find the key and open the door.",
                                            max_steps=4000, render_mode="rgb_array"))  # Use "rgb_array" for rendering

    # Action space dimension
    if has_continuous_action_space:
        action_dim = 3 if env_type == 'empty' else env.action_space
    else:
        action_dim = 3 if env_type == 'empty' else env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)

    # Initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # Load pre-trained weights
    print("Loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state_init = env.reset()[0]['image']
        state = minigrid_utils.ColRowCanl_to_CanlRowCol(state_init)
        
        frames = []  # Store frames for the GIF

        for step in range(1, max_ep_len + 1):
            state_norm = normalize_obs(state, obs_norm_values)
            if isinstance(state_norm, np.ndarray):
                state_norm = torch.tensor(state_norm, dtype=torch.float32).to(device)
                # ---- Select action ----
            action_out = ppo_agent.select_action(state_norm.flatten())

        if isinstance(action_out, tuple):
            action = action_out[0]
        else:
            action = action_out

        if isinstance(action, torch.Tensor):
            action = action.item()
            state_next, reward, done, trunc, _ = env.step(action)
            ep_reward += reward
            state_next = minigrid_utils.ColRowCanl_to_CanlRowCol(state_next['image'])

            # Save frames for the GIF
            if save_gif:
                frame = env.render()  # Get RGB frame
                frames.append(frame)

            if visualize_flag:
                print(f'Episode: {ep} \t Step: {step} \t Reward: {round(ep_reward, 2)}')
                visualize_obs.compare_states(state, state_next, action, f'ep:{ep} step:{step}', True)

            state = state_next

            if done or trunc:
                break

        # Save the GIF after an episode (optional: only for the first episode)
        if save_gif and ep == 1:
            imageio.mimsave(gif_filename, frames, fps=10)
            print(f"Saved test episode GIF as {gif_filename}")
            # Store episode reward for CSV

        if csv_output:
            data.append([ep, ep_reward])

        # Clear buffer
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print(f'Episode: {ep} \t\t Reward: {round(ep_reward, 2)}')

    # Save rewards to CSV
    if csv_output:
        df = pd.DataFrame(data, columns=["Episode", "Reward"])
        df.to_csv(csv_filename, index=False, header=True)

    env.close()
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    print(f"Average test reward : {round(avg_test_reward, 2)}")
    print("============================================================================================")


if __name__ == '__main__':
    test()
