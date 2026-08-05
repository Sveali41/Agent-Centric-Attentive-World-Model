import metaworld
import numpy as np
import gymnasium as gym

class CustomMetaWorldEnv(gym.Env):
    """
    Custom wrapper for Meta-World to support Curriculum / UED generation (Mini tasks),
    and zero-shot validation (Target tasks).
    """
    def __init__(self, env_name='sweep-into-v3', render_mode=None):
        super().__init__()
        self.env_name = env_name
        self.ml1 = metaworld.ML1(env_name)
        
        # Instantiate the unwrapped Meta-World environment
        if render_mode is not None:
            self.env = self.ml1.train_classes[env_name](render_mode=render_mode)
        else:
            self.env = self.ml1.train_classes[env_name]()
            
        # Set default task from ML1 (this is the Target task setting)
        self.default_task = self.ml1.train_tasks[0]
        self.env.set_task(self.default_task)
        
        # Generator customization parameters
        self.custom_obj_pos = None 
        self.custom_goal_pos = None
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def set_custom_layout(self, obj_pos=None, goal_pos=None):
        """
        Called by the UED curriculum generator to control mini-task difficulty
        through continuous object and goal coordinates.
        obj_pos: [x, y, z] initial object position
        goal_pos: [x, y, z] target goal position
        """
        if obj_pos is not None:
            self.custom_obj_pos = np.array(obj_pos, dtype=np.float32)
        else:
            self.custom_obj_pos = None
            
        if goal_pos is not None:
            self.custom_goal_pos = np.array(goal_pos, dtype=np.float32)
        else:
            self.custom_goal_pos = None

    def set_custom_layout_from_str(self, layout_str: str):
        """
        Called by the UED curriculum generator using a string-encoded continuous layout.
        Expected format: "obj_x,obj_y,obj_z;goal_x,goal_y,goal_z"
        Example: "0.0,0.6,0.02;0.2,0.8,0.0"
        Passing "" or "TARGET" restores the default randomized layout.
        """
        if not layout_str or layout_str.upper() == "TARGET":
            self.set_custom_layout(None, None)
            return
            
        if ";" in layout_str:
            try:
                obj_str, goal_str = layout_str.split(";")
                obj_pos = [float(x.strip()) for x in obj_str.split(",")]
                goal_pos = [float(x.strip()) for x in goal_str.split(",")]
                self.set_custom_layout(obj_pos, goal_pos)
            except Exception as e:
                print(f"Warning: Failed to parse layout_str '{layout_str}'. Fallback to Target. Error: {e}")
                self.set_custom_layout(None, None)

    def reset(self, seed=None, options=None):
        """
        Reset the environment. If `custom_layout` is set, override the default
        initialization with the provided task parameters.
        """
        # 1. Run the base reset so MuJoCo refreshes internal state.
        result = self.env.reset()
        
        # Old Gym API (MetaWorld v2 often returns just obs) vs Gymnasium API
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # 2. If external layout parameters are provided, overwrite the object and goal.
        if self.custom_obj_pos is not None or self.custom_goal_pos is not None:
            
            # (A) Override the target goal.
            if self.custom_goal_pos is not None:
                self.env._target_pos = self.custom_goal_pos.copy()
                try:
                    # Keep the MuJoCo render site in sync with the target position.
                    site_id = self.env.model.site_name2id('goal')
                    self.env.model.site_pos[site_id] = self.custom_goal_pos
                except Exception:
                    pass 
            
            # (B) Override the object's initial position.
            if self.custom_obj_pos is not None:
                self.env.obj_init_pos = self.custom_obj_pos.copy()
                try:
                    self.env._set_obj_xyz(self.custom_obj_pos)
                except AttributeError:
                    pass
            
            # 3. Recompute the observation after applying overrides.
            obs = self.env._get_obs()

        # Attach parsed entities to `info` for the P2E world model.
        entities = self._vector_to_entity_dict(obs)
        info['entities'] = entities
            
        return obs, info

    def _vector_to_entity_dict(self, obs_vector):
        """
        Pack the continuous 39D vector state into the entity-attention format
        expected by the world model.
        Based on the standard SawyerXYZ layout:
        [0:4]   end-effector xyz and gripper openness
        [4:18]  object position and quaternion
        """
        target = self.env._target_pos if hasattr(self.env, '_target_pos') else np.zeros(3)
        return {
            "robot": obs_vector[0:4],
            "object": obs_vector[4:18], 
            "goal": target
        }
    
    def step(self, action):
        step_returns = self.env.step(action)
        
        # Support both the legacy Gym 4-tuple and the Gymnasium 5-tuple API.
        if len(step_returns) == 4:
            obs, reward, done, info = step_returns
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_returns
            
        info['entities'] = self._vector_to_entity_dict(obs)
        
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(*args, **kwargs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

if __name__ == "__main__":
    # Basic smoke test.
    print("Testing CustomMetaWorldEnv...")
    env = CustomMetaWorldEnv('sweep-into-v3', render_mode=None)
    
    print("\n--- [Target Task Mode] (Default Randomization) ---")
    obs, info = env.reset()
    print("Random Object Pos:", env.env.obj_init_pos)
    print("Random Goal Pos:", env.env._target_pos)
    
    print("\n--- [Mini Task Mode] (UED Curriculum Parameterized) ---")
    custom_obj = np.array([0.0, 0.6, 0.02])
    custom_goal = np.array([0.2, 0.8, 0.0])
    
    env.set_custom_layout(obj_pos=custom_obj, goal_pos=custom_goal)
    obs, info = env.reset()
    
    print("Customized Object Pos:", env.env.obj_init_pos)
    print("Customized Goal Pos:", env.env._target_pos)
    print("\n[Entity Extraction Format Example]")
    print(info['entities'])
    
    env.close()
