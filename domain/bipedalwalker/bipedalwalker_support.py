import os


def read_layout_from_txt(file_path: str) -> str:
    """
    Read a BipedalWalker layout from text.
    Supports inline comments with '#', and multi-line tokens.
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Bipedal layout file not found: {file_path}")

    tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens.append(line)

    return " ".join(tokens).strip()


def wrap_env_from_text(file_path: str, render_mode=None):
    from domain.bipedalwalker.custom_bipedal_env import CustomBipedalEnv

    env = CustomBipedalEnv(render_mode=render_mode)
    layout_str = read_layout_from_txt(file_path)
    env.set_custom_layout_from_str(layout_str)
    return env

class BipedalHeuristicPolicy:
    """
    Heuristic policy for BipedalWalker, translated from gym box2d examples.
    Provides a walking prior that can be mixed with exploration noise.
    """
    def __init__(self, prior_weight=0.3):
        import numpy as np
        self.np = np
        self.prior_weight = prior_weight
        self.reset()

    def reset(self):
        self.STAY_ON_ONE_LEG, self.PUT_OTHER_DOWN, self.PUSH_OFF = 1, 2, 3
        self.SPEED = 0.29
        self.state = self.STAY_ON_ONE_LEG
        self.moving_leg = 0
        self.supporting_leg = 1 - self.moving_leg
        self.SUPPORT_KNEE_ANGLE = +0.1
        self.supporting_knee_angle = self.SUPPORT_KNEE_ANGLE

    def select_action(self, s, add_noise=True):
        # s is a 24-dim observation
        if isinstance(s, dict) and "image" in s:
            s = s["image"]
        a = self.np.zeros(4, dtype=self.np.float32)
        
        # State indicators
        moving_s_base = 4 + 5 * self.moving_leg
        supporting_s_base = 4 + 5 * self.supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        # Lidar indices s[14:24]
        # s[14] is straight down, s[23] is most forward.
        # We check front-down and front-facing lidar (e.g., indices 18-23 relative to s start)
        # If any object detected within ~1.0m, we boost the lift.
        lidar_s_base = 14
        front_lidar = s[lidar_s_base + 4 : lidar_s_base + 10]
        min_dist = self.np.min(front_lidar)
        
        # Calculate obstacle intensity weight (0.0 at 0.8m, 1.0 at 0.0m proximity)
        obstacle_weight = self.np.clip(1.25 * (0.8 - min_dist), 0.0, 1.0)

        if self.state == self.STAY_ON_ONE_LEG:
            # Proportional lift: Base 1.1 + variable boost based on obstacle closeness
            hip_targ[self.moving_leg] = 1.1 + (0.6 * obstacle_weight)
            # Proportional knee contraction: Base -0.6 + variable boost
            knee_targ[self.moving_leg] = -0.6 - (1.0 * obstacle_weight)

            self.supporting_knee_angle += 0.03
            if s[2] > self.SPEED:
                self.supporting_knee_angle += 0.03
            self.supporting_knee_angle = min(self.supporting_knee_angle, self.SUPPORT_KNEE_ANGLE)
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                self.state = self.PUT_OTHER_DOWN

        if self.state == self.PUT_OTHER_DOWN:
            hip_targ[self.moving_leg] = +0.1
            knee_targ[self.moving_leg] = self.SUPPORT_KNEE_ANGLE
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[moving_s_base + 4]:
                self.state = self.PUSH_OFF
                self.supporting_knee_angle = min(s[moving_s_base + 2], self.SUPPORT_KNEE_ANGLE)

        if self.state == self.PUSH_OFF:
            knee_targ[self.moving_leg] = self.supporting_knee_angle
            knee_targ[self.supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * self.SPEED:
                self.state = self.STAY_ON_ONE_LEG
                self.moving_leg = 1 - self.moving_leg
                self.supporting_leg = 1 - self.moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head straight
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = self.np.clip(0.5 * a, -1.0, 1.0)
        
        # Inject prior randomness
        if add_noise:
            # 30% priority -> e.g., mix 30% heuristic + 70% scaled uniform noise
            # or add Gaussian noise. We add huge scaled noise proportional to (1 - prior_weight)
            noise_scale = 1.0 - self.prior_weight
            random_a = self.np.random.uniform(-1, 1, size=4)
            a = self.prior_weight * a + noise_scale * random_a
            
            # An alternative: occasionally flip fully random
            # if self.np.random.rand() > self.prior_weight:
            #    a = random_a

        return self.np.clip(a, -1.0, 1.0)

def interpret_env(grid_1d_np, active_width=None):
    from generator.bipedal_env_designer import bipedal_array_to_layout_str
    
    layout_str = bipedal_array_to_layout_str(grid_1d_np, active_width=active_width)
    print(f"[Bipedal UED] Generated layout: {layout_str}")
    return layout_str, None
