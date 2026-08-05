import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

# --- Core constants ---
FPS = 50
SCALE = 30.0  # Keep fixed to preserve stable Box2D dynamics.
MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE
INITIAL_RANDOM = 5
HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

# --- Viewport constants ---
VIEWPORT_W = 900 
VIEWPORT_H = 400
# Global-view scaling tuned for the 900-pixel viewport width.
GLOBAL_SCALE = 9.5 

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10
TERRAIN_STARTPAD = 20
FRICTION = 2.5

# --- Physics definitions ---
HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,
    restitution=0.0,
)

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

from gymnasium.envs.box2d.bipedal_walker import BipedalWalker

class BipedalWalkerCustom(BipedalWalker):
    def __init__(self, render_mode=None, hardcore=False):
        super().__init__(render_mode=render_mode, hardcore=hardcore)
        self.custom_layout = None
        self.terrain_roughness = 1.0
        self.use_global_view = True # Enable the global overview by default.
        
    def set_custom_layout(self, layout_list):
        self.custom_layout = layout_list

    def reset(self, *, seed=None, options=None):
        """
        Custom reset to support spawning at any x-coordinate via the `options` dictionary.
        """
        # Call base reset to rebuild terrain and objects
        obs, info = super().reset(seed=seed, options=options)

        # Extract spawn_x from options if present
        spawn_x = options.get("spawn_x") if options is not None else None

        if spawn_x is not None and hasattr(self, "terrain_y"):
            # Ensure index is within range of recorded terrain heights
            idx = int(spawn_x / TERRAIN_STEP)

            idx = max(0, min(idx, len(self.terrain_y) - 1))
            ground_y = self.terrain_y[idx]

            # Move Hull (center mechanism) to spawn_x
            self.hull.position = (spawn_x, ground_y + 1.2)
            self.hull.linearVelocity = (0, 0)
            self.hull.angularVelocity = 0

            # Move legs as well to prevent extreme joint tension on first step
            for leg in self.legs:
                leg.position = (spawn_x, ground_y + 0.5)
                leg.linearVelocity = (0, 0)
                leg.angularVelocity = 0
            
            # 4. Refresh the observation with a zero-action step.
            obs, _, _, _, _ = super().step(np.zeros(4, dtype=np.float32))

        return obs, info

    def _generate_terrain(self, hardcore):
        if self.custom_layout is None:

            return super()._generate_terrain(hardcore)

        GRASS, STUMP, STAIRS, PIT = 0, 1, 2, 3
        y = TERRAIN_HEIGHT
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        velocity = 0.0
        
        layout_idx = 0
        token_counter = 0
        state = GRASS
        param = 0
        oneshot = True
        self.terrain_roughness = 1.0
        
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)
            
            while token_counter <= 0:
                if layout_idx < len(self.custom_layout):
                    type_char, param = self.custom_layout[layout_idx]
                    layout_idx += 1
                    oneshot = True
                    if type_char == 'R':
                        self.terrain_roughness = max(0.0, float(param))
                        continue
                    if type_char == 'G':
                        state = GRASS
                        token_counter = int(param)
                    elif type_char == 'S':
                        state = STUMP
                        token_counter = 1
                    elif type_char == 'P':
                        state = PIT
                        token_counter = 1
                    elif type_char == 'T':
                        state = STAIRS
                        token_counter = 1
                else:
                    state = GRASS
                    token_counter = 100
                    oneshot = True

            if state == GRASS:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.terrain_roughness * self.np_random.uniform(-1, 1) / SCALE
                y += velocity
                oneshot = False

            elif state == PIT and oneshot:
                pit_width = int(param)
                poly = [(x, y), (x + TERRAIN_STEP, y), (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP), (x, y - 4 * TERRAIN_STEP)]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                
                self.fd_polygon.shape.vertices = [(p[0] + TERRAIN_STEP * pit_width, p[1]) for p in poly]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                
                token_counter = pit_width + 2
                original_y = y
                oneshot = False

            elif state == PIT and not oneshot:
                y = original_y
                if token_counter > 1: y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                stump_h = float(param)
                poly = [(x, y), (x + stump_h * TERRAIN_STEP, y), (x + stump_h * TERRAIN_STEP, y + stump_h * TERRAIN_STEP), (x, y + stump_h * TERRAIN_STEP)]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                token_counter = 0
                oneshot = False

            elif state == STAIRS and oneshot:
                stair_h_dir = 1 if param > 0 else -1
                stair_width = 4
                stair_steps = abs(int(param))
                if stair_steps == 0: stair_steps = 3
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_h_dir) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_h_dir) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_h_dir) * TERRAIN_STEP),
                        (x + (s * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_h_dir) * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                token_counter = stair_steps * stair_width
                oneshot = False

            elif state == STAIRS and not oneshot:
                s_idx = (stair_steps * stair_width - token_counter - 1)
                n = s_idx // stair_width
                y = original_y + (n * stair_h_dir) * TERRAIN_STEP
                
            self.terrain_y.append(y)
            token_counter -= 1

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [(self.terrain_x[i], self.terrain_y[i]), (self.terrain_x[i + 1], self.terrain_y[i + 1])]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    # --- Override rendering to support the full-course global view ---
    def render(self):
        if self.render_mode is None: return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled('pygame is not installed') from e

        # Adjust the camera dynamically when global view is enabled.
        # The full track is about 93.3 physics units long; scaling by 11.0
        # yields roughly 1026 pixels, which fits a 1200px-wide window well.
        scl = GLOBAL_SCALE if self.use_global_view else SCALE
        scroll_val = 0 if self.use_global_view else self.scroll
        
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        self.surf.fill((215, 215, 255))  # Background color

        # Draw clouds. In global view, remove parallax so they span the full course.
        offset_x = 20 if self.use_global_view else -scroll_val * scl
        for poly, x1, x2 in self.cloud_poly:
            # Remove the 0.5 factor so cloud coordinates align 1:1 with the ground.
            cloud_scl = scl if self.use_global_view else scl * 0.5
            cloud_off = offset_x if self.use_global_view else offset_x * 0.5
            scaled_poly = [((p[0]*cloud_scl + cloud_off), p[1]*scl) for p in poly]
            pygame.draw.polygon(self.surf, color=(255, 255, 255), points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (255, 255, 255))

        # Draw terrain.
        for poly, color in self.terrain_poly:
            if not self.use_global_view:
                if poly[1][0] < scroll_val: continue
                if poly[0][0] > scroll_val + VIEWPORT_W / scl: continue
            
            # In global view, shift the origin slightly left for framing.
            offset_x = 20 if self.use_global_view else -scroll_val * scl
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([(coord[0] * scl) + offset_x, coord[1] * scl])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        # Draw the goal flag (red triangle).
        flag_x = (TERRAIN_STEP * (TERRAIN_LENGTH-TERRAIN_GRASS)) * scl + (20 if self.use_global_view else -scroll_val * scl)
        flag_y1 = TERRAIN_HEIGHT * scl
        flag_y2 = flag_y1 + 50  # Top of the flagpole
        pygame.draw.line(self.surf, (0,0,0), (flag_x, flag_y1), (flag_x, flag_y2), 2)
        
        # Coordinates of the red triangular flag.
        flag_poly = [
            (flag_x, flag_y2), 
            (flag_x, flag_y2 - 15), 
            (flag_x + 25, flag_y2 - 7.5)
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=flag_poly)

        # Draw all dynamic bodies, including the robot.
        offset_x = 20 if self.use_global_view else -scroll_val * scl
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                color1, color2 = obj.color1, obj.color2
                if type(f.shape) is circleShape:
                    pos = (trans * f.shape.pos)
                    pygame.draw.circle(self.surf, color=color1, center=(pos[0]*scl + offset_x, pos[1]*scl), radius=f.shape.radius*scl)
                else:
                    path = [((trans * v)[0]*scl + offset_x, (trans * v)[1]*scl) for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, color1)
                        pygame.draw.polygon(self.surf, color=color2, points=path, width=1)
                    else:
                        pygame.draw.aaline(self.surf, start_pos=path[0], end_pos=path[1], color=color1)

        self.surf = pygame.transform.flip(self.surf, False, True)
        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        
        # Always return the frame buffer so data collection and W&B logging
        # can consume a valid image instead of None.
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
        )
