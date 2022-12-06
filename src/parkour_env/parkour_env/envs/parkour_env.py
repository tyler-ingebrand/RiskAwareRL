from PIL import Image
import numpy
import numpy as np
import gym
from dm_env import StepType
from gym.spaces import Box
#@title All `dm_control` imports required for this tutorial

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import ant
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks
from matplotlib import pyplot as plt


class ParkourEnv(gym.Env):
    def __init__(self):
        walker = ant.Ant(observable_options={'sensors_rangefinder': dict(enabled=True),})
                                             # 'egocentric_camera':dict(enabled=False)})
        arena = corridor_arenas.GapsCorridor(platform_length=2.0)
        task = corridor_tasks.RunThroughCorridor(
            walker=walker,
            arena=arena,
            walker_spawn_position=(0.9, 0, 0),
            target_velocity=3.0,
            physics_timestep=0.005,
            control_timestep=0.03,
        )
        self.env = composer.Environment(
                        task=task,
                        time_limit=10,
                        random_state=np.random.RandomState(42),
                        strip_singleton_obs_buffer_dim=True,
                    )
        self.observation_space = self._format_observation_space(self.env.observation_spec())
        self.action_space = Box(self.env.action_spec().minimum, self.env.action_spec().maximum)
        self.frames = []

    def _format_observation_space(self, obs_space):
        exclude_image = {}
        for key in obs_space:
            if key != 'walker/egocentric_camera' and len(obs_space[key].shape) == 1:
                exclude_image[key] = obs_space[key]

        total_length = sum(sum(l for l in exclude_image[val].shape) for val in exclude_image)

        return Box(numpy.zeros(total_length), numpy.ones(total_length))


    def _format_observation(self, obs):
        exclude_image = {}
        for key in obs:
            if key != 'walker/egocentric_camera' and len(obs[key].shape) == 1:
                exclude_image[key] = obs[key]
        return numpy.concatenate([exclude_image[key] for key in exclude_image])


    def reset(self):
        ret = self.env.reset()
        return self._format_observation(ret.observation)

    def step(self, action):
        ret = self.env.step(action)
        return self._format_observation(ret.observation), ret.reward, ret.step_type==StepType.LAST, ret

    def render(self, mode: str = 'human' ):
        pixels = []
        for camera_id in [0, 2]:
            pixels.append(self.env.physics.render(camera_id=camera_id, width=240, ))
        img = Image.fromarray(np.hstack(pixels))
        plt.axis('off')
        plt.imshow(img)
        plt.pause(0.01)
