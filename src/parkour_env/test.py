
import gym
import parkour_env
from parkour_env.envs.parkour_env import ParkourEnv

# env = gym.make('parkour-v0')
try:
    env = gym.make("parkour-v0")
except gym.error.UnregisteredEnv:
    print("\n\n\nEnvironment not found. Try:\ncd parkour_env\npip install -e .\nwhich registers the environment, then try again.")
    exit(-1)

print("SS = ", env.observation_space)
print("AS = ", env.action_space)

obs = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()