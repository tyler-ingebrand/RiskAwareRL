from gym.envs.registration import register

register(
    id='parkour-v0',
    entry_point='parkour_env.envs:ParkourEnv',
)