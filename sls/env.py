from pysc2.env import sc2_env
from pysc2.lib import features


class Env:
    def __init__(self, screen_size=32, minimap_size=32, visualize=False):
        self.sc2_env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=screen_size, minimap=minimap_size),
                use_feature_units=True
            ),
            step_mul=8,
            visualize=visualize
        )

    def reset(self):
        return self.preprocess_obs(self.sc2_env.reset())

    def step(self, action):
        return self.preprocess_obs(self.sc2_env.step([action]))

    def preprocess_obs(self, timsteps):
        # Any kind of preprocessing can take place here
        return timsteps[0]
