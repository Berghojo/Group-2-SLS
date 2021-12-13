from absl import app
from sls import Env, Runner
from sls.agents import *
from sls.agents.QAgent import QAgent

_CONFIG = dict(
    episodes=600,
    screen_size=16,
    minimap_size=64,
    visualize=False,
    train=False,
    agent=DuelingDeepQAgent,
    load_path='./models/'
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
