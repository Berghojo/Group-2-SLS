from absl import app
from sls import Env, Runner
from sls.agents import *
from sls.A2Cmodel import A2CModel

_CONFIG = dict(
    episodes=5000,
    screen_size=16,
    minimap_size=16,
    visualize=True,
    train=True,
    agent=A2CAgent,
    load_path='./pickles/'
)


def create_env_func():
    return Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        network=A2CModel(_CONFIG['train'])
    )

    runner = Runner(
        agent=agent,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        config=_CONFIG
    )

    runner.run(episodes=_CONFIG['episodes'], env_func=create_env_func)


if __name__ == "__main__":
    app.run(main)