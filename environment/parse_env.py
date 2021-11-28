import yaml
import gym
from visual import record_videos


def parse_config(env_name):
    with open(f'{env_name}_config.yaml') as f:
        return yaml.safe_load(f)


def make_env(env_name, record=False, video_path=None):
    env = gym.make(f'{env_name}-v0')
    env.configure(parse_config(env_name))
    if record:
        if video_path is None:
            video_path = f'{env_name}_videos'
        env = record_videos(env, path=video_path)
    return env
