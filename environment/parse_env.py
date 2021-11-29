import os
import yaml
import gym
import highway_env
from visual import record_videos


def parse_config(env_config):
    with open(f'{env_config}_config.yaml') as f:
        return yaml.safe_load(f)


def make_env(env_name, env_config_dir, record=False, video_path=None):
    env = gym.make(f'{env_name}-v0')
    env.configure(parse_config(os.path.join(env_config_dir, env_name)))
    if record:
        if video_path is None:
            video_path = os.path.join('videos', f'{env_name}_videos')
        env = record_videos(env, path=video_path)
    return env
