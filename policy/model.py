import os
import yaml
from stable_baselines3 import DQN
from environment.parse_env import make_env


def parse_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def make_model(config_file, env_name, video_path=None, tensorboard_log=None):
    if tensorboard_log is None:
        tensorboard_log = os.path.join('logs', env_name)
    
    # Load config
    config = parse_config(config_file)
    
    model = DQN(
        'MlpPolicy', make_env(env_name, video_path),
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=tensorboard_log,
        **config,
    )

    return model
