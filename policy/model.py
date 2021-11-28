import os
import yaml
from stable_baselines3 import DQN
from environment.parse_env import make_env


def parse_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def make_model(config_file, env, tensorboard_log):
    # Load config
    config = parse_config(config_file)
    
    model = DQN(
        'MlpPolicy', env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=float(config['learning_rate']),
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        target_update_interval=config['target_update_interval'],
        exploration_fraction=config['exploration_fraction'],
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    return model
