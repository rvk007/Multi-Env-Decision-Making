import os
import random
from numpy import mod
import yaml
import gym
import highway_env
from gym.wrappers import Monitor

# TODO: Add TimeLimit and FrameSkip wrappers

class HighwayEnv:
    def __init__(self, env, config_dir, seed=42, max_random_noops=30, video_path=None):
        self.max_random_noops = max_random_noops
        
        self.envs = []
        for env_name in env.names:
            sub_env = gym.make(env_name)
            # sub_env.configure({'offscreen_rendering': True})
            if env.custom_config:
                sub_env.configure(self._load_config(os.path.join(config_dir, env_name + '.yaml')))
            sub_env.seed(seed)
            if video_path is not None:
                os.makedirs(video_path, exist_ok=True)
                sub_env = self._record_videos(sub_env, os.path.join(video_path, env_name), env.record_frequency)
            self.envs.append(sub_env)

        self.current_env = None
        self.current_env_idx = -1
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _record_videos(self, env, path, record_frequency):
        monitor = Monitor(
            env, path, force=True, video_callable=lambda episode: episode % record_frequency == 0
        )
        env.unwrapped.set_monitor(monitor)  # Capture intermediate frames
        return monitor
    
    def _validate(self):
        if self.current_env is None:
            raise AssertionError('No environment selected.')
    
    @property
    def current_env_name(self):
        self._validate()
        return self.current_env.__str__().split('-v0')[0].split('<')[-1]
    
    @property
    def num_envs(self):
        return len(self.envs)
    
    @property
    def observation_space(self):
        # FIXME: This currently assumes that all environments have the same observation space shape
        return self.envs[0].observation_space
    
    @property
    def action_space(self):
        # FIXME: This currently assumes that all environments have the same action space shape
        return self.envs[0].action_space

    def close(self):
        self.current_env = None
        self.current_env_idx = -1
        for env in self.envs:
            env.close()

    def _apply_random_noops(self):
        if self.max_random_noops <= 0:
            return
        # Do at least 1 no-op.
        self.current_env.reset()
        no_ops = random.randint(1, self.max_random_noops + 1)
        for _ in range(no_ops):
            _, _, game_over, _ = self.current_env.step(0)
            if game_over:
                self.current_env.reset()

    def reset(self):
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.current_env = self.envs[self.current_env_idx]
        # self._apply_random_noops()
        return self.current_env.reset()

    def step(self, action):
        self._validate()
        return self.current_env.step(action)

    def render(self):
        self._validate()
        self.current_env.render()


def create_env(config, config_dir, output_dir, mode='train'):
    return HighwayEnv(
        config.env, config_dir,
        seed=config.seed if mode == 'train' else config.seed + 1,
        max_random_noops=config.env.max_random_noops,
        video_path=output_dir if mode == 'test' and config.env.save_video else None
    )
