import os
import random
import gym
import highway_env
from gym.wrappers import Monitor

# TODO: Add TimeLimit and FrameSkip wrappers

class HighwayEnv:
    def __init__(self, envs, seed=42, max_random_noops=30, video_path=None):
        self.max_random_noops = max_random_noops

        self.envs = [gym.make(env) for env in envs]
        for env in self.envs:
            env.seed(seed)
        
        self.recording = video_path is not None
        if video_path:
            os.makedirs(video_path, exist_ok=True)
            self.envs = [
                self._record_videos(env, os.path.join(video_path, env.__str__().split('<')[1]))
                for env in self.envs
            ]

        self.current_env = None
        self.current_env_idx = -1
    
    def _validate(self):
        if self.current_env is None:
            raise AssertionError('No environment selected.')
    
    def _record_videos(self, env, path):
        monitor = Monitor(env, path, force=True, video_callable=lambda episode: False)
        env.unwrapped.set_monitor(monitor)  # Capture intermediate frames
        return monitor
    
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
        if not self.recording:
            self._apply_random_noops()
        return self.current_env.reset()

    def step(self, action):
        self._validate()
        return self.current_env.step(action)

    def render(self):
        self._validate()
        self.current_env.render()
