import os
import time

import numpy as np

import hydra
import torch
import utils
from highway import HighwayEnv
from logger import Logger
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency_step,
            agent=cfg.agent.name
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = HighwayEnv(cfg.env, cfg.seed)
        self.eval_env = HighwayEnv(
            cfg.env, cfg.seed + 1,
            video_path=os.path.join(self.work_dir, 'eval_video') if cfg.env.save_video else None
        )
        self.best_eval_reward = 0

        cfg.agent.params.obs_shape = int(np.prod(self.env.observation_space.shape))
        cfg.agent.params.action_shape = self.env.action_space.n
        cfg.agent.params.num_env_paths = self.env.num_envs
        self.agent = hydra.utils.instantiate(cfg.agent)

        if cfg.prioritized_replay:
            # Initialize the prioritized replay buffer
            self.replay_buffer = PrioritizedReplayBuffer(
                self.env.observation_space.shape, cfg.replay_buffer_capacity,
                cfg.prioritized_replay_alpha, self.device
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.env.observation_space.shape, cfg.replay_buffer_capacity, self.device
            )

        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        eval_step = 0
        num_eval_episodes = 0
        while eval_step < self.cfg.num_eval_steps:
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                if np.random.rand() < self.agent.eval_eps:
                    action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, self.eval_env.current_env_idx)

                obs, reward, terminal, info = self.eval_env.step(action)
                # time_limit = 'TimeLimit.truncated' in info
                # done = info['game_over'] or time_limit
                done = terminal or info['crashed']
                episode_reward += reward
                episode_step += 1
                eval_step += 1

            average_episode_reward += episode_reward
            num_eval_episodes += 1

        average_episode_reward /= num_eval_episodes
        self.logger.log(
            'eval/episode_reward', average_episode_reward, self.step
        )
        self.logger.dump(self.step, ty='eval')

        if self.cfg.save_checkpoint and average_episode_reward > self.best_eval_reward:
            self.best_eval_reward = average_episode_reward
            torch.save(self.agent.critic.state_dict(), 'policy.pt')

    def run(self):
        print('Running workspace')
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()

                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.start_training_steps), ty='train')

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # evaluate agent periodically
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                print('Evaluating agent')
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

            steps_left = self.cfg.num_exploration_steps + self.cfg.start_training_steps - self.step
            bonus = (1.0 - self.cfg.min_eps) * steps_left / self.cfg.num_exploration_steps
            bonus = np.clip(bonus, 0., 1. - self.cfg.min_eps)
            self.agent.eps = self.cfg.min_eps + bonus

            self.logger.log('train/eps', self.agent.eps, self.step)

            # sample action for data collection
            if np.random.rand() < self.agent.eps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, self.env.current_env_idx)

            # run training update
            if self.step >= self.cfg.start_training_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, terminal, info = self.env.step(action)

            # time_limit = 'TimeLimit.truncated' in info
            # done = info['game_over'] or time_limit
            done = terminal or info['crashed']

            terminal = float(terminal)
            # terminal = 0 if time_limit else terminal

            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, terminal, self.env.current_env_idx)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
