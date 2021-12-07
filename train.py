import os
import time

import numpy as np
import torch

import utils
from highway import create_env
from policy import create_agent, create_replay_buffer
from logger import Logger
from video import VideoRecorder


torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config, env_dir, output_dir, device, logger, policy_path):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.logger = logger
        self.policy_path = policy_path

        # Create environments
        self.env = create_env(config.env, env_dir, output_dir)
        self.eval_env = create_env(config.env, env_dir, output_dir, mode='eval')

        # Create agent
        self.agent = create_agent(config, self.env, device)

        # Create replay buffer
        self.replay_buffer = create_replay_buffer(config, self.env, device)
        
        # Create video recorder
        self.video_recorder = VideoRecorder(
            output_dir if config.env.save_video else None, fps=config.env.fps
        )

        self.best_eval_reward = 0
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        eval_step = 0
        num_eval_episodes = 0
        while eval_step < self.config.num_eval_steps:
            obs = self.eval_env.reset()
            self.video_recorder.init(self.eval_env.current_env, enabled=True)
            
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
                done = terminal or info['crashed']
                self.video_recorder.record(self.eval_env.current_env)
                episode_reward += reward
                episode_step += 1
                eval_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.eval_env.current_env_name}_{num_eval_episodes}.mp4')
            num_eval_episodes += 1

        average_episode_reward /= num_eval_episodes
        self.logger.log(
            'eval/episode_reward', average_episode_reward, self.step
        )
        self.logger.dump(self.step, ty='eval')

        if self.config.save_checkpoint and average_episode_reward > self.best_eval_reward:
            self.best_eval_reward = average_episode_reward
            torch.save(self.agent.critic.state_dict(), self.policy_path)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.config.num_train_steps:
            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()

                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(self.step, save=(self.step > self.config.start_training_steps), ty='train')

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # evaluate agent periodically
            if self.step > 0 and self.step % self.config.eval_frequency == 0:
                print('Evaluating agent')
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

            steps_left = self.config.num_exploration_steps + self.config.start_training_steps - self.step
            bonus = (1.0 - self.config.min_eps) * steps_left / self.config.num_exploration_steps
            bonus = np.clip(bonus, 0., 1. - self.config.min_eps)
            self.agent.eps = self.config.min_eps + bonus

            self.logger.log('train/eps', self.agent.eps, self.step)

            # sample action for data collection
            if np.random.rand() < self.agent.eps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, self.env.current_env_idx)

            # run training update
            if self.step >= self.config.start_training_steps:
                for _ in range(self.config.num_gradient_steps):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, terminal, info = self.env.step(action)
            done = terminal or info['crashed']
            terminal = float(terminal)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, terminal, self.env.current_env_idx)

            obs = next_obs
            episode_step += 1
            self.step += 1
        
        torch.save(
            self.agent.critic.state_dict(),
            f'{os.path.splitext(self.policy_path)[0]}_last.pt'
        )


def agent_trainer(config, env_dir, output_dir, device, policy_path):
    logger = Logger(
        output_dir,
        save_tb=config.log_save_tb,
        log_frequency=config.log_frequency_step,
    )
    return Trainer(config, env_dir, output_dir, device, logger, policy_path)
