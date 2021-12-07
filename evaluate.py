import numpy as np
import torch

import utils
from highway import create_env
from policy import create_agent
from logger import Logger


torch.backends.cudnn.benchmark = True


class Evaluator:
    def __init__(self, config, env_dir, output_dir, policy_path, device, logger, render_video):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.logger = logger

        # Create environments
        config.max_random_noops = 0
        self.env = create_env(
            config.env, env_dir, output_dir, mode='test', offscreen_rendering=not render_video
        )

        # Create agent
        self.agent = create_agent(config, self.env, device)
        self.agent.load(policy_path)

        self.best_eval_reward = 0
        self.step = 0

    def run(self):
        average_episode_reward = 0
        eval_step = 0
        num_eval_episodes = 0
        while eval_step < self.config.num_eval_steps:
            obs = self.env.reset()
            
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                if np.random.rand() < self.agent.eval_eps:
                    action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, self.env.current_env_idx)

                obs, reward, terminal, info = self.env.step(action)
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

        if self.config.save_checkpoint and average_episode_reward > self.best_eval_reward:
            self.best_eval_reward = average_episode_reward
        
        print('\n\nMean Reward Obtained:', average_episode_reward)
        print('Max Reward Obtained:', self.best_eval_reward)


def agent_evaluator(config, env_dir, output_dir, policy, device, render_video):
    logger = Logger(
        output_dir,
        save_tb=config.log_save_tb,
        log_frequency=config.log_frequency_step,
    )
    return Evaluator(config, env_dir, output_dir, policy, device, logger, render_video)
