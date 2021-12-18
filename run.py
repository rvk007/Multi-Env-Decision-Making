import os
import shutil
import argparse

import torch

import utils
from train import agent_trainer
from evaluate import agent_evaluator


torch.backends.cudnn.benchmark = True


def train(config, env_dir, output_dir, device, policy):
    print('Training Policy...')
    trainer = agent_trainer(config, env_dir, output_dir, device, policy)
    trainer.run()


def test(config, env_dir, output_dir, policy, device, render_video):
    print('Evaluating Policy...')
    evaluator = agent_evaluator(config, env_dir, output_dir, policy, device, render_video)
    evaluator.run()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', default=os.path.join(BASE_DIR, 'config.yaml'),
        help='Config file path'
    )
    parser.add_argument(
        '--env_dir', default=os.path.join(BASE_DIR, 'env_configs'),
        help='Directory containing environment configs'
    )
    parser.add_argument(
        '--output', default=os.path.join(BASE_DIR, 'experiments'),
        help='Directory to save logs and results'
    )
    parser.add_argument(
        '-m', '--mode', default='train', choices=['train', 'test'],
        help='Run script in train or test mode'
    )
    parser.add_argument(
        '-p', '--policy', default=None,
        help='Path to policy network. Used only in test mode. If no path given, a policy is picked from the output directory'
    )
    parser.add_argument(
        '--render_video', action='store_true',
        help='Render video of agent interacting with environment. Works only in test mode'
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = utils.read_config(args.config)
    utils.set_seed_everywhere(config.seed)

    args.output = os.path.join(args.output, config.experiment_name)
    os.makedirs(args.output, exist_ok=True)

    if args.policy is None:
        args.policy = os.path.join(args.output, f'{config.experiment_name}.pt')

    if args.mode == 'train':
        train(config, args.env_dir, args.output, device, args.policy)
    else:
        test(config, args.env_dir, args.output, args.policy, device, args.render_video)

    # Store configs in the output directory
    utils.dump_config(dict(config), os.path.join(args.output, 'config.yaml'))
    if config.env.custom_config:
        shutil.copytree(args.env_dir, os.path.join(args.output, os.path.basename(args.env_dir)))
