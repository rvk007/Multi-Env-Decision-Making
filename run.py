import os
import argparse
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy
from environment.parse_env import make_env, parse_config
from policy.model import make_model


def train_model(model_config, env_name, env_config_dir, total_timesteps, save_file, tensorboard_logs):
    train_env = make_env(env_name, env_config_dir)
    model = make_model(model_config, train_env, tensorboard_logs)

    model.learn(total_timesteps)
    model.save(save_file)

    return model


def evaluate_model(model, env_name, env_config_dir, num_episodes):
    return evaluate_policy(
        model, make_env(env_name, env_config_dir), n_eval_episodes=num_episodes
    )


def test_model(model, env_name, env_config_dir, num_episodes, record=True):
    env = make_env(env_name, env_config_dir, record=record)

    for episode in tqdm(num_episodes, desc='Test episodes'):
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
    
    env.close()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Environment name')
    parser.add_argument('-t', '--timesteps', type=int, default=1e6, help='Number of timesteps for training')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--test_episodes', type=int, default=3, help='Number of test episodes')
    parser.add_argument('--env_config', default=os.path.join(BASE_DIR, 'environment'), help='Directory containing the environment config')
    parser.add_argument('--model', default=os.path.join(BASE_DIR, 'policy', 'dqn.yaml'), help='Model yaml path')
    parser.add_argument('--weights', default=os.path.join(BASE_DIR, 'weights'), help='Path to store the trained models')
    parser.add_argument('--logs', default=os.path.join(BASE_DIR, 'logs'), help='Path to store the tensorboard logs')
    args = parser.parse_args()

    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    if args.logs is None:
        args.logs = os.path.join('logs', args.env)

    print('Training model...')
    model = train_model(args.model, args.env, args.env_config, int(args.timesteps), os.path.join(args.weights, args.env), args.logs)

    print('Evaluating model...')
    mean_reward, std_reward = evaluate_model(model, args.env, args.env_config, args.eval_episodes)
    print('\n\nMean Reward:', mean_reward)
    print('Std Reward:', std_reward)

    print('Testing model...')
    test_model(model, args.env, args.env_config, args.test_episodes)
