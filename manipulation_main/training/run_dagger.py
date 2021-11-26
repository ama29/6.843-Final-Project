import argparse
import logging
import os.path

import gym
import stable_baselines as sb
from gym.wrappers import Monitor
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines.common.vec_env import DummyVecEnv

from manipulation_main.common import io_utils
from manipulation_main.training.wrapper import TimeFeatureWrapper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_env_expert(args):
    config = io_utils.load_yaml(args.config)
    os.mkdir(args.model_dir)
    # Folder for best models
    os.mkdir(args.model_dir + "/best_model")
    model_dir = args.model_dir
    algo = args.algo

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True
    if args.simple:
        logging.info("Simplified environment is set")
        config['simplified'] = True
    if args.shaped:
        logging.info("Shaped reward function is being used")
        config['reward']['shaped'] = True
    if args.timestep:
        config[algo]['total_timesteps'] = args.timestep
    if not args.algo == 'DQN':
        config['robot']['discrete'] = False
    else:
        config['robot']['discrete'] = True

    config[algo]['save_dir'] = model_dir
    if args.timefeature:
        env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make('gripper-env-v0', config=config))])
    else:
        env = DummyVecEnv(
            [lambda: Monitor(gym.make('gripper-env-v0', config=config), os.path.join(model_dir, "log_file"))])

    agent = sb.SAC.load(os.path.join(SCRIPT_DIR, config["model_path"]))
    return env, agent


def main(args):
    env, expert = get_env_expert(args)
    scratch = os.path.join(os.path.join(SCRIPT_DIR, "..", "..", "trained_models", "dagger", args.model_dir))
    trainer = SimpleDAggerTrainer(venv=env, scratch_dir=scratch, expert_policy=expert)
    trainer.train(total_timesteps=args.num_timesteps)

if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')
    main(train_parser.parse_args())
