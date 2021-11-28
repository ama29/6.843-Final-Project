import logging
import os

import gym
import stable_baselines as sb
from gym.wrappers import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from manipulation_main.common import io_utils
from manipulation_main.training.wrapper import TimeFeatureWrapper

_script_dir = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(_script_dir, "..", "..")


def get_env_expert(args):
    config_dir = os.path.join(BASE_DIR, args.config)
    config = io_utils.load_yaml(config_dir)
    # os.mkdir(args.model_dir)
    # Folder for best models
    # os.mkdir(args.model_dir + "/best_model")
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
        # force = true to overwrite
        env = DummyVecEnv(
            [lambda: Monitor(gym.make('gripper-env-v0', config=config),
                             os.path.join(model_dir, "dagger", "gym_log_file"),
                             force=True)])

    agent = sb.SAC.load(os.path.join(BASE_DIR, args.model))
    return env, agent


def get_test_env(args):
    config_dir = os.path.join(BASE_DIR, args.config)
    config = io_utils.load_yaml(config_dir)
    task = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config, evaluate=True, test=False)])
    return task
