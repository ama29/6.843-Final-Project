import argparse
import logging
import os.path

import gym
import stable_baselines as sb
from gym.wrappers import Monitor
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util import logger
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from manipulation_main.common import io_utils
from manipulation_main.training.custom_obs_policy import TransposeNatureCNN
from manipulation_main.training.wrapper import TimeFeatureWrapper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")


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


def main(args):
    os.chdir(BASE_DIR)  # all filepaths are relative to repo root in other files
    env, expert = get_env_expert(args)
    scratch = os.path.join(os.path.join(SCRIPT_DIR, "..", "..", "trained_models", "dagger", args.model_dir))

    # define policy to be learned
    ob_space = env.observation_space
    ac_space = env.action_space
    # Default network arch is nature, which is also used in this repo. Might be good to verify later arch matches
    # TODO: is lr_schedule used? For now using constant lr scheduler
    train_policy = ActorCriticCnnPolicy(observation_space=ob_space, action_space=ac_space, lr_schedule=lambda x: 0.005,
                                        features_extractor_class=TransposeNatureCNN)
    log_dir = os.path.join(BASE_DIR, args.model_dir, "dagger", "logs")
    bc_trainer = BC(observation_space=ob_space, action_space=ac_space, demonstrations=None, policy=train_policy)

    # construct dagger instance and train
    dagger_logger = logger.configure(log_dir)
    trainer = SimpleDAggerTrainer(venv=env, scratch_dir=scratch, expert_policy=expert, bc_trainer=bc_trainer,
                                  custom_logger=dagger_logger)
    trainer.train(total_timesteps=args.num_timesteps, rollout_round_min_timesteps=100)
    trainer.save_trainer()


if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model', type=str)
    train_parser.add_argument("--num_timesteps", type=int)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')
    main(train_parser.parse_args())
