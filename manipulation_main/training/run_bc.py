import argparse
import os.path

from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.scripts.common.demonstrations import load_expert_trajs
from imitation.util import logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from manipulation_main.training.custom_obs_policy import TransposeNatureCNN
from manipulation_main.training.imitation_utils import BASE_DIR, get_env_expert

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def rollout_expert_traj(save_path: str, expert, env, min_timesteps: int, min_episodes: int):
    sample_until = rollout.make_sample_until(
        min_timesteps,
        min_episodes
    )
    rollout.rollout_and_save(save_path, expert, env, sample_until)


def main(args):
    os.chdir(BASE_DIR)  # all filepaths are relative to repo root in other files
    bc_dir = os.path.join(BASE_DIR, args.model_dir, "bc")

    env, expert = get_env_expert(args)

    # define policy to be learned
    ob_space = env.observation_space
    ac_space = env.action_space
    # Default network arch is nature, which is also used in this repo. Might be good to verify later arch matches
    # TODO: is lr_schedule used? For now using constant lr scheduler
    train_policy = ActorCriticCnnPolicy(observation_space=ob_space, action_space=ac_space, lr_schedule=lambda x: 0.005,
                                        features_extractor_class=TransposeNatureCNN)

    # load/create rollouts
    rollout_file = os.path.join(BASE_DIR, args.rollout_file)
    if not os.path.exists(rollout_file):
        # do rollouts
        rollout_expert_traj(rollout_file, expert, env, min_timesteps=args.num_timesteps,
                            min_episodes=args.num_episodes)

    # load rollouts
    demos = load_expert_trajs(rollout_file)

    log_dir = os.path.join(bc_dir, "logs")
    bc_logger = logger.configure(log_dir)
    bc_trainer = BC(observation_space=ob_space, action_space=ac_space, demonstrations=demos, policy=train_policy,
                    custom_logger=bc_logger)

    # train bc
    bc_trainer.train()
    bc_trainer.save_policy(policy_path=os.path.join(bc_dir, "final_policy.pt"))


if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model', type=str)
    train_parser.add_argument("--num_timesteps", type=int)
    train_parser.add_argument("--num_episodes", type=int)
    train_parser.add_argument("--rollout_file", type=str, required=True)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')
    main(train_parser.parse_args())
