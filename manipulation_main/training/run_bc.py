import argparse
import os.path

from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.scripts.common import train
from imitation.scripts.common.demonstrations import load_expert_trajs
from imitation.util import logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from manipulation_main.training.custom_obs_policy import TransposeNatureCNN
from manipulation_main.training.imitation_utils import BASE_DIR, get_env_expert, get_test_env

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def rollout_expert_traj(save_path: str, expert, env, min_timesteps: int, min_episodes: int):
    sample_until = rollout.make_sample_until(
        min_timesteps,
        min_episodes
    )
    # set unwrap to false otherwise wrapper fails to see info within trajectory
    rollout.rollout_and_save(save_path, expert, env, sample_until, unwrap=False)


def main(args):
    os.chdir(BASE_DIR)  # all filepaths are relative to repo root in other files
    bc_dir = os.path.join(BASE_DIR, args.model_dir, "bc")

    train_env, expert = get_env_expert(args)

    # define policy to be learned
    ob_space = train_env.observation_space
    ac_space = train_env.action_space
    # Default network arch is nature, which is also used in this repo. Might be good to verify later arch matches
    # TODO: is lr_schedule used? For now using constant lr scheduler
    train_policy = ActorCriticCnnPolicy(observation_space=ob_space, action_space=ac_space, lr_schedule=lambda x: 0.005,
                                        features_extractor_class=TransposeNatureCNN)

    # load/create rollouts
    rollout_file = os.path.join(BASE_DIR, args.rollout_file)
    if not os.path.exists(rollout_file):
        # do rollouts
        print("Generating rollouts")
        rollout_expert_traj(rollout_file, expert, train_env, min_timesteps=args.num_timesteps,
                            min_episodes=args.num_episodes)
    else:
        print("Rollout file exists. Using existing rollouts")

    # load rollouts
    print("Loading rollouts")
    demos = load_expert_trajs(rollout_file, n_expert_demos=None)

    log_dir = os.path.join(bc_dir, "logs")
    bc_logger = logger.configure(log_dir)
    bc_trainer = BC(observation_space=ob_space, action_space=ac_space, demonstrations=demos, policy=train_policy,
                    custom_logger=bc_logger)

    # train bc
    print("Training policy on bc")
    bc_trainer.train(n_epochs=args.num_epochs)
    print("Saving policy")
    bc_trainer.save_policy(policy_path=os.path.join(bc_dir, "final_policy.pt"))

    if args.num_test_episodes is not None:
        print("Evaluating policy")
        bc_policy = bc_trainer.policy
        train_stats = train.eval_policy(bc_policy, train_env, n_episodes_eval=args.num_test_episodes)
        test_env = get_test_env(args)
        test_stats = train.eval_policy(bc_policy, test_env, n_episodes_eval=args.num_test_episodes)
        return train_stats, test_stats



if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model', type=str)
    train_parser.add_argument("--num_timesteps", type=int)
    train_parser.add_argument("--num_episodes", type=int)
    train_parser.add_argument("--rollout_file", type=str, required=True)
    train_parser.add_argument("--num_epochs", type=int, default=1)
    train_parser.add_argument("--num_test_episodes", type=int, default=None)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')
    print(main(train_parser.parse_args()))
