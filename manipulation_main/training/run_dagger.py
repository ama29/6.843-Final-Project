import argparse
import os.path

from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util import logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from manipulation_main.training.custom_obs_policy import TransposeNatureCNN, TransposedVisTransformer
from manipulation_main.training.imitation_utils import BASE_DIR, get_env_expert


def main(args):
    os.chdir(BASE_DIR)  # all filepaths are relative to repo root in other files
    dagger_dir = os.path.join(BASE_DIR, args.model_dir, "dagger")

    env, expert = get_env_expert(args)

    # define policy to be learned
    ob_space = env.observation_space
    ac_space = env.action_space
    # Default network arch is nature, which is also used in this repo. Might be good to verify later arch matches
    # TODO: is lr_schedule used? For now using constant lr scheduler
    feat_cls = TransposedVisTransformer if args.use_transformer else TransposeNatureCNN
    train_policy = ActorCriticCnnPolicy(observation_space=ob_space, action_space=ac_space, lr_schedule=lambda x: 0.005,
                                        features_extractor_class=feat_cls)
    log_dir = os.path.join(dagger_dir, "logs")
    bc_trainer = BC(observation_space=ob_space, action_space=ac_space, demonstrations=None, policy=train_policy)

    # construct dagger instance and train
    dagger_logger = logger.configure(log_dir)
    save_dir = os.path.join(dagger_dir, "dagger_model")
    bc_train_args = {"log_rollouts_n_episodes": args.test_rollouts, "batch_size": 128}
    trainer = SimpleDAggerTrainer(venv=env, scratch_dir=save_dir, expert_policy=expert, bc_trainer=bc_trainer,
                                  custom_logger=dagger_logger, bc_train_args=bc_train_args)
    trainer.train(total_timesteps=args.num_timesteps, rollout_round_min_timesteps=100)

    trainer.save_policy(os.path.join(dagger_dir, "final_policy.pt"))


if __name__ == "__main__":
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model', type=str)
    train_parser.add_argument("--num_timesteps", type=int)
    train_parser.add_argument("--test_rollouts", type=int, default=20)
    train_parser.add_argument("--round_episodes", type=int, default=20)
    train_parser.add_argument("--use_transformer", action="store_true")

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')
    main(train_parser.parse_args())
