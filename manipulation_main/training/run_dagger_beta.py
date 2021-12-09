import argparse
import copy
import os.path

import pandas as pd
from imitation.data import rollout
from imitation.scripts.common import train
from imitation.scripts.common.demonstrations import load_expert_trajs

from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer, LinearBetaSchedule
from imitation.util import logger
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from manipulation_main.training.custom_obs_policy import TransposeNatureCNN, TransposedVisTransformer
from manipulation_main.training.imitation_utils import BASE_DIR, get_env_expert, get_test_env

def run_dagger_range(args, min_beta: int, max_beta: int, step: int):
    all_results = []
    for beta_val in range(min_beta, max_beta, step):  # want to use up to max_traj inclusive
        sub_args = copy.deepcopy(args)
        sub_args.max_expert_demos = beta_val
        sub_args.log_dir += str(beta_val)
        train_res, test_res = run_one_dagger(sub_args, beta=LinearBetaSchedule(beta_val))
        train_appended = {"train_" + k: v for k, v in train_res.items()}
        test_appended = {"test_" + k: v for k, v in test_res.items()}
        train_appended.update(test_appended)
        train_appended["num_expert_demos"] = beta_val
        all_results.append(train_appended)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(BASE_DIR, args.model_dir, "dagger", f"{args.log_dir}_all_logs.csv"))


def run_one_dagger(args, beta):
    os.chdir(BASE_DIR)  # all filepaths are relative to repo root in other files
    dagger_dir = os.path.join(BASE_DIR, args.model_dir, "dagger", args.log_dir)
    
    env, expert = get_env_expert(args)
    # define policy to be learned
    ob_space = env.observation_space
    ac_space = env.action_space
    # Default network arch is nature, which is also used in this repo. Might be good to verify later arch matches                                                                                                                                # TODO: is lr_schedule used? For now using constant lr scheduler
    feat_cls = TransposedVisTransformer if args.use_transformer else TransposeNatureCNN
    train_policy = ActorCriticCnnPolicy(observation_space=ob_space, action_space=ac_space, lr_schedule=lambda x: 0.005,                                         features_extractor_class=feat_cls)
                                                                                                                    
    log_dir = os.path.join(dagger_dir, f"logs") 
    bc_trainer = BC(observation_space=ob_space, action_space=ac_space, demonstrations=None, policy=train_policy, batch_size=128)
    # construct dagger instance and train
    dagger_logger = logger.configure(log_dir)
    save_dir = os.path.join(dagger_dir, "dagger_model")
    bc_train_kwargs = {"log_rollouts_n_episodes": args.test_rollouts, "n_epochs": 4}
    trainer = SimpleDAggerTrainer(venv=env, scratch_dir=save_dir, beta_schedule=beta, expert_policy=expert, bc_trainer=bc_trainer, custom_logger=dagger_logger)
    trainer.train(total_timesteps=args.num_timesteps, rollout_round_min_timesteps=args.round_episodes, bc_train_kwargs=bc_train_kwargs)
    trainer.save_policy(os.path.join(dagger_dir, "final_policy.pt"))
    
    if args.num_test_episodes is not None:
        print("Evaluating policy")
        dagger_policy = bc_trainer.policy
        train_stats = train.eval_policy(dagger_policy, env, n_episodes_eval=args.num_test_episodes)
        test_env = get_test_env(args)
        test_stats = train.eval_policy(dagger_policy, test_env, n_episodes_eval=args.num_test_episodes)         
        return train_stats, test_stats

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
    train_parser.add_argument("--num_episodes", type=int)
    train_parser.add_argument("--expert_on_fail", action="store_true")     
    train_parser.add_argument("--num_test_episodes", type=int, default=None)  
    train_parser.add_argument("--log_dir", type=str, default="")
    train_parser.add_argument("--max_expert_demos", type=int, default=None)
    train_parser.add_argument("--run_range", action="store_true")
    train_parser.add_argument("--range_min", type=int, default=None)
    train_parser.add_argument("--range_max", type=int, default=None)
    train_parser.add_argument("--range_step", type=int, default=None)
    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')
    args = (train_parser.parse_args())     
    
    if args.run_range:
        run_dagger_range(args, min_beta=args.range_min, max_beta=args.range_max, step=args.range_step)
    else:
        print("Run single DAgger with beta = 0.1")
        print(run_one_dagger(args, LinearBetaSchedule(0.1)))
