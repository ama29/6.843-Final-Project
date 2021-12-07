import argparse

from manipulation_main.training.imitation_utils import get_env_expert

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
train_parser.add_argument("--log_dir", type=str, default=None)
train_parser.add_argument("--max_expert_demos", type=int, default=None)
train_parser.add_argument("--use_transformer", action="store_true")
train_parser.add_argument("--run_range", action="store_true")
train_parser.add_argument("--range_min", type=int, default=None)
train_parser.add_argument("--range_max", type=int, default=None)
train_parser.add_argument("--range_step", type=int, default=None)

train_parser.add_argument('--timestep', type=str)
train_parser.add_argument('-s', '--simple', action='store_true')
train_parser.add_argument('-sh', '--shaped', action='store_true')
train_parser.add_argument('-v', '--visualize', action='store_true')
train_parser.add_argument('-tf', '--timefeature', action='store_true')
args = train_parser.parse_args()

train_env, expert = get_env_expert(args)
robot_env = train_env.venv.venv.envs[0].env.env

# reset should change stae
train_env.reset()
train_env.reset()
train_env.reset()

# set rng to get same setup
rng_state = robot_env._rng.get_state()
train_env.reset()
robot_env._rng.set_state(rng_state)
train_env.reset()
robot_env._rng.set_state(rng_state)
train_env.reset()
