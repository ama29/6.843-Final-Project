from imitation.scripts.common.demonstrations import load_expert_trajs


def inspect_rollouts(rollouts_filepath: str):
    demos = load_expert_trajs(rollouts_filepath, n_expert_demos=None)
    print(demos)

if __name__ == "__main__":
    inspect_rollouts("trained_models/SAC_depth_1mbuffer/bc/rollouts.npz")