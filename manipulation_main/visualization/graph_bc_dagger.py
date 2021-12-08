import os.path

import pandas as pd
from matplotlib import pyplot as plt

from manipulation_main.training.imitation_utils import BASE_DIR

EPISODE_MEAN_LEN = 14.5426


def graph_bc_dagger(bc_log_path: str, dagger_log_path: str):
    bc_df = pd.read_csv(bc_log_path)
    dagger_df = pd.read_csv(dagger_log_path)
    combined_df = pd.DataFrame()
    for index, row in bc_df.iterrows():
        combined_df.at[row["num_expert_demos"] * EPISODE_MEAN_LEN, "bc_success"] = row["train_success_mean"]

    for index, row in dagger_df.iterrows():
        if pd.isna(row["dagger/total_episode_count"]):
            continue
        combined_df.at[row["dagger/total_episode_count"], "dagger_success"] = row["rollout/success_mean"]

    plt.plot(combined_df)
    plt.xlabel("Number of expert demonstrations")
    plt.ylabel("Mean episode success")
    plt.title("Comparison of Daggger vs BC Expert Demo Utilization")
    plt.legend(combined_df.columns)
    plt.show()


def graph_bc_transformer(cnn_log_path: str, trans_log_path: str):
    cnn_df = pd.read_csv(cnn_log_path)
    trans_df = pd.read_csv(trans_log_path)
    combined_df = pd.DataFrame()
    for index, row in cnn_df.iterrows():
        combined_df.at[row["num_expert_demos"] * EPISODE_MEAN_LEN, "cnn_success"] = row["train_success_mean"]
    for index, row in trans_df.iterrows():
        combined_df.at[row["num_expert_demos"] * EPISODE_MEAN_LEN, "trans_success"] = row["train_success_mean"]

    plt.plot(combined_df)
    plt.xlabel("Number of expert demonstrations")
    plt.ylabel("Mean episode success")
    plt.title("Comparison of CNN vs Transformer Performance on Behavior Cloning Task")
    plt.legend(combined_df.columns)
    plt.show()


def graph_env_interactions(bc_log_path: str, dagger_log_path: str, rl_log_path: str):
    # TODO: have bc count number of timesteps, not just expert trajectories
    pass


if __name__ == "__main__":
    cnn_bc_log_path = os.path.join(BASE_DIR, "trained_models", "SAC_depth_1mbuffer", "bc", "cnn_precise_all_logs.csv")
    tran_bc_log_path = os.path.join(BASE_DIR, "trained_models", "SAC_depth_1mbuffer", "bc", "tran_precise_all_logs.csv")
    dagger_log_path = os.path.join(BASE_DIR, "trained_models", "SAC_depth_1mbuffer", "dagger", "logs", "progress.csv")
    graph_bc_transformer(cnn_bc_log_path, tran_bc_log_path)
    # graph_bc_dagger(bc_log_path, dagger_log_path)
