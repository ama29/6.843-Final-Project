import os.path

import pandas as pd
from manipulation_main.training.imitation_utils import BASE_DIR
from matplotlib import pyplot as plt


def graph_bc_dagger(bc_log_path: str, dagger_log_path: str):
    bc_df = pd.read_csv(bc_log_path)
    dagger_df = pd.read_csv(dagger_log_path)
    combined_df = pd.DataFrame()
    for index, row in bc_df.iterrows():
        combined_df.at[row["num_expert_demos"], "bc_success"] = row["train_success_mean"]

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

if __name__ == "__main__":
    bc_log_path = os.path.join(BASE_DIR, "trained_models", "SAC_depth_1mbuffer", "bc", "all_logs.csv")
    dagger_log_path = os.path.join(BASE_DIR, "trained_models", "SAC_depth_1mbuffer", "dagger", "logs", "progress.csv")
    graph_bc_dagger(bc_log_path, dagger_log_path)