#!/bin/bash

#SBATCH -o run_dagger_expert_on_fail.log-%j
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1


# load modules
source /etc/profile
module load anaconda/2020a
module load cuda/10.0

eval "$(conda shell.bash hook)"
conda activate tf_grasp_env_workaround

# run script
cd ~/up_6.843-Final-Project
python manipulation_main/training/run_dagger.py --config config/sac_config.yaml --algo SAC --model_dir trained_models/SAC_depth_1mbuffer --timestep 100000 --model trained_models/SAC_depth_1mbuffer/best_model/best_model.zip --test_rollouts 20 --round_episodes 50 --num_timesteps 10000 --log_dir dagger_expert_on_fail --expert_on_fail