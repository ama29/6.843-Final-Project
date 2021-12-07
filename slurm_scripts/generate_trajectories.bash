#!/bin/bash

#SBATCH -o generate_trajectories.log-%j
#SBATCH -c 20

# load modules
source /etc/profile
module load anaconda/2020a
module load cuda/10.0

eval "$(conda shell.bash hook)"
conda activate tf_grasp_env_workaround

# run script
cd ~/up_6.843-Final-Project
python manipulation_main/training/run_bc.py --config config/sac_config.yaml --algo SAC --model_dir trained_models/SAC_depth_1mbuffer --timestep 100000 --model trained_models/SAC_depth_1mbuffer/best_model/best_model.zip --num_timesteps 500 --num_episodes 10000 --rollout_file trained_models/SAC_depth_1mbuffer/bc/rollouts_nocur.npz --num_test_episodes 10