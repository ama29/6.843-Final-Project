#!/bin/bash

#SBATCH -o run_bc_cnn_precise.log-%j
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
python manipulation_main/training/run_bc.py --run_range --config config/sac_config.yaml --algo SAC --model_dir trained_models/SAC_depth_1mbuffer --timestep 100000 --model trained_models/SAC_depth_1mbuffer/best_model/best_model.zip --num_timesteps 500 --num_episodes 10000 --rollout_file trained_models/SAC_depth_1mbuffer/bc/rollouts_nocur.npz --num_test_episodes 20 --num_epochs 4 --range_min 10 --range_max 1000 --range_step 50 --log_dir cnn_precise