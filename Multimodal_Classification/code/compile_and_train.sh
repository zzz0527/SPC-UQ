#!/bin/bash --login

#SBATCH -p gpuA              # v100 GPUs         [up to  8 CPU cores per GPU permitted]
### Required flags
#SBATCH -G 1                 # (or --gpus=N) Number of GPUs 
#SBATCH -t 2-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)
### Optional flags
#SBATCH --ntasks-per-node=8         # (or --ntasks=) Number of CPU (host) cores (default is 1)

module load apps/binapps/anaconda3/2023.03  # Python 3.10.10
source activate luma_env


python compile_dataset.py -c cfg/default.yml
python compile_dataset.py -c cfg/noise_sample.yml
python compile_dataset.py -c cfg/noise_label.yml
python compile_dataset.py -c cfg/noise_diversity.yml

python run_baselines.py
python run_baselines.py --noise_type diversity
python run_baselines.py --noise_type label
python run_baselines.py --noise_type sample
