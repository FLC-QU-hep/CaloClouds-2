#!/bin/bash
#SBATCH --time 5-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name CCgeneration
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=EMAIL
#SBATCH --output ../joblog/%j.out      # terminal output
#SBATCH --error ../joblog/%j.err
#SBATCH --constraint="GPUx1&A100"

bash
source ~/.bashrc

conda activate torch_113

cd HOME/6_PointCloudDiffusion/evaluation

python generate_for_metrics.py -cc cm

exit
