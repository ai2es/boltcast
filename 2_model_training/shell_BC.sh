#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --job-name=BCData
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --output=batch_out/BCData_%j_stdout.txt
#SBATCH --error=batch_out/BCData_%j_stderr.txt
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0
#SBATCH --chdir=/home/bmac87/BoltCast/2_model_training/
#SBATCH --time=8:00:00
#SBATCH --exclude=
#SBATCH --dependency=
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python BC_data_loader.py

