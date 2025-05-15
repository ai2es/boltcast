#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=BC_pd_std
#SBATCH --output=batch_out/BC_pd_%j_stdout.txt
#SBATCH --error=batch_out/BC_pd_%j_stderr.txt
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0
#SBATCH --time=2:00:00
#SBATCH --chdir=/home/bmac87/BoltCast/3_model_analysis/
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python /home/bmac87/BoltCast/3_model_analysis/2_BC_perf_diagram.py

