#!/bin/bash
#
#SBATCH --partition=ai2es_a100
#SBATCH --exclude=c314,c315,c731,c732
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=BCSnEval_Edits
#SBATCH --output=batch_out/BCSnEval_Edits_%j_stdout.txt
#SBATCH --error=batch_out/BCSnEval_Edits_%j_stderr.txt
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00
#SBATCH --chdir=/home/bmac87/BoltCast/3_model_analysis/
#SBATCH --array=0
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python 9a_BC_seasonal_eval.py