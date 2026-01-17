#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=BCMovie
#SBATCH --output=batch_out/BCMovie_%j_stdout.txt
#SBATCH --error=batch_out/BCMovie_%j_stderr.txt
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/bmac87/BoltCast/3_model_analysis/
#SBATCH --array=0-4
#SBATCH --dependency=
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python 0a_BC_output_maps_for_movie.py --rotation=$SLURM_ARRAY_TASK_ID
