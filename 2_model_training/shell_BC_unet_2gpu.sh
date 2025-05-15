#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --exclude=c830,c829,c315,c314
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=batch_out/%j/BC_LSTM_%j_stdout.txt
#SBATCH --error=batch_out/%j/BC_LSTM_%j_stderr.txt
#SBATCH --job-name=BCLSTMv2
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=3,4
#SBATCH --chdir=/home/bmac87/BoltCast/2_model_training/
#SBATCH --time=06:00:00
#################################################

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python BC_train.py --model2train=LSTM --exp=$SLURM_ARRAY_TASK_ID @txt_exp.txt @txt_proj.txt @txt_unet.txt @txt_lstm.txt --label=_no_drop_no_shuffle
