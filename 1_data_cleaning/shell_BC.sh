#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=BCEdits
#SBATCH --gres=gpu:0
#SBATCH --output=batch_out/BCEdits_%J_stdout.txt
#SBATCH --error=batch_out/BCEdits_%J_stderr.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --chdir=/home/bmac87/BoltCast/1_data_cleaning/

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python 10_build_folds_v2.py
