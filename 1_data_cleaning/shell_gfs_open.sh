#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=CAPE_PR
#SBATCH --gres=gpu:0
#SBATCH --output=batch_out/CAPE_PR_%J_stdout.txt
#SBATCH --error=batch_out/CAPE_PR_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --chdir=/home/bmac87/BoltCast/1_data_cleaning/

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python 11_build_cape_pr.py
