#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=ENTLN_Coll
#SBATCH --output=batch_out/ENTLN_%J_stdout.txt
#SBATCH --error=batch_out/ENTLN_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/BoltCast/0_data_acquisition/
#SBATCH --time=48:00:00

conda init
conda activate wget

python wget_entln.py 
