#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=GLMs3fs
#SBATCH --output=batch_out/GLM_s3fs_%J_stdout.txt
#SBATCH --error=batch_out/GLM_s3fs_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/BoltCast/0_data_acquisition/
#SBATCH --array=6,7
#SBATCH --time=96:00:00

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python s3fs_glm.py --year=$SLURM_ARRAY_TASK_ID --sat=G18 --download
python s3fs_glm.py --year=$SLURM_ARRAY_TASK_ID --sat=G16 --download