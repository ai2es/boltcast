#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=GFSmv
#SBATCH --output=batch_out/BC_gfsmv%J_stdout.txt
#SBATCH --error=batch_out/BC_gfsmv_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/bmac87/
#SBATCH --array=1-32%2

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python /scratch/bmac87/wget_gfs.py --init_time=00 --fcst_hour=$SLURM_ARRAY_TASK_ID
python /scratch/bmac87/wget_gfs.py --init_time=06 --fcst_hour=$SLURM_ARRAY_TASK_ID
python /scratch/bmac87/wget_gfs.py --init_time=12 --fcst_hour=$SLURM_ARRAY_TASK_ID
python /scratch/bmac87/wget_gfs.py --init_time=18 --fcst_hour=$SLURM_ARRAY_TASK_ID

