#!/bin/bash

#SBATCH --job-name=tensorflow
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu4_short
#SBATCH --error=/gpfs/home/liuy08/slurm_outputs/%x_%j.err 
#SBATCH --output=/gpfs/home/liuy08/slurm_outputs/%x_%j.out
##SBATCH --dependency=afterany:job_id

module purge
module add python/gpu/3.6.5
wait
python $1 -n $2 -b $3 -e $4 -g $5 -c $6 -v
