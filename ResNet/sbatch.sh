#!/bin/bash

#SBATCH --job-name=tesorflow
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --partition=gpu4_medium
#SBATCH --error=/gpfs/home/liuy08/slurm_outputs/%x_%j.err
#SBATCH --output=/gpfs/home/liuy08/slurm_outputs/%x_%j.out
##SBATCH --dependency=afterany:job_id

module add anaconda/gpu/5.2.0
source activate tf 
python3 $1 