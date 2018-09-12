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
#module add cuda91/blas/9.1.85
#module add cuda91/fft/9.1.85
#module add cuda91/nsight/9.1.85
#module add cuda91/profiler/9.1.85
#module add cuda91/toolkit/9.1.85
#module add cuda90/blas/9.0.176
#module add cuda90/fft/9.0.176
#module add cuda90/nsight/9.0.176
#module add cuda90/profiler/9.0.176
#module add cuda90/toolkit/9.0.176
#module add cuda80/blas/8.0.61
#module add cuda80/fft/8.0.61
#module add cuda80/nsight/8.0.61
#module add cuda80/profiler/8.0.61
#module add cuda80/toolkit/8.0.61
wait
python $1 -n $2 -b $3 -e $4 -g $5 -c $6 -v
