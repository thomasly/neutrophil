#!/bin/bash

#SBATCH --job-name=tensorflow_test
#SBATCH --partition gpu4_dev

module load anaconda3/gpu
source activate tf
python3 test_tensorflow.py
