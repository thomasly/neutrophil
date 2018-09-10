#!/bin/bash
#$ -S /bin/bash
#$ -cwd

module load anaconda3/gpu
source activate tf

python train_resnet.py true 32 1 true
