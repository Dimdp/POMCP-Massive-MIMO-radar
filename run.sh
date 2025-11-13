#!/bin/sh
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --nodelist=n4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs.txt

python ./run.py
