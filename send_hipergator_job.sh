#!/bin/bash
#SBATCH --job-name=FaceAnonEval
#SBATCH --output=FaceAnonEval.out
#SBATCH --error=FaceAnonEval.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ethanwilson@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

nvidia-smi

module load conda
conda activate FaceAnonEval

while [[ "$1x" != "x" ]]; do 
    STRING="$STRING$1 " 
    shift 
done 

$STRING