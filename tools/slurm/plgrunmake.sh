#! /bin/bash
#SBATCH -p plgrid-gpu-v100
#SBATCH -A plggpurateselection-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -c 2
#SBATCH -e /net/ascratch/people/plgkrusek/ml4wifi/log/%x.err
#SBATCH -o /net/ascratch/people/plgkrusek/ml4wifi/log/%x.out


cd $SCRATCH/ml4wifi

#module add tensorflow/2.8.0-fosscuda-2020b

make "$SLURM_JOB_NAME"