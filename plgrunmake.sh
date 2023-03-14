#! /bin/bash
#SBATCH -p plgrid-gpu-v100
#SBATCH -A plggpurateselection-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -c 2
#SBATCH -e /net/ascratch/people/plgkrusek/ftmrate_internal/log/%x.err
#SBATCH -o /net/ascratch/people/plgkrusek/ftmrate_internal/log/%x.out


cd $SCRATCH/ftmrate_internal

module load python/3.10.4-gcccore-11.3.0
module load cuda/11.3.1
module load cudnn/8.2.1.32-cuda-11.3.1

source ../venv310/bin/activate


#export XLA_FLAGS=--xla_gpu_cuda_data_dir=/net/software/local/cuda/11.2
#module load plgrid/libs/tensorflow-gpu/2.8.0-python-3.9
#module load plgrid/tools/python/3.9

make "$SLURM_JOB_NAME"