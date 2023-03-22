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

mkdir -p $SCRATCH/ftmrate_internal/log

cd $PLG_GROUPS_STORAGE/plggml4wifi/ftmrate_internal

module load python/3.10.4-gcccore-11.3.0
module load cuda/11.3.1
module load cudnn/8.2.1.32-cuda-11.3.1

source $PLG_GROUPS_STORAGE/plggml4wifi/venv/bin/activate
echo "$SLURM_JOB_NAME"
make "$SLURM_JOB_NAME"