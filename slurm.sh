#!/bin/bash
#SBATCH --partition=learn
#SBATCH --time=1-00:00:00
#SBATCH --job-name=IMAGENET_training
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=200G
#SBATCH --gpus-per-task=1
#SBATCH --output=/home/%u/rsc/rsc-examples/imagenet/slurm-%j.out
#SBATCH --error=/home/%u/rsc/rsc-examples/imagenet/slurm-%j.err
#SBATCH --qos=urgent_deadline

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

source /uca/conda-envs/activate-latest
cd /home/$USER/rsc/rsc-examples/imagenet/ || exit 1
srun python train.py
# srun python train.py --random-access
# srun python train.py --dummy
