#!/bin/bash
#SBATCH --job-name=rnn-sac-pendulum
#SBATCH --output=slurm-out/%x_%j.out
#SBATCH --error=slurm-out/%x_%j.err
#SBATCH --partition=kempner,kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p slurm-out outputs
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

python train_rnn_sac_pendulum.py
