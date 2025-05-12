#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint vram40
#SBATCH --time=01-00:00:00

python model_output.py \
    --seed 42 \
    --csv_path '' \
    --model_dir '' \
    --model_name '' \

