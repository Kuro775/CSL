#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint vram40
#SBATCH --time=12:00:00
#SBATCH --output=

python harm_rates.py \
    --csv_path '' \
    --instr_col '' \
    --out_col '' \
    --hf_token ''
