"""# Imports"""

import pandas as pd
import os
import json
import csv
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast
import matplotlib.pyplot as plt
from pprint import pprint


"""# Configs"""

paths = {      # Put the file path here
    "original": "",
    "satirical": "",
    "cultural": "",
    "research": "",
    "imaginary": "",
    "translated1": "",
    "translated2": ""
}

model_cols = [
    'Mistral7B output flags', 'Llama2_7B output flags', 'Llama32_3B output flags',
    'Llama32_3BInstr output flags', 'Tulu3_8B output flags', 'Gemma2_IT output flags'
]

model_name_map = {
    'Mistral7B output flags': 'Mistral_7BInstruct-v0.2',
    'Llama2_7B output flags': 'Llama2_7B-chat',
    'Llama32_3B output flags': 'Llama3.2_3B',
    'Llama32_3BInstr output flags': 'Llama3.2_3BInstr',
    'Tulu3_8B output flags': 'Tulu3_8B',
    'Gemma2_IT output flags': 'Gemma2_9B'
}

"""# Functions"""

def load_dataset(path, model_cols):
    df = pd.read_csv(path)
    for col in model_cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def get_stats(df, model_col, harm_col, harm_val):
    filtered = df[df[harm_col] == harm_val]
    harm_true = filtered[model_col].apply(lambda d: d.get('harm') if isinstance(d, dict) else None).sum()
    harm_false = len(filtered) - harm_true
    refusal_true = filtered[model_col].apply(lambda d: d.get('refusal') if isinstance(d, dict) else None).sum()
    return {'total': len(filtered), 'harm=True': harm_true, 'harm=False': harm_false, 'refusal=True': refusal_true}

def calculate_normalized_harm(df, model_cols, harm_col):
    df_harm = df[df[harm_col] == True]
    total = len(df_harm)
    rates = {}
    for model in model_cols:
        harm_true = df_harm[model].apply(lambda d: d.get('harm') if isinstance(d, dict) else False).sum()
        rates[model] = harm_true / total if total > 0 else 0.0
    return rates

def plot_rates(df, title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.12
    index = range(len(df))
    x_labels = list(df.index)
    ordered_models = list(model_name_map.keys())

    for i, model_col in enumerate(ordered_models):
        display_name = model_name_map[model_col]
        values = df[model_col].values
        ax.bar([pos + i * bar_width for pos in index], values, width=bar_width, label=display_name)

    ax.set_xticks([pos + bar_width * (len(ordered_models) / 2 - 0.5) for pos in index])
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def find_intersection_prompts(dfs, harm_col_name='input harm'):  # Find the prompts that are harmful across all contexts
    prompt_sets = {name: set(df['prompt']) for name, df in dfs.items()}
    return set.intersection(*prompt_sets.values())

def build_intersection_df(harmful_dfs, intersection_prompts):
    original_df = harmful_dfs['original']
    merged_df = original_df[original_df['prompt'].isin(intersection_prompts)][['prompt', 'category']].copy()

    for name, df in harmful_dfs.items():
        if name == 'original':
            # For original, use 'prompt' → 'input_original'
            join_df = df[['prompt']].copy()
            join_df = join_df.rename(columns={'prompt': 'input_original'})
            join_df['prompt'] = join_df['input_original']  # restore prompt for merge
        elif name in ['translated1', 'translated2']:
            # For translated1 and translated2, use 'prompt' → 'input_translated1/2'
            join_df = df[['prompt']].copy()
            join_df = join_df.rename(columns={'prompt': f'input_{name}'})
            join_df['prompt'] = join_df[f'input_{name}']  # restore prompt for merge
        else:
            # For others, use 'input' → 'input_{name}'
            join_df = df[['prompt', 'input']].copy()
            join_df = join_df.rename(columns={'input': f'input_{name}'})

        merged_df = merged_df.merge(join_df, on='prompt', how='left')

    return merged_df

"""# Harmful rating"""



def main():
    # Load datasets
    dfs = {name: load_dataset(path, model_cols) for name, path in paths.items()}
    
    # --- Basic Stats ---
    results = {}
    for name, df in dfs.items():
        harm_col = 'input harm'
        stats = {}
        for model in model_cols:
            stats[model] = {
                'input_harm=True': get_stats(df, model, harm_col, True),
                'input_harm=False': get_stats(df, model, harm_col, False)
            }
        results[name] = stats
    
    pprint(results)
    # --- Normalized harmfulness rate ---
    normalized_rates = {}
    for name, df in dfs.items():
        harm_col = 'input harm'
        normalized_rates[name] = calculate_normalized_harm(df, model_cols, harm_col)
    
    normalized_df = pd.DataFrame(normalized_rates).T
    normalized_df.index.name = "Dataset"
    print(normalized_df.round(3))
    
    plot_rates(normalized_df, "Normalized Harmfulness Rate Across Contexts and Models", "Harmfulness Rate (input harm=True)")
    
    # --- Harmfulness on shared prompts ---
    harmful_dfs = {name: df[df['input harm'] == True] for name, df in dfs.items()}
    intersection_prompts = find_intersection_prompts(harmful_dfs)
    
    print(f"Number of shared harmful prompts: {len(intersection_prompts)}")
    
    intersected_rates = {}
    for name, df in harmful_dfs.items():
        filtered_df = df[df['prompt'].isin(intersection_prompts)]
        intersected_rates[name] = calculate_normalized_harm(filtered_df, model_cols, harm_col='input harm')
    
    intersected_df = pd.DataFrame(intersected_rates).T
    intersected_df.index.name = "Dataset"
    print(intersected_df.round(3))
    
    plot_rates(intersected_df, "Harmfulness Rate on Shared Harmful Prompts", "Harmfulness Rate (Intersection Only)")
    
    # --- Build intersection table ---
    intersection_df = build_intersection_df(harmful_dfs, intersection_prompts)
    print(intersection_df.head())
    
    # intersection_df.to_csv('')

if __name__ == "__main__":
    main()
