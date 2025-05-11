import torch
import json
import pandas as pd
import csv
from transformers import AutoModel, AutoTokenizer
import random

# model_list = ["google/gemma-2-9b-it", "meta-llama/Meta-Llama-2-7b-chat-hf", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B", "allenai/Llama-3.1-Tulu-3-8B", ""mistralai/Mistral-7B-Instruct-v0.2""]
FILENAME = "data/safe.csv"
MODELNAME = "google/gemma-2-9b-it"
OUTPUT_DIR = "context/safe/gemma_2_9b_it"
COLUMNS_TO_USE = ["safe1", "safe2"]

def get_last_token_embeddings(model_name, prompts):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    embeddings_list = []
    
    for prompt in prompts:
        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract hidden states (list of tensors, one per layer)
        hidden_states = outputs.hidden_states  # Shape: (num_layers, batch, seq_len, hidden_dim)
        
        # Get the last token's embedding from each layer
        last_token_index = inputs.input_ids.shape[1] - 1
        last_token_embeddings = [layer[:, last_token_index, :].squeeze(0).cpu().tolist() for layer in hidden_states]
        
        embeddings_list.append({"prompt": prompt, "embeddings": last_token_embeddings})
    
    return embeddings_list

def load_prompts_from_csv(filename, sample_size=None, column_name = "input"):
    prompts = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row[column_name])
    
    if sample_size:
        # Return a random sample of the prompts if a sample size is provided
        return random.sample(prompts, min(sample_size, len(prompts)))
    
    return prompts

def save_embeddings_to_json(embeddings, filename="embeddings.json"):
    with open(filename, "w") as f:
        json.dump(embeddings, f, indent=4)

def save_embeddings_to_csv(embeddings, filename="embeddings.csv"):
    data = []
    for i, entry in enumerate(embeddings):
        for layer_idx, layer_embedding in enumerate(entry["embeddings"]):
            data.append({"prompt": entry["prompt"], "layer": layer_idx, "embedding": layer_embedding})
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

for column in COLUMNS_TO_USE:
    print(f"Processing column: {column}")
    prompts = load_prompts_from_csv(FILENAME, column_name=column)
    embeddings = get_last_token_embeddings(MODELNAME, prompts)
    
    output_filename = OUTPUT_DIR + f"{column}_embeddings.json"
    save_embeddings_to_json(embeddings, filename=output_filename)
    print(f"Saved embeddings to {output_filename}")
