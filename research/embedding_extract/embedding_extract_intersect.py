import torch
import json
import pandas as pd
import csv
from transformers import AutoModel, AutoTokenizer
import random

FILENAME = "data/intersect.csv"
# OUTPUT_FILENAME = "context/safe2/safe2_embeddings.json"
OUTPUT_DIR = "context/intersection/llama_32_3b_ins/"
MODELNAME = "meta-llama/Llama-3.2-3B-Instruct"
COLUMNS_TO_USE = ["input_original", "input_satirical", "input_cultural", "input_research", "input_imaginary"]

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


# prompts = load_prompts_from_csv(FILENAME, sample_size=70, column_name="safe2")
# embeddings = get_last_token_embeddings(MODELNAME, prompts)

# Save to files
# save_embeddings_to_json(embeddings, filename=OUTPUT_FILENAME)
# save_embeddings_to_csv(embeddings, filename=OUTPUT_FILENAME)