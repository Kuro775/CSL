import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
FILENAME = "cultural.csv"
OUTPUT_FILENAME = "context/safe2/safe2_embeddings.json"

df = pd.read_csv(FILENAME)
prompts = df["prompt"].tolist()
contexts = df["context"].tolist()
unique_contexts = sorted(set(contexts))

# Choose which model to analyze: LLaMA-2 or Mistral
model_name = "meta-llama/Llama-2-7b-hf"   # e.g., LLaMA-2 7B Hugging Face model
model_label = "LLaMA-2 7B"               # label for plots/prints
# model_name = "mistralai/Mistral-7B-v0.1"
# model_label = "Mistral 7B"

# Load tokenizer and model (with output_hidden_states enabled)
print(f"Loading model {model_name} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device.startswith("cuda") else torch.float32  # use FP16 on GPU for efficiency
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, output_hidden_states=True
).to(device)
model.eval()

# Determine layer indices for early, mid, late
num_layers = model.config.num_hidden_layers  # total transformer layers
early_layer = 1  # after first transformer block
mid_layer = num_layers // 2  # middle layer (round down)
final_layer = num_layers     # last layer index (transformer blocks are 1-indexed in this scheme)
selected_layers = [early_layer, mid_layer, final_layer]
print(f"Model has {num_layers} layers; analyzing layers: {selected_layers}")

# Prepare storage for hidden states
layer_hidden_vectors = {layer: [] for layer in selected_layers}

# Batch processing to handle many prompts without out-of-memory
batch_size = 8
for start in range(0, len(prompts), batch_size):
    batch_prompts = prompts[start : start + batch_size]
    # Tokenize batch (pad to max length in batch)
    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, layer2, ..., layerN)
    # For each sequence in the batch, find the index of its last non-pad token
    lengths = attention_mask.sum(dim=1)  # tensor of shape (batch,) with lengths
    # Extract the last-token hidden state for each selected layer
    for layer in selected_layers:
        # hidden_states[layer] has shape (batch, seq_len, hidden_dim)
        layer_state = hidden_states[layer]  # get the hidden state tensor for this layer
        # Gather last token state for each sequence in the batch
        # We'll use advanced indexing to select [batch_index, length-1] for each batch item
        batch_indices = torch.arange(layer_state.size(0), device=device)
        last_token_indices = lengths - 1  # (batch,) indices of last token
        # Select hidden state at the last token position for each sequence
        last_hidden = layer_state[batch_indices, last_token_indices, :]  # shape: (batch, hidden_dim)
        # Move to CPU and convert to numpy for PCA
        layer_hidden_vectors[layer].append(last_hidden.cpu().numpy())

# Concatenate all batches for each layer
for layer in selected_layers:
    layer_hidden_vectors[layer] = np.concatenate(layer_hidden_vectors[layer], axis=0)
    print(f"Layer {layer} hidden states collected: shape = {layer_hidden_vectors[layer].shape}")

# Now perform PCA for each selected layer and plot the results
plt.figure(figsize=(6 * len(selected_layers), 5))
for idx, layer in enumerate(selected_layers, start=1):
    X = layer_hidden_vectors[layer]
    # PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    # Create subplot for this layer
    ax = plt.subplot(1, len(selected_layers), idx)
    for ctx in unique_contexts:
        # boolean mask for points belonging to this context
        mask = np.array(contexts) == ctx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=ctx, alpha=0.7)
    ax.set_title(f"Layer {layer} PCA - {model_label}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
plt.tight_layout()
plt.show()
