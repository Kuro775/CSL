import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modeling Output"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file with 'input' column.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the pretrained model directory.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name used for naming the output column.",
    )

    args = parser.parse_args()
    return args


def load_model(model_dir, device):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if device == "cuda" else "auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def generate_output(model, tokenizer, prompt, device, max_length=2000):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_length=max_length)
    generated_tokens = output[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def process_dataframe(df, model, tokenizer, device, model_name):
    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        prompt = row["input"]
        response = generate_output(model, tokenizer, prompt, device)
        outputs.append(response)

    output_column = f"{model_name} output"
    df[output_column] = outputs
    return df

def main():
    args = parse_args()
    csv_path = args.csv_path
    model_dir = args.model_dir
    model_name = args.model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    model, tokenizer = load_model(model_dir, device)
    df = process_dataframe(df, model, tokenizer, device, model_name)

    df.to_csv(csv_path, index=False)
    print(f"Overwritten original CSV at: {csv_path}")

if __name__ == "__main__":
    main()
    
