from src.utils.chat_models import get_chat_model
from src.utils.main_utils import write_standard_data, load_standard_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
import pandas as pd
import argparse
import numpy as np


def calculate_perplexity(model, tokenizer, text, device="cuda"):
    # Tokenize input text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    # Calculate perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    return torch.exp(neg_log_likelihood).item()


def load_data_to_evaluate(data_path):
    data = pd.read_csv(data_path)
    data_to_evaluate = []
    for index, row in data.iterrows():
        user_query = row["User Request"]
        response = row["Response"]
        data_to_evaluate.append(
            {"user_query": user_query, "response": response})
    return data_to_evaluate


def process_model(model_name, prompts, responses, n_devices):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    # Load model with device map for multiple GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # max_memory={i: "40GiB" for i in range(n_devices)}
    )
    model.eval()

    all_outputs = []

    # Calculate perplexity for each prompt-response pair
    for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
        # Format as chat using tokenizer's chat template
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False)

        # Calculate perplexity for the full conversation
        conversation_ppl = calculate_perplexity(
            model, tokenizer, formatted_text, device="cuda")

        output = {
            "user_query": prompt,
            "response": response,
            "conversation_perplexity": conversation_ppl,
            "formatted_conversation": formatted_text,
            "model": model_name
        }
        all_outputs.append(output)

    print(all_outputs[0]["formatted_conversation"])

    # Save results
    save_model_name = model_name.replace("/", "_")
    save_path = f"/results/{save_model_name}_perplexity_scores.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_standard_data(all_outputs, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_devices", type=int, default=2)
    args = parser.parse_args()

    data_path = "/data/adhoc_save/030325_final_human_annotation/human_eval_inputs/abs_data_15_inputs.csv"
    data_to_evaluate = load_data_to_evaluate(data_path)

    prompts = [item["user_query"] for item in data_to_evaluate]
    responses = [item["response"] for item in data_to_evaluate]

    process_model(args.model_name, prompts, responses, args.n_devices)
