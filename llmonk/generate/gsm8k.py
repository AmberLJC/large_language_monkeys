import torch
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing
import random
import requests
from functools import partial
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from openai import AzureOpenAI
import argparse
import os

@dataclass
class InferenceConfig:
    save_dir: str = './logs/gsm8k_samples'
    vllm_port: int = 8000
    num_samples: int = 1
    batch_size: int = 1
    model_name: str = 'gpt-4o'
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop_strings: List[str] = None
    num_few_shot: int = 5
    seed: int = 42
    limit: Optional[int] = None
    stride: Optional[int] = None
    offset: Optional[int] = None
    num_workers: Optional[int] = None

def save_yaml(path: Path, data: dict):
    with open(path, 'w') as f:
        yaml.dump(data, f)

    print(f"Saving data to {path}")
    # print(f"Data: {data}")
def get_few_shot_prompt(item):
    few_shot_items = item["few_shot_items"]
    few_shot_pieces = []
    
    for f in few_shot_items:
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)

    return "".join(few_shot_pieces)

def run_api_inference(item, config: InferenceConfig):
    save_dir = Path(config.save_dir)
    outpath = save_dir / f"{item['id']}.yaml"


    few_shot_prompt = get_few_shot_prompt(item)
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:" 
    
    client = AzureOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint='https://api.umgpt.umich.edu/azure-openai-api',
        # api_version='2023-03-15-preview',
        api_version="2024-02-01",
        organization=os.getenv('OPENAI_ORGANIZATION'),
    ) 
 

    print(f'Start {config.num_samples // config.batch_size} generation for item {item["id"]}.') 
    samples = []
    for _ in tqdm(range(config.num_samples // config.batch_size), desc=f"Item {item['id']}"):
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                messages = [
                    {"role": "user","content": prompt},
                ],
                max_tokens=config.max_tokens,
                n=config.batch_size,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_strings,
            )
            # print('====================')
            # print(prompt)

            batch_samples = [choice.message.content for choice in response.choices]
            samples.extend(batch_samples)
            
            # print(f'Response: {batch_samples}') 
            # print('====================')
            
        except Exception as e:
            print(f"Error generating completion for item {item['id']}: {str(e)}")
            continue

    out = {
        "prompt": prompt,
        "question": item["question"],
        "samples": samples,
        "gt_answer": item["answer"],
    }

    save_yaml(outpath, out)

def main(config: InferenceConfig = InferenceConfig()):
    test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)

    for i, data in enumerate(train_dataset):
        data["id"] = i

    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        data["id"] = i
        data["few_shot_items"] = few_shot_items

    random.shuffle(test_dataset)

    limit = config.limit if config.limit is not None else len(test_dataset)
    stride = config.stride if config.stride is not None else 1
    offset = config.offset if config.offset is not None else 0

    test_dataset = test_dataset[offset:limit:stride]
    print(f"Total number of items to process: {len(test_dataset)}")

    # Create save directory if it doesn't exist
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    go_func = partial(run_api_inference, config=config)
    
    if config.num_workers not in [0, None]:
        multiprocessing.set_start_method('spawn', force=True)
        
        with multiprocessing.Pool(config.num_workers) as pool:
            predictions = list(
                tqdm(
                    pool.imap_unordered(go_func, test_dataset),
                    total=len(test_dataset),
                    desc="Processing dataset"
                )
            )
    else:
        # Single process execution
        predictions = []
        for item in tqdm(test_dataset, desc="Processing dataset"):
            predictions.append(go_func(item))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Configure inference parameters.")

    # Add arguments
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--num_few_shot", type=int, default=2, help="Number of few-shot examples.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing.")
    parser.add_argument("--limit", type=int, default=10, help="Limit on the number of inputs to process.")
    parser.add_argument("--save-dir", type=str, default="./logs/gsm8k_samples", help="Directory to save generated samples.")
                        
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name for inference.")

    args = parser.parse_args()

    # Create InferenceConfig using parsed arguments
    custom_config = InferenceConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        num_few_shot=args.num_few_shot,
        num_workers=args.num_workers,
        limit=args.limit,
        model_name=args.model_name,
        save_dir=args.save_dir,
    )

    main(custom_config)