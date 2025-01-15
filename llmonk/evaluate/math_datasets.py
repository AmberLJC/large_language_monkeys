from pathlib import Path
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
import re
import yaml
from dataclasses import dataclass
from typing import Optional
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    get_unnormalized_answer,
    remove_boxed,
    is_equiv,
)

# Constants
ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
INVALID_ANS_GSM8k = "[invalid]"
GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]

@dataclass
class EvalConfig:
    samples_dir: str
    save_dir: str
    dset: str = "gsm8k"
    num_workers: Optional[int] = None
    offset: int = 0
    limit: Optional[int] = None
    stride: int = 1
    sample_path: Optional[Path] = None
    save_path: Optional[Path] = None

def load_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, data: dict):
    with open(path, 'w') as f:
        yaml.dump(data, f)

def filter_ignores(st: str, regexes_to_ignore: list) -> str:
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st

def extract_answer_gsm8k(completion: str) -> str:
    match = ANS_RE_GSM8k.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = filter_ignores(
            match_str,
            GSM8K_IGNORE_REGEXES,
        )
        return match_str
    return INVALID_ANS_GSM8k

def is_correct_gsm8k(model_completion: str, gt_example: str) -> bool:
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    model_answer = extract_answer_gsm8k(model_completion)
    return model_answer == gt_answer or is_equiv(model_answer, gt_answer)

def is_correct_minerva(og_pred: str, gt: str) -> bool:
    pred = normalize_final_answer(get_unnormalized_answer(og_pred))
    gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    return pred == gt or is_equiv(pred, gt)

def is_correct(sample: str, gt_answer: str, dset: str) -> bool:
    if dset == "gsm8k":
        return is_correct_gsm8k(sample, gt_answer)
    elif dset == "math":
        return is_correct_minerva(sample, gt_answer)
    else:
        raise ValueError(f"Dataset {dset} not supported")

def process_sample(config: EvalConfig):
    if config.save_path.exists():
        return

    result = load_yaml(config.sample_path)
    corrects = []

    for sample in result["samples"]:
        correct = is_correct(sample, result["gt_answer"], config.dset)
        corrects.append(correct)

    result["is_corrects"] = corrects
    result['coverage'] = sum(corrects) > 0
    save_yaml(config.save_path, result)

def get_tasks(config: EvalConfig) -> list:
    sample_paths = Path(config.samples_dir).glob("*.yaml")
    tasks = []
    
    for sample_path in tqdm(sample_paths, desc="Loading generations"):
        save_path = Path(config.save_dir) / sample_path.name
        task_config = deepcopy(config)
        task_config.sample_path = sample_path
        task_config.save_path = save_path
        tasks.append(task_config)

    return tasks

def main(config: EvalConfig):
    # Create save directory if it doesn't exist
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    tasks = get_tasks(config)
    tasks = sorted(tasks, key=lambda x: str(x.save_path))
    tasks = tasks[config.offset : config.limit : config.stride]

    print(f"Evaluating {len(tasks)} problems.")

    if config.num_workers not in [0, None]:
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            _ = list(tqdm(pool.map(process_sample, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_sample(task)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate math dataset samples')
    parser.add_argument('--samples_dir', help='Directory containing sample files', default='./logs/gsm8k_samples' )
    parser.add_argument('--save_dir', help='Directory to save evaluation results', default='./logs/gsm8k_eval' )
    parser.add_argument('--dset', default='gsm8k', choices=['gsm8k', 'math'], help='Dataset type')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--offset', type=int, default=0, help='Starting offset')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--stride', type=int, default=1, help='Stride for processing samples')
    
    args = parser.parse_args()
    
    config = EvalConfig(
        samples_dir=args.samples_dir,
        save_dir=args.save_dir,
        dset=args.dset,
        num_workers=args.num_workers,
        offset=args.offset,
        limit=args.limit,
        stride=args.stride
    )
    
    main(config)