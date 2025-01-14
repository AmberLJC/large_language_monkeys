from pathlib import Path
import yaml
import re
from pydantic import BaseModel as Config
from typing import ClassVar
REQUIRED = ...
import signal

def load_yaml(path: Path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    return data


def save_yaml(path: Path, data, sort_keys=True):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def dataclass_to_dict(obj) -> dict:
    """
    Converts a dataclass to a dictionary. Will recurse through
    lists, dicts, and nested dataclasses.
    """

    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def get_theorem_name(id):
    """
    Map the huggingface datasets id to the theorem name in the Lean repository
    """
    if "math" in id or "algebra" in id or "numbertheory" in id or "induction" in id:
        return id
    elif "imo_1968_5_1" in id:
        return "imo_1968_p5_1"
    elif "imo_2007_6" in id:
        return "imosl_2007_algebra_p6"
    elif "aime" in id:
        return "aime_" + id.split("aime")[1].split("_")[0] + "_p" + id.split("_")[-1]
    else:
        return "_".join(id.split("_")[:-1]) + "_p" + id.split("_")[-1]


def is_valid_python(snippet):
    try:
        compile(snippet, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_first_code(output_string: str):
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # sometimes the block of code is ```python ... ``` instead of ``` ... ```
        # in this case strip the python out

        if code.startswith("python"):
            code = code[len("python") :].strip()

        return code

    if is_valid_python(trimmed):
        return trimmed

    return None


class GenerateScriptConfig(Config):
    model: ClassVar = REQUIRED
    save_dir: ClassVar = REQUIRED

    num_workers: int = None
    gpus: int = None
    vllm_args: str = None
    vllm_port: int = 8000

    seed: int = 0
    limit: int = None
    offset: int = None
    stride: int = None

    num_few_shot: int = 2
    max_tokens: int = 1024
    stop_strings: list = []
    num_samples: int = 2
    batch_size: int = 2
    top_p: float = 0.95
    temperature: float = 0.6

    def finalize(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


class EvaluateScriptConfig(Config):
    samples_dir: Path = REQUIRED
    save_dir: Path = REQUIRED

    num_workers: int = 1

    offset: int = 0
    stride: int = 1
    limit: int = 100_000

    def finalize(self):
        self.samples_dir = Path(self.samples_dir)
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


class Timeout():
  """Timeout class using ALARM signal"""
  class Timeout(Exception): pass

  def __init__(self, sec):
    self.sec = sec

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.raise_timeout)
    signal.alarm(self.sec)

  def __exit__(self, *args):
    signal.alarm(0) # disable alarm

  def raise_timeout(self, *args):
    raise Timeout.Timeout()