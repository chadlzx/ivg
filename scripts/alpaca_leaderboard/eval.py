import os
from dataclasses import dataclass, field
from typing import Optional, Text

import tyro
import tqdm
import torch
from datasets import Dataset, load_dataset

from inference_time_alignment.utils import set_seeds
from scripts.alpaca_leaderboard.src.evaluator.base import EvaluationInput
from scripts.alpaca_leaderboard.src.utils import get_evaluator


@dataclass(kw_only=True)
class ScriptArguments:
    evaluator_name: Text = field(default="UltraRM-13b", metadata={"help": "Starling-RM-34B or UltraRM-13b"})
    generation_path: Text = field(metadata={"help": "output path *.jsonl"})
    evaluation_path: Text = field(default="tmp/alpaca_leaderboard/eval", metadata={"help": "output path *.jsonl"})
    overwrite: Optional[bool] = field(default=False, metadata={"help": "whether to overwrite evaluation_path"})
    seed: Optional[int] = field(default=1, metadata={"help": "optional distributed"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "generate on a small fraction of query dataset"})

script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.seed)

generation = load_dataset(script_args.generation_path, split="train")
if script_args.sanity_check:
    generation = generation.select(range(20))

evaluator = get_evaluator(script_args.evaluator_name)

results = []
with torch.no_grad():
    for sample in tqdm.tqdm(generation):
        score = evaluator.eval(EvaluationInput(prompt=sample["instruction"], response=sample["output"]))
        results.append({
            "instruction": sample["instruction"],
            "output": sample["output"],
            "score": score,
        })

# raw
dataset = Dataset.from_list(results)
dataset.to_json(os.path.join(script_args.evaluation_path, "raw.jsonl"))

# mean
scores = [result["score"] for result in results]
mean_score = sum(scores) / len(scores)
with open(os.path.join(script_args.evaluation_path, "mean.txt"), "w") as f:
    f.write(str(mean_score))
