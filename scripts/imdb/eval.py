import os
from dataclasses import dataclass, field
from typing import Optional, Text

import tyro
import tqdm
from datasets import Dataset, load_dataset
from transformers import pipeline

from inference_time_alignment.utils import set_seeds

@dataclass(kw_only=True)
class ScriptArguments:
    generation_path: Text = field(metadata={"help": "output path *.jsonl"})
    evaluation_path: Text = field(default="tmp/imdb/eval", metadata={"help": "output path *.jsonl"})
    overwrite: Optional[bool] = field(default=False, metadata={"help": "whether to overwrite evaluation_path"})
    seed: Optional[int] = field(default=1, metadata={"help": "optional distributed"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "generate on a small fraction of query dataset"})

script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.seed)

generation = load_dataset(script_args.generation_path, split="train")
if script_args.sanity_check:
    generation = generation.select(range(20))

rm = pipeline(model="lvwerra/distilbert-imdb", device=0, function_to_apply="none", return_all_scores=True, max_length=512)
rm.tokenizer.pad_token_id = rm.model.config.eos_token_id

results = []
for sample in tqdm.tqdm(generation):
    rm_output = rm(sample["prompt"] + sample["response"])[0]
    assert rm_output[1]["label"] == "POSITIVE"
    # log_p positive - log_p negative
    score = rm_output[1]["score"] - rm_output[0]["score"]
    results.append({
        "prompt": sample["prompt"],
        "response": sample["response"],
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
